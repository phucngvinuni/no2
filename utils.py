# utils.py
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Iterable # Ensure these are imported
import math
from pathlib import Path
import os
import io
import json
# from timm.utils import get_state_dict # Already available via utils.get_state_dict
# from timm.models import create_model # Already available via utils.create_model
from collections import OrderedDict
from pytorch_msssim import ssim # ms_ssim removed for brevity if not used

# --- NativeScalerWithGradNormCount (from previous correct version) ---
class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"
    def __init__(self):
        self._scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    def __call__(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer,
                 parameters: Iterable[torch.nn.Parameter], clip_grad: float = None,
                 update_grad: bool = True, create_graph: bool = False):
        if self._scaler.is_enabled(): self._scaler.scale(loss).backward(create_graph=create_graph)
        else: loss.backward(create_graph=create_graph)
        norm = None
        if update_grad:
            if self._scaler.is_enabled(): self._scaler.unscale_(optimizer)
            if clip_grad is not None and clip_grad > 0: norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            if self._scaler.is_enabled(): self._scaler.step(optimizer); self._scaler.update()
            else: optimizer.step()
            optimizer.zero_grad()
        return norm
    def state_dict(self): return self._scaler.state_dict()
    def load_state_dict(self, state_dict): self._scaler.load_state_dict(state_dict)

def sel_criterion(args):
    # For weighted loss in training, the base criterion needs reduction='none'
    # For evaluation, we'll take the mean of this per-pixel loss.
    criterion = nn.L1Loss(reduction='none')
    print(f"Base Reconstruction Criterion (for per-pixel loss) = {str(criterion)}")
    return criterion

def get_model(args): # Ensure this passes relevant args to your registration function
    print(f"Creating model: {args.model} with input_size: {args.input_size}")
    from timm.models import create_model as timm_create_model # Local import to avoid circulars if utils is imported by model.py too early

    model_kwargs = {
        'input_size': args.input_size,
        'img_size': args.input_size, # Some models might use this
        'patch_size': args.patch_size,
        'encoder_embed_dim': args.encoder_embed_dim,
        'encoder_depth': args.encoder_depth,
        'encoder_num_heads': args.encoder_num_heads,
        'decoder_embed_dim': args.decoder_embed_dim,
        'decoder_depth': args.decoder_depth,
        'decoder_num_heads': args.decoder_num_heads,
        'quantizer_dim': args.quantizer_dim,
        'bits_for_quantizer': args.bits_for_quantizer,
        'drop_rate': args.drop_rate,
        'drop_path_rate': args.drop_path_rate,
        # Add other model-specific args from base_args if needed
    }
    print(f"  kwargs for timm.create_model: {model_kwargs}")

    model = timm_create_model(
        args.model, # Name of your registered model
        pretrained=False,
        **model_kwargs
    )
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'=> Number of params: {n_parameters / 1e6:.2f} M')
    return model

def create_bbox_weight_map(image_shape: Tuple[int, int],
                           gt_boxes_xyxy: torch.Tensor,
                           inside_box_weight: float = 5.0,
                           outside_box_weight: float = 1.0,
                           target_device: torch.device = torch.device('cpu')
                           ) -> torch.Tensor:
    H, W = image_shape
    weight_map = torch.full((H, W), outside_box_weight, dtype=torch.float32, device=target_device)
    if gt_boxes_xyxy.numel() > 0:
        gt_boxes_on_device = gt_boxes_xyxy.to(target_device)
        for box in gt_boxes_on_device:
            x1, y1, x2, y2 = box.long()
            x1_c = torch.clamp(x1, 0, W) # Slice end is exclusive, so W is okay
            y1_c = torch.clamp(y1, 0, H)
            x2_c = torch.clamp(x2, x1_c, W)
            y2_c = torch.clamp(y2, y1_c, H)
            if x2_c > x1_c and y2_c > y1_c:
                weight_map[y1_c:y2_c, x1_c:x2_c] = inside_box_weight # Slicing is [start:end-1]
    return weight_map

# --- (Keep other utility functions: as_img_array, calc_psnr, calc_ssim, save_model, cosine_scheduler etc.
#      from the "full code for utils.py" I provided in the response that fixed the NativeScaler and clip_grad.) ---
# For brevity, I'm not repeating all of them here, but ensure they are present and correct.

def as_img_array(image: torch.Tensor) -> torch.Tensor:
    if image.dtype != torch.float32: image = image.float()
    image = torch.clamp(image * 255.0, 0, 255)
    return torch.round(image)

def calc_psnr(predictions: torch.Tensor, targets: torch.Tensor) -> list:
    if predictions.ndim == 3: predictions = predictions.unsqueeze(0)
    if targets.ndim == 3: targets = targets.unsqueeze(0)
    pred_arr = as_img_array(predictions.float())
    targ_arr = as_img_array(targets.float())
    mse = torch.mean((pred_arr - targ_arr) ** 2.0, dim=(1, 2, 3))
    psnr_val = torch.where(mse == 0, torch.tensor(100.0, device=mse.device, dtype=torch.float32), 20 * torch.log10(255.0 / torch.sqrt(mse.clamp(min=1e-8)))) # clamp mse for stability
    return psnr_val.tolist()

def calc_ssim(predictions: torch.Tensor, targets: torch.Tensor) -> list:
    if predictions.ndim == 3: predictions = predictions.unsqueeze(0)
    if targets.ndim == 3: targets = targets.unsqueeze(0)
    pred_for_ssim = as_img_array(predictions.float())
    targ_for_ssim = as_img_array(targets.float())
    try:
        ssim_val = ssim(pred_for_ssim, targ_for_ssim, data_range=255.0, size_average=False, nonnegative_ssim=True)
    except RuntimeError as e:
        print(f"RuntimeError during SSIM calculation: {e}"); ssim_val = torch.zeros(predictions.shape[0], device=predictions.device)
    return ssim_val.tolist()

def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    train_type_str = getattr(args, 'train_type', 'default_train') # Handle if train_type is missing
    output_dir = output_dir / f'ckpt_{train_type_str}' # Keep subfolder if train_type is used
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_name = str(epoch)
    checkpoint_path = output_dir / f'checkpoint-{epoch_name}.pth'
    to_save = {'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args}
    if loss_scaler is not None: to_save['scaler'] = loss_scaler.state_dict()
    if model_ema is not None: to_save['model_ema'] = utils.get_state_dict(model_ema) # Assuming utils.get_state_dict
    torch.save(to_save, checkpoint_path)
    return checkpoint_path

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0: warmup_iters = warmup_steps
    if niter_per_ep == 0 and warmup_epochs > 0 : # Handle case with very small dataset
        print(f"Warning: niter_per_ep is 0, but warmup_epochs is {warmup_epochs}. Setting warmup_iters to 0.")
        warmup_iters = 0
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_iters > 0: warmup_schedule = np.linspace(start_warmup_value, base_value, int(warmup_iters)) # Ensure integer
    
    total_iters = epochs * niter_per_ep
    if total_iters < warmup_iters: # Handle cases where total_iters is less than warmup_iters
        print(f"Warning: Total iterations ({total_iters}) < warmup iterations ({warmup_iters}). Adjusting schedule.")
        if total_iters > 0 :
            return np.linspace(start_warmup_value, final_value, int(total_iters)) # simple linspace to final_value
        else:
            return np.array([base_value]) # single point

    main_iters = np.arange(total_iters - warmup_iters)
    if len(main_iters) == 0: # Only warmup phase
        schedule = warmup_schedule
    else:
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(math.pi * main_iters / (len(main_iters) -1 if len(main_iters) > 1 else 1 ))) # Avoid div by zero
        schedule = np.concatenate((warmup_schedule, schedule))
    
    if len(schedule) != total_iters and total_iters > 0: # Pad if miscalculation due to rounding
        print(f"Scheduler length mismatch: {len(schedule)} vs {total_iters}. Padding/truncating.")
        if len(schedule) < total_iters:
            schedule = np.pad(schedule, (0, int(total_iters - len(schedule))), 'edge')
        else:
            schedule = schedule[:int(total_iters)]
    elif total_iters == 0:
        return np.array([base_value])

    return schedule