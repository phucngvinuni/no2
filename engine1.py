# engine1.py
import torch
import math
import torch.nn as nn
import sys
import numpy as np
from typing import Iterable, Dict, Tuple, List, Optional # <<<< ADD Optional HERE
import os

from timm.utils import AverageMeter
import utils
from torchvision.utils import save_image
from torchvision import transforms

# ... (rest of your imports: cv2, torchmetrics) ...
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV (cv2) not found. YOLO detection visualizations will not be saved. `pip install opencv-python`")

try:
    from torchmetrics.detection import MeanAveragePrecision
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not found. mAP calculation will be skipped. `pip install torchmetrics`")

# ... (yolo_image_normalizer and other functions like evaluate_semcom_with_yolo, train_semcom_reconstruction_batch) ...

# engine1.py

# ... (imports) ...

@torch.no_grad()
def evaluate_semcom_with_yolo(
    semcom_net: torch.nn.Module,
    yolo_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    reconstruction_criterion: torch.nn.Module,
    args,
    current_epoch_num: any,
    viz_output_dir: str,
    print_freq=20,
    visualize_batches=1,
    visualize_images_per_batch=1
):
    semcom_net.eval()
    if yolo_model: yolo_model.eval()

    rec_loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    vq_loss_meter_eval = AverageMeter()

    # --- ENSURE THIS BLOCK IS CORRECT AND OUTSIDE/BEFORE THE LOOP ---
    map_calculator = None # Initialize to None
    if yolo_model and TORCHMETRICS_AVAILABLE:
        try:
            map_calculator = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
        except TypeError:
            print("Warning: 'respect_labels' arg not supported (older torchmetrics?). Initializing MeanAveragePrecision without it.")
            map_calculator = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
        
        if map_calculator: # Check if initialization was successful
            map_calculator = map_calculator.to(device)
    else: # Print reasons if not initializing
        if not yolo_model: print("INFO: YOLO model not provided for evaluation. mAP will be skipped.")
        if not TORCHMETRICS_AVAILABLE: print("INFO: Torchmetrics not available. mAP will be skipped.")
    # --- END ENSURE ---
    
    if viz_output_dir: os.makedirs(viz_output_dir, exist_ok=True) # Make sure this is also before the loop if used for paths inside
    print(f"\n--- Starting Evaluation Loop (Epoch/Run: {current_epoch_num}, Eval SNR: {args.snr_db_eval:.1f} dB) ---")


    for batch_idx, (semcom_data_input, targets_tuple) in enumerate(dataloader): # Loop starts here
        (original_imgs_tensor, bm_pos) = semcom_data_input
        (semcom_recon_target_gt, yolo_gt_targets_list_of_dicts) = targets_tuple

        original_imgs_tensor = original_imgs_tensor.to(device, non_blocking=True)
        if bm_pos is not None:
            bm_pos = bm_pos.to(device, non_blocking=True)
        semcom_recon_target_gt = semcom_recon_target_gt.to(device, non_blocking=True)

        yolo_gt_for_metric_device = []
        # This 'if' condition is where the NameError occurred
        if yolo_model and map_calculator: # Check if map_calculator was successfully initialized
            for gt_dict in yolo_gt_targets_list_of_dicts:
                yolo_gt_for_metric_device.append({
                    "boxes": gt_dict["boxes"].to(device),
                    "labels": gt_dict["labels"].to(device)
                })
        
        # ... (rest of the loop: semcom_net call, loss, psnr, ssim, visualizations, yolo preds, map_calculator.update) ...
        # ... (The content of the loop from the full engine1.py you last provided should be here) ...
        semcom_outputs_dict = semcom_net(img=original_imgs_tensor,bm_pos=bm_pos,_eval=True,eval_snr_db=args.snr_db_eval)
        reconstructed_image_batch = semcom_outputs_dict['reconstructed_image']
        pixel_wise_rec_loss = reconstruction_criterion(reconstructed_image_batch, semcom_recon_target_gt)
        scalar_rec_loss_no_vq = pixel_wise_rec_loss.mean()
        current_eval_vq_loss = 0.0
        if 'vq_loss' in semcom_outputs_dict and semcom_outputs_dict['vq_loss'] is not None:
            vq_loss_tensor = semcom_outputs_dict['vq_loss']
            current_eval_vq_loss = vq_loss_tensor.item()
        rec_loss_meter.update(scalar_rec_loss_no_vq.item(), original_imgs_tensor.size(0))
        vq_loss_meter_eval.update(current_eval_vq_loss, original_imgs_tensor.size(0))
        batch_psnr = utils.calc_psnr(reconstructed_image_batch.detach().cpu(), semcom_recon_target_gt.detach().cpu())
        batch_ssim = utils.calc_ssim(reconstructed_image_batch.detach().cpu(), semcom_recon_target_gt.detach().cpu())
        psnr_meter.update(np.mean(batch_psnr) if isinstance(batch_psnr, list) and batch_psnr else 0.0, original_imgs_tensor.size(0))
        ssim_meter.update(np.mean(batch_ssim) if isinstance(batch_ssim, list) and batch_ssim else 0.0, original_imgs_tensor.size(0))
        if batch_idx < visualize_batches and viz_output_dir:
            for i in range(min(reconstructed_image_batch.size(0), visualize_images_per_batch)):
                save_image(reconstructed_image_batch[i].cpu(), os.path.join(viz_output_dir, f"ep{current_epoch_num}_b{batch_idx}_i{i}_snr{args.snr_db_eval:.0f}dB_RECON.png"))
                save_image(original_imgs_tensor[i].cpu(), os.path.join(viz_output_dir, f"ep{current_epoch_num}_b{batch_idx}_i{i}_ORIG.png"))
        if yolo_model and map_calculator:
            if batch_idx < visualize_batches and viz_output_dir and CV2_AVAILABLE:
                try:
                    yolo_input_orig = original_imgs_tensor.detach(); yolo_results_orig = yolo_model(yolo_input_orig, verbose=False, conf=args.yolo_conf_thres, iou=args.yolo_iou_thres)
                    for i in range(min(len(yolo_results_orig), visualize_images_per_batch)):
                        if hasattr(yolo_results_orig[i], 'plot'): cv2.imwrite(os.path.join(viz_output_dir, f"ep{current_epoch_num}_b{batch_idx}_i{i}_YOLO_ON_ORIG.png"), yolo_results_orig[i].plot())
                except Exception as e: print(f"    Error YOLO on Original: {e}")
            try: yolo_input_recon = reconstructed_image_batch.detach(); yolo_results_list_recon = yolo_model(yolo_input_recon, verbose=False, conf=args.yolo_conf_thres, iou=args.yolo_iou_thres)
            except Exception as e: yolo_results_list_recon = []; print(f"    Error YOLO on Reconstructed: {e}")
            yolo_preds_for_metric = []
            for i in range(len(yolo_results_list_recon)):
                result_r = yolo_results_list_recon[i]
                if result_r.boxes is not None:
                    yolo_preds_for_metric.append({"boxes": result_r.boxes.xyxy.to(device), "scores": result_r.boxes.conf.to(device), "labels": result_r.boxes.cls.to(torch.int64).to(device)})
                    if batch_idx < visualize_batches and i < visualize_images_per_batch and viz_output_dir and CV2_AVAILABLE:
                        if hasattr(result_r, 'plot'): cv2.imwrite(os.path.join(viz_output_dir, f"ep{current_epoch_num}_b{batch_idx}_i{i}_snr{args.snr_db_eval:.0f}dB_YOLO_ON_RECON.png"), result_r.plot())
                else: yolo_preds_for_metric.append({"boxes": torch.empty((0,4),dtype=torch.float32,device=device), "scores": torch.empty(0,dtype=torch.float32,device=device), "labels": torch.empty(0,dtype=torch.int64,device=device)})
            if yolo_preds_for_metric and len(yolo_preds_for_metric) == len(yolo_gt_for_metric_device):
                try: map_calculator.update(yolo_preds_for_metric, yolo_gt_for_metric_device)
                except Exception as e: print(f"    Error updating mAP calculator: {e}")
        if (batch_idx + 1) % print_freq == 0 or (batch_idx + 1) == len(dataloader):
            print(f'Test Batch {batch_idx+1}/{len(dataloader)}: [RecLoss: {rec_loss_meter.avg:.4f}] [VQLoss: {vq_loss_meter_eval.avg:.4f}] [PSNR: {psnr_meter.avg:.2f}] [SSIM: {ssim_meter.avg:.4f}]')
            sys.stdout.flush()


    final_stats = { # ... (final_stats dictionary as before) ...
        'rec_loss': rec_loss_meter.avg, 'vq_loss': vq_loss_meter_eval.avg,
        'psnr': psnr_meter.avg, 'ssim': ssim_meter.avg,
    }
    if map_calculator:
        try:
            final_map_results = map_calculator.compute()
            print(f"\nFinal mAP results (from torchmetrics) for SNR {args.snr_db_eval:.1f} dB (Epoch/Run: {current_epoch_num}):")
            for k, v_tensor in final_map_results.items():
                v_item = v_tensor.item() if isinstance(v_tensor, torch.Tensor) else float(v_tensor)
                print(f"  {k}: {v_item:.4f}")
                safe_key = str(k).replace("map_per_class","map_cls").replace("(","_").replace(")","").replace("[","").replace("]","").replace(" ","_").replace("'","").replace(":","_")
                final_stats[safe_key] = v_item
            map_calculator.reset()
        except Exception as e: print(f"Error computing final mAP: {e}"); final_stats['map'] = 0.0
            
    print("--- Finished Evaluation Loop ---")
    sys.stdout.flush()
    return final_stats

# ... (train_semcom_reconstruction_batch and train_epoch_semcom_reconstruction as they were in the last full correct version) ...
# Ensure they are correctly defined below. For brevity, I am not repeating them here if they were correct in the previous response.
# The version of train_epoch_semcom_reconstruction from the response that fixed the max_norm TypeError
# and train_semcom_reconstruction_batch that correctly calls model.forward with train_snr_db_min/max
# should be used.

def train_semcom_reconstruction_batch(
    model: torch.nn.Module,
    input_samples_for_semcom: torch.Tensor,
    original_images_for_loss: torch.Tensor,
    yolo_gt_for_this_batch: List[Dict[str, torch.Tensor]],
    bm_pos: torch.Tensor,
    base_reconstruction_criterion: torch.nn.Module,
    args
) -> Tuple[torch.Tensor, torch.Tensor, float]:

    outputs_dict = model(
        img=input_samples_for_semcom, bm_pos=bm_pos, targets=None, _eval=False,
        train_snr_db_min=args.snr_db_train_min, train_snr_db_max=args.snr_db_train_max
    )
    reconstructed_image_batch = outputs_dict['reconstructed_image']

    batch_size = reconstructed_image_batch.size(0)
    current_device = reconstructed_image_batch.device
    total_weighted_loss = torch.tensor(0.0, device=current_device)

    for i in range(batch_size):
        reconstructed_img_single = reconstructed_image_batch[i]
        original_img_single = original_images_for_loss[i]
        gt_boxes_single_img_abs = yolo_gt_for_this_batch[i]['boxes']

        weight_map_2d = utils.create_bbox_weight_map(
            image_shape=reconstructed_img_single.shape[1:],
            gt_boxes_xyxy=gt_boxes_single_img_abs,
            inside_box_weight=args.inside_box_loss_weight,
            outside_box_weight=args.outside_box_loss_weight,
            target_device=current_device
        )
        pixel_wise_loss = base_reconstruction_criterion(reconstructed_img_single, original_img_single)
        weighted_pixel_loss = pixel_wise_loss * weight_map_2d.unsqueeze(0)
        total_weighted_loss += weighted_pixel_loss.mean()

    final_reconstruction_loss = total_weighted_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=current_device) # Avoid div by zero
    loss = final_reconstruction_loss
    
    current_vq_loss_val = 0.0
    if 'vq_loss' in outputs_dict and outputs_dict['vq_loss'] is not None:
        vq_loss_tensor = outputs_dict['vq_loss']
        if isinstance(vq_loss_tensor, torch.Tensor):
            loss += args.vq_loss_weight * vq_loss_tensor
            current_vq_loss_val = vq_loss_tensor.item()
            
    return loss, reconstructed_image_batch, current_vq_loss_val


def train_epoch_semcom_reconstruction(
    model: torch.nn.Module, criterion: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer, device: torch.device, epoch: int,
    loss_scaler: Optional[torch.cuda.amp.GradScaler], 
    args, max_norm: Optional[float] = None,           
    start_steps=0, lr_schedule_values=None, wd_schedule_values=None,
    update_freq=1, print_freq=50
):
    model.train(True)
    rec_loss_meter = AverageMeter(); psnr_meter = AverageMeter()
    ssim_meter = AverageMeter(); vq_loss_meter = AverageMeter()

    if update_freq > 1 : optimizer.zero_grad()

    for data_iter_step, (semcom_data_input, targets_tuple) in enumerate(data_loader):
        effective_iter = start_steps + (data_iter_step // update_freq)

        if (data_iter_step % update_freq == 0):
            if lr_schedule_values is not None and effective_iter < len(lr_schedule_values):
                for param_group in optimizer.param_groups: param_group["lr"] = lr_schedule_values[effective_iter] * param_group.get("lr_scale", 1.0)
            if wd_schedule_values is not None and effective_iter < len(wd_schedule_values):
                for param_group in optimizer.param_groups:
                    if param_group.get("weight_decay", 0.0) > 0: param_group["weight_decay"] = wd_schedule_values[effective_iter]

        (original_images, bm_pos) = semcom_data_input
        (original_images_for_loss, yolo_gt_targets_list) = targets_tuple
        
        original_images = original_images.to(device, non_blocking=True)
        original_images_for_loss = original_images_for_loss.to(device, non_blocking=True)
        if bm_pos is not None: bm_pos = bm_pos.to(device, non_blocking=True)
        
        yolo_gt_on_device = []
        for gt_dict in yolo_gt_targets_list:
            yolo_gt_on_device.append({ "boxes": gt_dict["boxes"].to(device), "labels": gt_dict["labels"].to(device) })

        samples_for_semcom_input = original_images
        
        with torch.amp.autocast(device_type=args.device, enabled=(loss_scaler is not None)):
            loss, reconstructed_batch, current_vq_loss_val = train_semcom_reconstruction_batch(
                model, samples_for_semcom_input, original_images_for_loss,
                yolo_gt_on_device, bm_pos, criterion, args
            )
        
        loss_value = loss.item(); main_rec_loss_value = loss_value - current_vq_loss_val
        if not math.isfinite(loss_value): print(f"Loss is {loss_value}, stopping training"); sys.exit(1)

        loss_for_backward = loss / update_freq
        clip_grad_val_for_scaler = max_norm if max_norm is not None and max_norm > 0 else None

        if loss_scaler is not None:
            loss_scaler(loss_for_backward, optimizer, parameters=model.parameters(), clip_grad=clip_grad_val_for_scaler, update_grad=((data_iter_step + 1) % update_freq == 0))
        else:
            loss_for_backward.backward()
            if (data_iter_step + 1) % update_freq == 0:
                if clip_grad_val_for_scaler: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_val_for_scaler)
                optimizer.step(); optimizer.zero_grad()

        if device == 'cuda': torch.cuda.synchronize()

        rec_loss_meter.update(main_rec_loss_value, original_images.size(0))
        vq_loss_meter.update(current_vq_loss_val, original_images.size(0))
        
        batch_psnr_train = utils.calc_psnr(reconstructed_batch.detach().cpu(), original_images_for_loss.detach().cpu())
        batch_ssim_train = utils.calc_ssim(reconstructed_batch.detach().cpu(), original_images_for_loss.detach().cpu())
        psnr_meter.update(np.mean(batch_psnr_train) if isinstance(batch_psnr_train, list) and batch_psnr_train else 0.0, original_images.size(0))
        ssim_meter.update(np.mean(batch_ssim_train) if isinstance(batch_ssim_train, list) and batch_ssim_train else 0.0, original_images.size(0))

        if (data_iter_step +1) % print_freq == 0 or (data_iter_step + 1) == len(data_loader) :
            lr = optimizer.param_groups[0]["lr"]
            print(f'Epoch:[{epoch}] Iter:[{data_iter_step+1}/{len(data_loader)}] '
                  f'RecLoss: {rec_loss_meter.avg:.4f} VQLoss: {vq_loss_meter.avg:.4f} '
                  f'PSNR: {psnr_meter.avg:.2f} SSIM: {ssim_meter.avg:.4f} '
                  f'LR: {lr:.2e} GradAccum: {((data_iter_step % update_freq) + 1)}/{update_freq}')
            sys.stdout.flush()

    train_stat = {'loss': rec_loss_meter.avg, 'vq_loss': vq_loss_meter.avg, 'psnr': psnr_meter.avg, 'ssim': ssim_meter.avg}
    print(f"--- Finished Training Epoch {epoch} ---")
    sys.stdout.flush()
    return train_stat