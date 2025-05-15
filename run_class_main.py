# run_class_main.py
import sys
import os

# --- Ensure local modules are prioritized ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
# print(f"--- INFO: Project directory '{CURRENT_DIR}' prepended to sys.path. ---") # Optional: Keep for sanity check

import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from ultralytics import YOLO
from pathlib import Path

import utils # Assuming utils.py contains necessary functions
import engine1 # engine1.py
from base_args import get_args
import datasets # datasets.py
import optim_factory # optim_factory.py
import model # model.py

# Alias for convenience, assuming NativeScalerWithGradNormCount is in utils
NativeScaler = utils.NativeScalerWithGradNormCount


def seed_initial(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        if torch.cuda.device_count() > 1:
             torch.cuda.manual_seed_all(seed)
        # For full reproducibility, but can slow down training significantly.
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
    print(f"Seed set to {seed}")


def main(args):
    args.distributed = False # For single GPU focus in this script
    device = torch.device(args.device)
    seed_initial(seed=args.seed)

    print(f"Creating SemCom model: {args.model}")
    semcom_model = utils.get_model(args) # utils.get_model should handle args to model's __init__

    if args.resume:
        print(f"Resuming SemCom model from: {args.resume}")
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            state_dict = checkpoint.get('model',
                                checkpoint.get('state_dict',
                                checkpoint.get('model_state_dict', checkpoint)))

            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            missing_keys, unexpected_keys = semcom_model.load_state_dict(new_state_dict, strict=False)
            if missing_keys: print(f"  Warning: Missing keys in loaded SemCom state_dict: {missing_keys}")
            if unexpected_keys: print(f"  Warning: Unexpected keys in loaded SemCom state_dict: {unexpected_keys}")
            
            if 'epoch' in checkpoint and args.start_epoch == 0:
                args.start_epoch = checkpoint['epoch'] + 1
                print(f"  Resuming training from epoch {args.start_epoch}")
            if 'optimizer' in checkpoint and hasattr(optimizer, 'load_state_dict'): # Check if optimizer is defined
                 print("  Loading optimizer state...")
                 optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scaler' in checkpoint and loss_scaler is not None: # Check if loss_scaler is defined
                 print("  Loading loss scaler state...")
                 loss_scaler.load_state_dict(checkpoint['scaler'])
        else:
            print(f"  Warning: Resume checkpoint not found at {args.resume}. Starting from scratch.")


    if hasattr(semcom_model, 'img_encoder') and hasattr(semcom_model.img_encoder, 'patch_embed'):
        patch_size_h = semcom_model.img_encoder.patch_embed.patch_size[0]
        patch_size_w = semcom_model.img_encoder.patch_embed.patch_size[1]
        args.window_size = (args.input_size // patch_size_h, args.input_size // patch_size_w)
    else:
        args.window_size = (args.input_size // 16, args.input_size // 16) # Fallback
    print(f"SemCom Patch Grid (window_size derived): {args.window_size} for input_size {args.input_size}")

    semcom_model.to(device)
    semcom_model_without_ddp = semcom_model

    yolo_model = None
    if args.yolo_weights and os.path.exists(args.yolo_weights):
        print(f"Loading YOLO model from: {args.yolo_weights}")
        try:
            yolo_model = YOLO(args.yolo_weights)
            if hasattr(yolo_model, 'fuse'): yolo_model.fuse() 
            if hasattr(yolo_model, 'model') and hasattr(yolo_model.model, 'eval'): yolo_model.model.eval()
            elif hasattr(yolo_model, 'eval'): yolo_model.eval()
            # yolo_model.to(device) # Ultralytics models typically handle device during inference call
            print("YOLO model loaded and set to eval mode.")
        except Exception as e:
            print(f"  Error loading YOLO model: {e}. Proceeding without YOLO evaluation.")
            yolo_model = None
    else:
        print(f"YOLO weights not found at '{args.yolo_weights}' or not specified. No YOLO evaluation.")

    print("Building train dataset...")
    trainset = datasets.build_dataset(is_train=True, args=args)
    print("Building validation dataset...")
    valset = datasets.build_dataset(is_train=False, args=args)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_mem,
        drop_last=True, collate_fn=datasets.yolo_collate_fn
    )
    dataloader_val = None
    if valset and len(valset) > 0:
        dataloader_val = torch.utils.data.DataLoader(
            valset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=args.pin_mem,
            drop_last=False, collate_fn=datasets.yolo_collate_fn
        )
    else:
        print("Validation dataset is empty or not created. Skipping validation dataloader.")


    print("Creating optimizer...")
    optimizer = optim_factory.create_optimizer(args, semcom_model_without_ddp)
    loss_scaler = NativeScaler() if args.device == 'cuda' else None # Instantiated after optimizer
    # If resuming and optimizer/scaler states were loaded, they are now applied to these instances.


    num_training_steps_per_epoch = len(trainloader) // args.update_freq
    if num_training_steps_per_epoch == 0 and len(trainloader) > 0 :
        num_training_steps_per_epoch = 1
        print(f"Warning - num_training_steps_per_epoch is 0, setting to 1. Dataloader len: {len(trainloader)}")
    elif len(trainloader) == 0:
        print("ERROR - Training dataloader is empty! Check dataset path and filtering.")
        exit()
    print(f"Number of training steps per epoch: {num_training_steps_per_epoch}")


    lr_schedule_values = utils.cosine_scheduler(args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch, warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,)
    wd_schedule_values = None
    if args.weight_decay > 0:
        wd_schedule_values = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end if args.weight_decay_end is not None else args.weight_decay, args.epochs, num_training_steps_per_epoch)
        if wd_schedule_values is not None: print(f"  Weight Decay Schedule: Max WD = {max(wd_schedule_values):.7f}, Min WD = {min(wd_schedule_values):.7f}")

    reconstruction_criterion = utils.sel_criterion(args).to(device)
    
    if args.eval:
        print("Evaluation mode selected (args.eval=True)")
        eval_viz_dir_path = Path(args.eval_viz_output_dir) / "final_eval_run"
        eval_viz_dir_path.mkdir(parents=True, exist_ok=True)

        if dataloader_val is None: print("  Validation dataloader is not available. Skipping evaluation."); exit(0)
        
        print("  Starting evaluation only run...")
        test_stats = engine1.evaluate_semcom_with_yolo(
            semcom_net=semcom_model, yolo_model=yolo_model, dataloader=dataloader_val,
            device=device, reconstruction_criterion=reconstruction_criterion, args=args,
            current_epoch_num="final_eval",
            viz_output_dir=str(eval_viz_dir_path)
        )
        print(f"\n--- Final Evaluation Results on Reconstructed Images (SNR: {args.snr_db_eval} dB) ---")
        print(f"  Reconstruction Loss: {test_stats.get('rec_loss', float('nan')):.4f}")
        print(f"  VQ Loss: {test_stats.get('vq_loss', float('nan')):.4f}")
        print(f"  PSNR: {test_stats.get('psnr', float('nan')):.2f} dB")
        print(f"  SSIM: {test_stats.get('ssim', float('nan')):.4f}")
        if yolo_model and engine1.TORCHMETRICS_AVAILABLE: # Check if TORCHMETRICS_AVAILABLE
            print(f"  Object Detection mAP: {test_stats.get('map', 0.0):.4f}")
            print(f"  Object Detection mAP@50: {test_stats.get('map_50', 0.0):.4f}")
            print(f"  Object Detection mAP@75: {test_stats.get('map_75', 0.0):.4f}")
        print("-------------------------------------------------------------------------")
        sys.stdout.flush()
        exit(0)

    print(f"Start training SemCom for image reconstruction for {args.epochs} epochs")
    max_psnr_eval = 0.0
    best_map_eval = 0.0 # Using mAP@0.50 for tracking best detection model

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        print(f"\n--- Starting Training Epoch {epoch}/{args.epochs-1} ---")
        semcom_model.train()
        train_stats = engine1.train_epoch_semcom_reconstruction(
            model=semcom_model, criterion=reconstruction_criterion,
            data_loader=trainloader, optimizer=optimizer, device=device, epoch=epoch,
            loss_scaler=loss_scaler, args=args, max_norm=args.clip_grad,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            update_freq=args.update_freq, print_freq=args.save_freq
        )

        save_path_this_epoch = None
        if args.output_dir and args.save_ckpt:
            # Save checkpoint at specified frequency or at the last epoch
            if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
                save_path_this_epoch = utils.save_model(
                    args=args, model=semcom_model, model_without_ddp=semcom_model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
                )
                if save_path_this_epoch : print(f"  Checkpoint saved to {save_path_this_epoch}")


        # Perform evaluation periodically
        if dataloader_val is not None and ((epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs):
            print(f"\n--- Evaluating at end of Epoch {epoch} (Eval SNR: {args.snr_db_eval} dB) ---")
            
            current_epoch_viz_dir_name = f"epoch_{epoch}"
            current_epoch_viz_dir_path = Path(args.eval_viz_output_dir) / current_epoch_viz_dir_name
            current_epoch_viz_dir_path.mkdir(parents=True, exist_ok=True)
            
            eval_stats = engine1.evaluate_semcom_with_yolo(
                semcom_net=semcom_model, yolo_model=yolo_model, dataloader=dataloader_val,
                device=device, reconstruction_criterion=reconstruction_criterion, args=args,
                current_epoch_num=epoch,
                viz_output_dir=str(current_epoch_viz_dir_path)
            )
            
            print(f"  Epoch {epoch} Eval: PSNR: {eval_stats.get('psnr',0.0):.2f}, SSIM: {eval_stats.get('ssim',0.0):.4f}", end="")
            current_map_50_value = eval_stats.get('map_50', 0.0) # Use mAP@0.50 for tracking

            if yolo_model and engine1.TORCHMETRICS_AVAILABLE:
                print(f", mAP: {eval_stats.get('map',0.0):.4f}, mAP@50: {current_map_50_value:.4f}")
            else:
                print("")

            current_psnr_value = eval_stats.get('psnr', 0.0)
            if current_psnr_value > max_psnr_eval:
                max_psnr_eval = current_psnr_value
                print(f"  *** New best PSNR on eval: {max_psnr_eval:.2f} at epoch {epoch} ***")
                if args.output_dir and args.save_ckpt: # Save best model based on PSNR
                    utils.save_model(args=args, model=semcom_model, model_without_ddp=semcom_model_without_ddp,
                               optimizer=optimizer, loss_scaler=loss_scaler, epoch="best_psnr")
            
            if yolo_model and engine1.TORCHMETRICS_AVAILABLE and current_map_50_value > best_map_eval:
                best_map_eval = current_map_50_value
                print(f"  *** New best mAP@50 on eval: {best_map_eval:.4f} at epoch {epoch} ***")
                if args.output_dir and args.save_ckpt: # Save best model based on mAP
                     utils.save_model(args=args, model=semcom_model, model_without_ddp=semcom_model_without_ddp,
                                 optimizer=optimizer, loss_scaler=loss_scaler, epoch="best_map")
            print("-------------------------------------------------------------------\n")
            sys.stdout.flush()


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Total Training time for SemCom: {total_time_str}')
    sys.stdout.flush()

if __name__ == '__main__':
    # print("--- DEBUG MAIN: Script execution started (__name__ == '__main__') ---") # Optional
    opts = get_args()
    # print(f"--- DEBUG MAIN: Parsed arguments: {opts} ---") # Optional

    if opts.output_dir: Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    if hasattr(opts, 'eval_viz_output_dir') and opts.eval_viz_output_dir :
        Path(opts.eval_viz_output_dir).mkdir(parents=True, exist_ok=True)
    
    # print("--- DEBUG MAIN: Calling main(opts) ---") # Optional
    main(opts)
    print("--- Script execution finished. ---") # Changed final debug print