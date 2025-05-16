import argparse

IMGC_NUMCLASS = 1   # For the OBJECT DETECTION task (e.g., 1 for 'fish')

def get_args():
    parser = argparse.ArgumentParser('SemCom for Reconstruction + YOLO Eval', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int) # Potentially smaller due to increased model size
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--save_freq', default=10, type=int) # Save less frequently if epochs are long
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--chep', default='', type=str, help='checkpoint path')

    # Dataset parameters
    parser.add_argument('--mask_ratio', default=0.0, type=float,
                        help='ratio of visual tokens/patches for SemCom encoder (0.0 for full recon)')
    parser.add_argument('--data_path', default='../yolo_fish_dataset_root/', type=str,
                        help='Root dataset path (containing train/, valid/, test/)')
    parser.add_argument('--data_set', default='fish', choices=['cifar_S32','cifar_S224', 'imagenet','fish'], type=str)
    parser.add_argument('--input_size', default=640, type=int, help='Image pixel H, W for SemCom & YOLO')
    parser.add_argument('--drop_path_rate', type=float, default=0.1, metavar='PCT', # Renamed from drop_path for clarity
                        help='Drop path rate (stochastic depth) (default: 0.1)')
    parser.add_argument('--drop_rate', type=float, default=0.0, metavar='PCT', # For MLP/Attention dropout
                        help='Dropout rate for MLP/Attention layers (default: 0.0)')
    parser.add_argument('--num_object_classes', default=1, type=int,
                        help='Number of object classes for detection task (e.g., 1 for fish)')

    # Channel parameters
    parser.add_argument('--channel_type', default='rayleigh', choices=['none', 'awgn', 'rayleigh', 'rician'], type=str)
    parser.add_argument('--snr_db_train_min', default=30.0, type=float, help='Min SNR dB for training (start high)')
    parser.add_argument('--snr_db_train_max', default=30.0, type=float, help='Max SNR dB for training (start high)')
    parser.add_argument('--snr_db_eval', default=30.0, type=float, help='Fixed SNR dB for evaluation')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+') # AdamW defaults
    parser.add_argument('--clip_grad', type=float, default=1.0, metavar='NORM', help='Clip gradient norm (e.g., 1.0)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay (e.g., 0.05 for AdamW)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="Final value of WD.")
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N') # Longer warmup
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N')

    # Model parameters
    parser.add_argument('--model', default='ViT_Reconstruction_Model_Default_CNNHead', type=str, metavar='MODEL')
    parser.add_argument('--patch_size', default=32, type=int, help="Patch size for ViT")
    parser.add_argument('--encoder_embed_dim', default=512, type=int)
    parser.add_argument('--encoder_depth', default=8, type=int)
    parser.add_argument('--encoder_num_heads', default=8, type=int)
    parser.add_argument('--decoder_embed_dim', default=512, type=int) # Input to ViT blocks in decoder
    parser.add_argument('--decoder_depth', default=8, type=int)       # ViT blocks in decoder
    parser.add_argument('--decoder_num_heads', default=8, type=int)
    parser.add_argument('--quantizer_dim', default=512, type=int)
    parser.add_argument('--bits_for_quantizer', default=8, type=int)


    # Loss Weight Parameters
    parser.add_argument('--inside_box_loss_weight', default=5.0, type=float,
                        help='Weight for reconstruction loss inside bounding boxes.')
    parser.add_argument('--outside_box_loss_weight', default=1.0, type=float,
                        help='Weight for reconstruction loss outside bounding boxes.')
    parser.add_argument('--vq_loss_weight', default=0.25, type=float,
                        help='Weight for the VQ quantization loss component.')
    # Add perceptual_loss_weight if you implement LPIPS
    # parser.add_argument('--perceptual_loss_weight', default=0.1, type=float)


    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--output_dir', default='ckpt_recon_cnnhead_weightedloss', type=str)
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # ... (other args: auto_resume, start_epoch, eval, num_workers, pin_mem, distributed, save_ckpt)
    parser.add_argument('--auto_resume', action='store_true'); parser.set_defaults(auto_resume=False)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true'); parser.set_defaults(pin_mem=True)
    parser.add_argument('--world_size', default=1, type=int); parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true'); parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--dist_eval', action='store_true', default=False)
    parser.add_argument('--save_ckpt', action='store_true'); parser.set_defaults(save_ckpt=True)


    # YOLO specific args
    parser.add_argument('--yolo_weights', default='best.pt', type=str)
    parser.add_argument('--yolo_conf_thres', default=0.01, type=float, help='Lower for debugging mAP=0')
    parser.add_argument('--yolo_iou_thres', default=0.45, type=float)
    parser.add_argument('--eval_viz_output_dir', default='eval_visualizations_cnnhead', type=str)

    return parser.parse_args()