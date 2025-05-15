import argparse

IMGC_NUMCLASS = 1   # For the OBJECT DETECTION task (e.g., 1 for 'fish' if it's the only class)
                    # This will be used by torchmetrics for mAP calculation.
                    # If your SemCom has an auxiliary FIM classifier, it would use this too.

def get_args():
    parser = argparse.ArgumentParser('SemCom for Reconstruction + YOLO Eval', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int) # May need to be smaller for reconstruction
    parser.add_argument('--epochs', default=50, type=int)    # Reconstruction might need more epochs
    parser.add_argument('--save_freq', default=5, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--chep', default='', type=str,
                        help='chceckpint path')

    # Dataset parameters
    parser.add_argument('--mask_ratio', default=0.0, type=float, # For full recon by encoder, set to 0.0
                        help='ratio of the visual tokens/patches need be masked by SemCom encoder')
    parser.add_argument('--data_path', default='../yolo_fish_dataset_root/', type=str, # POINT THIS TO THE ROOT OF YOLO DATASET
                        help='Root dataset path (containing train/, valid/, test/)')
    parser.add_argument('--data_set', default='fish',
                        choices=['cifar_S32','cifar_S224', 'imagenet','fish'],
                        type=str)
    parser.add_argument('--input_size', default=640, type=int, # Pixel H, W of image for SemCom & YOLO
                        help='images input size for data')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--num_object_classes', default=1, type=int, # For YOLO mAP calculation
                        help='Number of object classes for detection task (e.g., 1 for fish)')


    # Channel parameters (for SemCom simulation)
    parser.add_argument('--channel_type', default='rayleigh', choices=['none', 'awgn', 'rayleigh', 'rician'], type=str,
                        help='Type of channel for SemCom simulation (default: rayleigh)')
    parser.add_argument('--snr_db_train_min', default=-5.0, type=float,
                        help='Min SNR in dB for SemCom training (default: -5.0)')
    parser.add_argument('--snr_db_train_max', default=20.0, type=float,
                        help='Max SNR in dB for SemCom training (default: 20.0)')
    parser.add_argument('--snr_db_eval', default=10.0, type=float,
                        help='Fixed SNR in dB for SemCom evaluation (default: 10.0)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default e.g. 0.9 0.999)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', # e.g., 1.0
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of WD.""")
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', # Recon might need different LR
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', # Longer warmup often helps
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='steps to warmup LR, overrides warmup_epochs if > 0')

    parser.add_argument('--model', default='ViT_Reconstruction_Model_Default', type=str, metavar='MODEL', # Default to reconstruction model
                        help='Name of SemCom model to train')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--output_dir', default='ckpt_reconstruction_yolo',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=False)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    # parser.add_argument('--if_attack_train', action='store_true' ,help='active train attack on SemCom input')
    # parser.add_argument('--if_attack_test', action='store_true', help='active test attack on SemCom input')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist_eval', action='store_true', default=False, help='Enabling distributed evaluation')

    # Augmentation parameters (less critical for pure reconstruction, more for YOLO training if you fine-tune it)
    # parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing (default: 0.0)')
    # parser.add_argument('--train_interpolation', type=str, default='bicubic', help='Training interpolation')

    parser.add_argument('--save_ckpt', action='store_true')
    parser.set_defaults(save_ckpt=True)

    # parser.add_argument('--train_type', default='std_train', choices=['std_train','fim_train'], # FIM less relevant for pure recon
    #                     type=str)

    # YOLO specific args (add more as needed for your YOLOv11)
    parser.add_argument('--yolo_weights', default='best.pt', type=str,
                        help='Path to pre-trained YOLOv11 weights')
    parser.add_argument('--yolo_conf_thres', default=0.25, type=float, help='YOLO confidence threshold for NMS')
    parser.add_argument('--yolo_iou_thres', default=0.45, type=float, help='YOLO IoU threshold for NMS')
    parser.add_argument('--eval_viz_output_dir', default='eval_visualizations', type=str,
                        help='Directory to save evaluation visualizations')

    return parser.parse_args()