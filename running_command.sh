# running_command.sh
export CUDA_VISIBLE_DEVICES=0 # Or your desired GPU

python3 run_class_main.py \
    --model ViT_Reconstruction_Model_Default \
    --output_dir ckpt_semcom_reconstruction_yolo_fish \
    --data_set fish \
    --data_path "" \
    --num_object_classes 1 \
    --yolo_weights "best.pt" \
    --batch_size 8 \
    --input_size 224 \
    --lr 1e-4 \
    --min_lr 1e-6 \
    --warmup_epochs 5 \
    --epochs 400 \
    --opt adamw \
    --weight_decay 0.01 \
    --save_freq 5 \
    --mask_ratio 0.0 \
    --snr_db_train_min 25 \
    --snr_db_train_max 25 \
    --snr_db_eval 25 \
    --num_workers 2 \
    --pin_mem \
    # --resume path/to/your/semcom_checkpoint.pth # To resume SemCom model training
    # --eval # To run only evaluation