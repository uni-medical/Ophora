export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH=$PYTHONPATH:$PWD

# Echoing the job settings
echo "PYTHONPATH: $PYTHONPATH"
echo "PWD: $PWD"
echo "Conda ENV: `conda env list`"

# Training Configurations
# Experiment with as many hyperparameters as you want!
LEARNING_RATES=("5e-5")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("adamw")
MAX_TRAIN_STEPS=("4500")

# Single GPU uncompiled training
ACCELERATE_CONFIG_FILE="./accelerate_configs/multinode.yaml"

# Absolute path to where the data is located. Make sure to have read the README for how to prepare data.
DATA_ROOT="/path/dataset_root"
VIDEO_DATA_ROOT="/path/training_data/visual_embedding"
TEXT_DATA_ROOT="/path/training_data/text_embedding"
CAPTION_COLUMN="/path/training_data/video_ids.txt"
VIDEO_COLUMN="/path/training_data/video_ids.txt"

# Launch experiments with different hyperparameters
for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do
        output_dir="/path/output_dir"

        cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --multi_gpu --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT}  ./training/cogvideox_text_to_video_sft.py \
          --pretrained_model_name_or_path ./TPT-Model \
          --data_root $DATA_ROOT \
          --instance_data_root $VIDEO_DATA_ROOT \
          --instance_text_data_root $TEXT_DATA_ROOT \
          --caption_column $CAPTION_COLUMN \
          --video_column $VIDEO_COLUMN \
          --height_buckets 480 \
          --width_buckets 720 \
          --frame_buckets 49 \
          --dataloader_num_workers 64 \
          --pin_memory \
          --validation_prompt /path/val_instructions.txt \
          --video_ids /path/val_instructions.txt \
          --validation_prompt_separator ::: \
          --num_validation_videos 1 \
          --validation_epochs 1 \
          --seed 42 \
          --mixed_precision fp16 \
          --output_dir $output_dir \
          --max_num_frames 49 \
          --train_batch_size 8 \
          --max_train_steps $steps \
          --checkpointing_steps 500 \
          --gradient_accumulation_steps 1 \
          --gradient_checkpointing
          --learning_rate $learning_rate \
          --lr_scheduler $lr_schedule \
          --lr_warmup_steps 800 \
          --lr_num_cycles 1 \
          --enable_slicing \
          --enable_tiling \
          --optimizer $optimizer \
          --beta1 0.9 \
          --beta2 0.95 \
          --weight_decay 0.001 \
          --max_grad_norm 1.0 \
          --allow_tf32 \
          --report_to tensorboard \
          --nccl_timeout 1800 \
          --load_tensors \
          --resume_from_checkpoint latest"
        
        echo "Running command: $cmd"
        eval $cmd
        echo -ne "-------------------- Finished executing script --------------------\n\n"
      done
    done
  done
done
