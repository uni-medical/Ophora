#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$PWD
export CUDA_VISIBLE_DEVICES="0"

NUM_GPUS=1

CKPT="path/Ophora-ckpt"
OUTPUT_DIR="path/synthesized_video_path"

test_example_casepath="path/input_prompts.csv"

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=12345 samples.py --csv_path ${test_example_casepath} --model_path ${CKPT} --output_path ${OUTPUT_DIR} --guidance_scale 6 --num_inference_steps 50 --num_frames 49 --width 720 --height 480 --fps 8 --dtype "float16" --seed 42 --batch_size 1 --text_prompt_key "instruction" --name_prompt_key "clip id" --num_videos_per_prompt 1
