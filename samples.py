"""
The inference code of Ophora. 
"""

import argparse
from typing import Literal

import torch
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)

from diffusers.utils import export_to_video, load_image, load_video
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from dataset.t2i import TextDataset
from tqdm import tqdm
import os
import pdb
import random


def generate_video(
    pipe, 
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: int = 1360,
    height: int = 768,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    fps: int = 8,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """

    if generate_type == "t2v":
        video_generate = pipe(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]
    export_to_video(video_generate, output_path, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using Ophora")
    parser.add_argument("--csv_path", type=str, required=True, help="The input prompt of the video to be generated")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default="", help="Path of the pre-trained model use"
    )
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument("--output_path", type=str, default="./output.mp4", help="The path save generated video")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of steps for the inference process")
    parser.add_argument("--width", type=int, default=1360, help="Number of steps for the inference process")
    parser.add_argument("--height", type=int, default=768, help="Number of steps for the inference process")
    parser.add_argument("--fps", type=int, default=16, help="Number of steps for the inference process")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--generate_type", type=str, default="t2v", help="The type of video generation")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for inference")
    parser.add_argument("--text_prompt_key", type=str, default="", help="prompts type")
    parser.add_argument("--name_prompt_key", type=str, default="", help="prompts type")
    parser.add_argument("--local-rank", type=int, default=0)

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    # setup ddp
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.seed
    torch.cuda.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank {rank}, seed {seed}, world size {dist.get_world_size()}")

    dataset = TextDataset(args.csv_path, text_key=args.text_prompt_key, name_key=args.name_prompt_key, return_name=True)
    sampler = DistributedSampler(dataset, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=32,
        pin_memory=True,
        sampler=sampler,
        drop_last=False
    )

    dist.barrier()
    if rank == 0:
        print(f"Generating {len(dataset)} samples")

    image = None
    video = None

    
    if args.generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype)

    # If you're using with lora, add this code
    if args.lora_path:
        pipe.load_lora_weights(args.lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(lora_scale=1 / lora_rank)

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    for name, text in tqdm(dataloader):
               
        for i in range(len(name)):
            if not name[i].endswith('.mp4'):
                name[i] = f"{name[i]}.mp4"

            video_outpath = os.path.join(args.output_path, name[0])
            os.makedirs(os.path.dirname(video_outpath), exist_ok=True)
        
        for j in range(args.num_videos_per_prompt):
            video_outpath = os.path.join(args.output_path, f"{name[0]}", "{:}.mp4".format(j+1))
            os.makedirs(os.path.dirname(video_outpath), exist_ok=True)
            
            gen_seed = 12345
            generate_video(
                pipe=pipe,
                prompt=text,
                model_path=args.model_path,
                lora_path=args.lora_path,
                lora_rank=args.lora_rank,
                output_path=video_outpath,
                num_frames=args.num_frames,
                width=args.width,
                height=args.height,
                image_or_video_path=args.image_or_video_path,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_videos_per_prompt=1,
                dtype=dtype,
                generate_type=args.generate_type,
                seed=gen_seed,
                fps=args.fps,
            )
