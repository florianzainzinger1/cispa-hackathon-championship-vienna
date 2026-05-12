#!/usr/bin/env python3
"""
Test BitMark watermark detection on original vs attacked images.
"""
import sys
sys.path.insert(0, '/p/project1/training2562/syed1/Hackathon/task_3_watermark_removal/repo/BitMark')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Check GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Import BitMark components
from helper import set_seeds
from architecture_wrapper import VAEWrapper, get_architecture_arguments, get_vae
from detect_watermark import get_detector, detect

set_seeds(42)

# Create args manually with ALL required fields
class Args:
    # Architecture
    architecture = "infinity"
    
    # Infinity-specific args
    cfg = "3"
    tau = 1
    pn = "1M"
    model_path = "./weights/Infinity/infinity_2b_reg.pth"
    cfg_insertion_layer = 0
    vae_type = 32
    vae_path = "./weights/Infinity/infinity_vae_d32reg.pth"
    add_lvl_embeding_only_first_block = 1
    use_bit_label = 1
    model_type = "infinity_2b"
    rope2d_each_sa_layer = 1
    rope2d_normalized_by_hw = 2
    use_scale_schedule_embedding = 0
    sampling_per_bits = 1
    text_encoder_ckpt = ""
    text_channels = 2048
    apply_spatial_patchify = 0
    h_div_w_template = 1.0
    use_flex_attn = 0
    enable_positive_prompt = 0
    cache_dir = "/dev/shm"
    enable_model_cache = 0
    checkpoint_type = "torch"
    bf16 = 1
    inject_scales = 0
    inject_scales_path = ""
    decode_per_scale = 0
    
    # Watermark args
    watermark_delta = 2
    watermark_context_width = 2
    watermark_scales = 2
    watermark_method = "2-bit_pattern"
    watermark_remove_duplicates = 0
    watermark_add_noise = 0
    watermark_count_bit_flip = 0
    watermark_gen_image = 0
    watermark_count_bit_loss_after_reencoding = 0
    watermark_seeding_scheme = "selfhash"
    
    # Common args
    seed = 42
    set = "00,11"  # Green list for 2-bit pattern
    batch_size = 1

args = Args()

print("\n=== Loading VAE ===")
os.chdir('/p/project1/training2562/syed1/Hackathon/task_3_watermark_removal/repo/BitMark')
vae_wrapper = get_vae(args)
print("✓ VAE loaded")

print("\n=== Loading Detector ===")
watermark_detector = get_detector(args)
print("✓ Detector loaded")

# Test images
original_path = "/p/project1/training2562/syed1/Hackathon/task_3_watermark_removal/Dataset/1.png"
attacked_path = "/p/project1/training2562/syed1/Hackathon/task_3_watermark_removal/output/test_attacked_safe_1.png"

print("\n=== Testing Original Image ===")
print(f"Path: {original_path}")
original_results = detect(
    args, 
    original_path, 
    watermark_detector=watermark_detector, 
    vae_wrapper=vae_wrapper, 
    watermark_scales=[2],
    detect_on_each_scale=False
)
print(f"  z_score: {original_results['z_score']:.4f}")
print(f"  green_fraction: {original_results['green_fraction']:.4f}")
print(f"  Watermark detected: {'YES' if original_results['z_score'] > 4.0 else 'NO'}")

print("\n=== Testing Attacked Image ===")
print(f"Path: {attacked_path}")
if os.path.exists(attacked_path):
    attacked_results = detect(
        args, 
        attacked_path, 
        watermark_detector=watermark_detector, 
        vae_wrapper=vae_wrapper, 
        watermark_scales=[2],
        detect_on_each_scale=False
    )
    print(f"  z_score: {attacked_results['z_score']:.4f}")
    print(f"  green_fraction: {attacked_results['green_fraction']:.4f}")
    print(f"  Watermark detected: {'YES' if attacked_results['z_score'] > 4.0 else 'NO'}")
    
    print("\n=== COMPARISON ===")
    print(f"Original z_score:  {original_results['z_score']:.4f}")
    print(f"Attacked z_score:  {attacked_results['z_score']:.4f}")
    print(f"Reduction:         {original_results['z_score'] - attacked_results['z_score']:.4f}")
    print(f"Success: {'YES - Watermark removed!' if attacked_results['z_score'] < 4.0 else 'NO - Watermark still detected'}")
else:
    print(f"ERROR: Attacked image not found at {attacked_path}")

print("\n✓ Done!")
