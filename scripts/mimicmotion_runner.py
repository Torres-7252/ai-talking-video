#!/usr/bin/env python3
"""Run pinned MimicMotion inference inside its isolated Python environment."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-VRAM MimicMotion runner")
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration", required=True, type=float)
    parser.add_argument("--intensity", required=True, type=float)
    parser.add_argument("--resolution", required=True, type=int)
    parser.add_argument("--fps", default=15, type=int)
    parser.add_argument("--tile-size", default=16, type=int)
    parser.add_argument("--tile-overlap", default=6, type=int)
    parser.add_argument("--steps", default=25, type=int)
    parser.add_argument("--decode-chunk-size", default=1, type=int)
    return parser.parse_args()


def _validate_driver_pose(driver: Path, minimum_ratio: float = 0.8) -> float:
    import decord
    import numpy as np
    from mimicmotion.dwpose.dwpose_detector import dwpose_detector

    reader = decord.VideoReader(str(driver), ctx=decord.cpu(0))
    if not len(reader):
        raise RuntimeError(f"Gesture driver contains no frames: {driver}")
    indices = np.linspace(0, len(reader) - 1, min(12, len(reader)), dtype=int).tolist()
    frames = reader.get_batch(indices).asnumpy()
    valid = 0
    torso_and_arms = np.array([1, 2, 3, 4, 5, 6, 7])
    for frame in frames:
        pose = dwpose_detector(frame)
        subset = pose["bodies"]["subset"]
        if len(subset) and np.count_nonzero(subset[0][torso_and_arms] >= 0) >= 6:
            valid += 1
    dwpose_detector.release_memory()
    ratio = valid / len(frames)
    if ratio < minimum_ratio:
        raise RuntimeError(
            "Gesture driver torso/arm landmark coverage is too low: "
            f"{ratio:.1%} (minimum {minimum_ratio:.0%})"
        )
    return ratio


def _loop_pose_frames(pose_pixels, target_frames: int):
    import torch

    if pose_pixels.size(0) < 2:
        raise RuntimeError("MimicMotion preprocessing returned no driver pose frames")
    driver_frames = pose_pixels[1:]
    repeats = math.ceil((target_frames - 1) / driver_frames.size(0))
    looped = driver_frames.repeat((repeats, 1, 1, 1))[: target_frames - 1]
    return torch.cat((pose_pixels[:1], looped), dim=0)


def _force_cpu_vae_decode(pipeline) -> None:
    """Keep frame decoding out of limited VRAM after denoising completes."""
    original_decode = pipeline.decode_latents

    def decode_on_cpu(latents, num_frames, decode_chunk_size=1):
        pipeline.vae.decoder.cpu()
        return original_decode(
            latents.cpu(), num_frames, decode_chunk_size=decode_chunk_size
        )

    pipeline.decode_latents = decode_on_cpu


def main() -> None:
    args = parse_args()
    runtime = args.runtime.resolve()
    sys.path.insert(0, str(runtime))

    import torch
    from omegaconf import OmegaConf
    from torchvision.transforms.functional import to_pil_image

    from inference import preprocess
    from mimicmotion.utils.loader import create_pipeline
    from mimicmotion.utils.utils import save_to_mp4

    if not torch.cuda.is_available():
        raise RuntimeError("MimicMotion requires a CUDA GPU")
    if args.duration <= 0 or not 0.0 <= args.intensity <= 1.0:
        raise ValueError("Invalid duration or intensity")

    print(f"Driver pose coverage: {_validate_driver_pose(args.driver.resolve()):.1%}")
    torch.set_default_dtype(torch.float16)
    device = torch.device("cuda")
    pose_pixels, image_pixels = preprocess(
        str(args.driver.resolve()),
        str(args.image.resolve()),
        resolution=args.resolution,
        sample_stride=1,
    )
    target_frames = math.ceil(args.duration * args.fps) + 1
    pose_pixels = _loop_pose_frames(pose_pixels, target_frames)
    config = OmegaConf.create(
        {
            "base_model_path": str(
                runtime / "models" / "stable-video-diffusion-img2vid-xt-1-1"
            ),
            "ckpt_path": str(runtime / "models" / "MimicMotion_1-1.pth"),
        }
    )
    pipeline = create_pipeline(config, device)
    _force_cpu_vae_decode(pipeline)
    images = [
        to_pil_image(image.to(torch.uint8))
        for image in (image_pixels + 1.0) * 127.5
    ]
    generator = torch.Generator(device=device).manual_seed(42)
    guidance = 1.5 + 2.0 * args.intensity
    frames = pipeline(
        images,
        image_pose=pose_pixels,
        num_frames=target_frames,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        height=pose_pixels.shape[-2],
        width=pose_pixels.shape[-1],
        fps=args.fps,
        noise_aug_strength=0,
        num_inference_steps=args.steps,
        generator=generator,
        min_guidance_scale=guidance,
        max_guidance_scale=guidance,
        decode_chunk_size=args.decode_chunk_size,
        output_type="pt",
        device=device,
    ).frames.cpu()[0, 1:]
    save_to_mp4((frames * 255.0).clamp(0, 255).to(torch.uint8), args.output, fps=args.fps)


if __name__ == "__main__":
    main()
