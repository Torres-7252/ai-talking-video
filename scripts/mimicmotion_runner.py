#!/usr/bin/env python3
"""Run pinned MimicMotion inference inside its isolated Python environment."""

from __future__ import annotations

import argparse
import copy
import math
import subprocess
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


def _scale_pose_motion(reference_pose, moving_pose, intensity: float):
    """Scale detected landmark displacement around the avatar's reference pose."""
    import numpy as np

    scaled = copy.deepcopy(moving_pose)

    def blend(reference, moving):
        reference = np.asarray(reference)
        moving = np.asarray(moving)
        if reference.shape != moving.shape:
            return moving.copy()
        return reference + (moving - reference) * intensity

    scaled["bodies"]["candidate"] = blend(
        reference_pose["bodies"]["candidate"],
        moving_pose["bodies"]["candidate"],
    )
    scaled["bodies"]["score"] = blend(
        reference_pose["bodies"]["score"], moving_pose["bodies"]["score"]
    )
    for landmarks, scores in (("faces", "faces_score"), ("hands", "hands_score")):
        scaled[landmarks] = blend(reference_pose[landmarks], moving_pose[landmarks])
        scaled[scores] = blend(reference_pose[scores], moving_pose[scores])
    return scaled


def _get_video_pose(video_path: str, ref_image, sample_stride: int, intensity: float):
    """Run the pinned DWPose mapping with controllable landmark displacement."""
    import decord
    import numpy as np

    from mimicmotion.dwpose.dwpose_detector import dwpose_detector
    from mimicmotion.dwpose.util import draw_pose

    reference_pose = dwpose_detector(ref_image)
    reference_ids = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    reference_ids = [
        index
        for index in reference_ids
        if len(reference_pose["bodies"]["subset"])
        and reference_pose["bodies"]["subset"][0][index] >= 0
    ]
    if len(reference_ids) < 4:
        raise RuntimeError("Avatar reference image has insufficient body landmarks")
    reference_body = reference_pose["bodies"]["candidate"][reference_ids]

    reader = decord.VideoReader(video_path, ctx=decord.cpu(0))
    sample_stride *= max(1, int(reader.get_avg_fps() / 24))
    frames = reader.get_batch(list(range(0, len(reader), sample_stride))).asnumpy()
    detected_poses = [dwpose_detector(frame) for frame in frames]

    valid_bodies = [
        pose["bodies"]["candidate"]
        for pose in detected_poses
        if pose["bodies"]["candidate"].shape[0] == 18
    ]
    if len(valid_bodies) != len(detected_poses):
        dwpose_detector.release_memory()
        raise RuntimeError("DWPose lost the primary presenter in one or more frames")
    detected_bodies = np.stack(valid_bodies)[:, reference_ids]

    height, width, _ = ref_image.shape
    ay, by = np.polyfit(
        detected_bodies[:, :, 1].flatten(),
        np.tile(reference_body[:, 1], len(detected_bodies)),
        1,
    )
    source_height, source_width, _ = reader[0].shape
    ax = ay / (source_height / source_width / height * width)
    bx = np.mean(
        np.tile(reference_body[:, 0], len(detected_bodies))
        - detected_bodies[:, :, 0].flatten() * ax
    )
    scale = np.array([ax, ay])
    offset = np.array([bx, by])

    output = []
    for pose in detected_poses:
        pose["bodies"]["candidate"] = pose["bodies"]["candidate"] * scale + offset
        pose["faces"] = pose["faces"] * scale + offset
        pose["hands"] = pose["hands"] * scale + offset
        pose = _scale_pose_motion(reference_pose, pose, intensity)
        output.append(np.array(draw_pose(pose, height, width)))
    dwpose_detector.release_memory()
    return np.stack(output)


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


def _save_to_mp4(frames, output: Path, fps: int) -> None:
    height, width = frames.shape[-2:]
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output.resolve()),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.permute(1, 2, 0).contiguous().numpy().tobytes())
    process.stdin.close()
    assert process.stderr is not None
    error = process.stderr.read().decode("utf-8", errors="replace")
    if process.wait() != 0:
        raise RuntimeError(f"FFmpeg failed to save MimicMotion output: {error}")


def main() -> None:
    args = parse_args()
    runtime = args.runtime.resolve()
    sys.path.insert(0, str(runtime))

    import torch
    from omegaconf import OmegaConf
    from torchvision.transforms.functional import to_pil_image

    import inference
    from mimicmotion.utils.loader import create_pipeline

    if not torch.cuda.is_available():
        raise RuntimeError("MimicMotion requires a CUDA GPU")
    if args.duration <= 0 or not 0.0 <= args.intensity <= 1.0:
        raise ValueError("Invalid duration or intensity")

    print(f"Driver pose coverage: {_validate_driver_pose(args.driver.resolve()):.1%}")
    torch.set_default_dtype(torch.float16)
    device = torch.device("cuda")
    original_get_video_pose = inference.get_video_pose
    inference.get_video_pose = lambda video_path, ref_image, sample_stride=1: (
        _get_video_pose(video_path, ref_image, sample_stride, args.intensity)
    )
    try:
        pose_pixels, image_pixels = inference.preprocess(
            str(args.driver.resolve()),
            str(args.image.resolve()),
            resolution=args.resolution,
            sample_stride=1,
        )
    finally:
        inference.get_video_pose = original_get_video_pose
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
    _save_to_mp4(
        (frames * 255.0).clamp(0, 255).to(torch.uint8), args.output, args.fps
    )


if __name__ == "__main__":
    main()
