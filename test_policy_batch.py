from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


STATE_KEY = "observation.state"
RADAR_KEY = "observation.radar"
IMAGE_KEY = "observation.images.front"
ACTION_KEY = "action"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate a trained LeRobot ACT policy on real captured frames."
    )
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=Path("/home/Actdog/policy/checkpoints/last/pretrained_model"),
        help="Directory containing config.json and trained weights.",
    )
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=Path("/home/Actdog/capture"),
        help="Directory containing episode_xxxx folders.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of random real frames to evaluate.",
    )
    parser.add_argument(
        "--latest-episodes",
        type=int,
        default=100,
        help="Only sample from the latest N episodes for faster iteration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for frame sampling.",
    )
    parser.add_argument(
        "--lerobot-src",
        type=Path,
        default=Path("/home/zhangzhong/lerobot/src"),
        help="Path to the local lerobot source tree.",
    )
    return parser.parse_args()


def add_lerobot_to_path(lerobot_src: Path) -> None:
    lerobot_src = lerobot_src.resolve()
    if not lerobot_src.exists():
        raise FileNotFoundError(f"LeRobot source directory not found: {lerobot_src}")
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))


def find_stats(processor, key: str) -> tuple[torch.Tensor, torch.Tensor]:
    for step in getattr(processor, "steps", []):
        tensor_stats = getattr(step, "_tensor_stats", None)
        if tensor_stats and key in tensor_stats:
            stats = tensor_stats[key]
            mean = stats.get("mean")
            std = stats.get("std")
            if mean is None or std is None:
                raise ValueError(f"Stats for {key} are present but mean/std are missing.")
            return mean.detach().cpu().float(), std.detach().cpu().float()
    raise KeyError(f"Could not find normalization stats for key: {key}")


def reshape_state_stats(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(1, -1)


def reshape_image_stats(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor.reshape(1, -1, 1, 1)
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f"Unsupported image stats shape: {tuple(tensor.shape)}")


def load_episode(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def image_to_chw_float(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.float32)
    chw = np.transpose(image_np, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    args = parse_args()
    add_lerobot_to_path(args.lerobot_src)

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    episode_jsons = sorted(args.capture_root.glob("episode_*/*.json"))
    if not episode_jsons:
        raise FileNotFoundError(f"No episode json files found under: {args.capture_root}")
    if args.latest_episodes > 0:
        episode_jsons = episode_jsons[-args.latest_episodes :]

    frame_pool: list[tuple[Path, int]] = []
    for episode_json in episode_jsons:
        episode = load_episode(episode_json)
        frame_count = len(episode[ACTION_KEY])
        for frame_idx in range(frame_count):
            frame_pool.append((episode_json, frame_idx))

    if not frame_pool:
        raise RuntimeError("No frames available for evaluation.")

    sample_count = min(args.num_samples, len(frame_pool))
    sampled_frames = random.sample(frame_pool, sample_count)

    cfg = PreTrainedConfig.from_pretrained(args.policy_dir)
    cfg.device = "cpu"

    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(args.policy_dir, config=cfg)
    policy.eval()
    policy.to("cpu")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.policy_dir,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    state_mean, state_std = find_stats(preprocessor, "observation.state")
    image_mean, image_std = find_stats(preprocessor, "observation.images.front")
    action_mean, action_std = find_stats(postprocessor, "action")

    state_mean = reshape_state_stats(state_mean)
    state_std = reshape_state_stats(state_std)
    image_mean = reshape_image_stats(image_mean)
    image_std = reshape_image_stats(image_std)
    action_mean = reshape_state_stats(action_mean)
    action_std = reshape_state_stats(action_std)

    labels = []
    preds_first = []
    preds_chunk_mean = []

    for episode_json, frame_idx in sampled_frames:
        episode = load_episode(episode_json)

        raw_state = np.asarray(episode[STATE_KEY][frame_idx], dtype=np.float32)
        raw_radar = np.asarray(episode[RADAR_KEY][frame_idx], dtype=np.float32)
        merged_state = np.concatenate([raw_state, raw_radar], axis=0)
        recorded_action = np.asarray(episode[ACTION_KEY][frame_idx], dtype=np.float32)

        image_name = episode[IMAGE_KEY][frame_idx]
        image_path = episode_json.parent / "images" / image_name

        state_raw = torch.from_numpy(merged_state).unsqueeze(0)
        image_raw = image_to_chw_float(image_path)

        state_norm = (state_raw - state_mean) / (state_std + 1e-8)
        image_norm = (image_raw / 255.0 - image_mean) / (image_std + 1e-8)

        batch = {
            "observation.state": state_norm,
            "observation.images": [image_norm],
        }

        with torch.no_grad():
            actions_norm = policy.model(batch)[0][:, : cfg.n_action_steps, :]
            actions_real = actions_norm * action_std.unsqueeze(1) + action_mean.unsqueeze(1)

        actions_real_np = actions_real.squeeze(0).cpu().numpy()

        labels.append(recorded_action)
        preds_first.append(actions_real_np[0])
        preds_chunk_mean.append(actions_real_np.mean(axis=0))

    labels = np.asarray(labels, dtype=np.float32)
    preds_first = np.asarray(preds_first, dtype=np.float32)
    preds_chunk_mean = np.asarray(preds_chunk_mean, dtype=np.float32)

    dims = ["vx", "vy", "wz"]

    print(f"policy dir: {args.policy_dir}")
    print(f"sampled frames: {sample_count}")
    print(f"sample source episodes: {len(episode_jsons)}")

    print("\n=== First Action Vs Label ===")
    for i, name in enumerate(dims):
        mae = float(np.mean(np.abs(preds_first[:, i] - labels[:, i])))
        label_mean = float(np.mean(labels[:, i]))
        label_std = float(np.std(labels[:, i]))
        pred_mean = float(np.mean(preds_first[:, i]))
        pred_std = float(np.std(preds_first[:, i]))
        corr = corrcoef_safe(labels[:, i], preds_first[:, i])
        print(
            f"{name}: label_mean={label_mean:.6f}, label_std={label_std:.6f}, "
            f"pred_mean={pred_mean:.6f}, pred_std={pred_std:.6f}, mae={mae:.6f}, corr={corr:.6f}"
        )

    print("\n=== Chunk Mean Vs Label ===")
    for i, name in enumerate(dims):
        mae = float(np.mean(np.abs(preds_chunk_mean[:, i] - labels[:, i])))
        label_mean = float(np.mean(labels[:, i]))
        label_std = float(np.std(labels[:, i]))
        pred_mean = float(np.mean(preds_chunk_mean[:, i]))
        pred_std = float(np.std(preds_chunk_mean[:, i]))
        corr = corrcoef_safe(labels[:, i], preds_chunk_mean[:, i])
        print(
            f"{name}: label_mean={label_mean:.6f}, label_std={label_std:.6f}, "
            f"pred_mean={pred_mean:.6f}, pred_std={pred_std:.6f}, mae={mae:.6f}, corr={corr:.6f}"
        )


if __name__ == "__main__":
    main()
