from __future__ import annotations

import argparse
import json
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
        description="Replay real captured frames through a trained LeRobot ACT policy."
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
        "--episode-json",
        type=Path,
        default=None,
        help="Optional explicit episode json path. If omitted, use the latest episode under capture-root.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index to replay.",
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


def latest_episode_json(capture_root: Path) -> Path:
    episode_jsons = sorted(capture_root.glob("episode_*/*.json"))
    if not episode_jsons:
        raise FileNotFoundError(f"No episode json files found under: {capture_root}")
    return episode_jsons[-1]


def load_episode(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        episode = json.load(f)

    for key in [STATE_KEY, RADAR_KEY, IMAGE_KEY, ACTION_KEY]:
        if key not in episode:
            raise KeyError(f"{json_path} is missing key: {key}")
    return episode


def image_to_chw_float(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.float32)
    chw = np.transpose(image_np, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def main() -> None:
    args = parse_args()
    add_lerobot_to_path(args.lerobot_src)

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    episode_json = args.episode_json.resolve() if args.episode_json else latest_episode_json(args.capture_root)
    episode = load_episode(episode_json)

    frame_count = len(episode[ACTION_KEY])
    if args.frame_index < 0 or args.frame_index >= frame_count:
        raise IndexError(f"frame-index {args.frame_index} is out of range for {frame_count} frames")

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

    raw_state = np.asarray(episode[STATE_KEY][args.frame_index], dtype=np.float32)
    raw_radar = np.asarray(episode[RADAR_KEY][args.frame_index], dtype=np.float32)
    merged_state = np.concatenate([raw_state, raw_radar], axis=0)
    recorded_action = np.asarray(episode[ACTION_KEY][args.frame_index], dtype=np.float32)

    image_name = episode[IMAGE_KEY][args.frame_index]
    image_path = episode_json.parent / "images" / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image file: {image_path}")

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

    actions_norm_np = actions_norm.squeeze(0).cpu().numpy()
    actions_real_np = actions_real.squeeze(0).cpu().numpy()

    print(f"policy dir: {args.policy_dir}")
    print(f"episode: {episode_json}")
    print(f"frame index: {args.frame_index}")
    print(f"image: {image_name}")
    print(f"merged state dim: {merged_state.shape[0]}")
    print(f"recorded action: {recorded_action.tolist()}")
    print(f"first normalized action: {actions_norm_np[0].tolist()}")
    print(f"first real action:       {actions_real_np[0].tolist()}")
    print(f"real action mean over chunk: {actions_real_np.mean(axis=0).tolist()}")
    print(f"real action abs mean over chunk: {np.abs(actions_real_np).mean(axis=0).tolist()}")
    print(f"real action min over chunk: {actions_real_np.min(axis=0).tolist()}")
    print(f"real action max over chunk: {actions_real_np.max(axis=0).tolist()}")


if __name__ == "__main__":
    main()
