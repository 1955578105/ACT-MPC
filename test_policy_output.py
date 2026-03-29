from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a trained LeRobot ACT policy, feed synthetic observations, and print action magnitudes."
    )
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=Path("/home/Actdog/policy/checkpoints/last/pretrained_model"),
        help="Directory containing config.json and model weights.",
    )
    parser.add_argument(
        "--lerobot-src",
        type=Path,
        default=Path("/home/zhangzhong/lerobot/src"),
        help="Path to the local lerobot source tree.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible synthetic inputs.",
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


def chw_from_hwc_uint8(image_hwc: np.ndarray) -> torch.Tensor:
    chw = np.transpose(image_hwc, (2, 0, 1)).astype(np.float32, copy=False)
    return torch.from_numpy(chw).unsqueeze(0)


def summarize_chunk(name: str, actions_norm: torch.Tensor, actions_real: torch.Tensor) -> None:
    norm_np = actions_norm.squeeze(0).detach().cpu().numpy()
    real_np = actions_real.squeeze(0).detach().cpu().numpy()

    print(f"\n=== {name} ===")
    print(f"chunk shape: {tuple(real_np.shape)}")
    print(f"first normalized action: {norm_np[0].tolist()}")
    print(f"first real action:       {real_np[0].tolist()}")
    print(f"real action min:         {real_np.min(axis=0).tolist()}")
    print(f"real action max:         {real_np.max(axis=0).tolist()}")
    print(f"real action mean:        {real_np.mean(axis=0).tolist()}")
    print(f"real action abs mean:    {np.abs(real_np).mean(axis=0).tolist()}")


def main() -> None:
    args = parse_args()
    add_lerobot_to_path(args.lerobot_src)

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    image_key = "observation.images.front"
    state_dim = int(cfg.robot_state_feature.shape[0]) if cfg.robot_state_feature else 0
    image_c, image_h, image_w = cfg.input_features[image_key].shape

    if state_dim <= 0:
        raise ValueError("The checkpoint does not expose observation.state.")
    if image_c != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got {image_c}.")

    image_mean_hwc = np.transpose(image_mean.squeeze(0).cpu().numpy(), (1, 2, 0))
    mean_image_uint8 = np.clip(image_mean_hwc * 255.0, 0.0, 255.0).astype(np.uint8)

    def run_case(case_name: str, state_raw_np: np.ndarray, image_hwc_uint8: np.ndarray) -> None:
        state_raw = torch.from_numpy(state_raw_np.astype(np.float32, copy=False)).unsqueeze(0)
        image_raw = chw_from_hwc_uint8(image_hwc_uint8)

        state_norm = (state_raw - state_mean) / (state_std + 1e-8)
        image_01 = image_raw / 255.0
        image_norm = (image_01 - image_mean) / (image_std + 1e-8)

        batch = {
            "observation.state": state_norm,
            "observation.images": [image_norm],
        }

        with torch.no_grad():
            actions_norm = policy.model(batch)[0][:, : cfg.n_action_steps, :]
            actions_real = actions_norm * action_std.unsqueeze(1) + action_mean.unsqueeze(1)

        summarize_chunk(case_name, actions_norm, actions_real)

    state_mean_np = state_mean.squeeze(0).cpu().numpy()

    zero_state = np.zeros((state_dim,), dtype=np.float32)
    black_image = np.zeros((image_h, image_w, image_c), dtype=np.uint8)

    mean_state = state_mean_np.astype(np.float32, copy=True)
    mean_image = mean_image_uint8

    forward_state = state_mean_np.astype(np.float32, copy=True)
    if state_dim >= 3:
        forward_state[0] = 0.4
        forward_state[1] = 0.0
        forward_state[2] = 0.0
    if state_dim > 3:
        forward_state[3:] = 2.5
    red_ball_image = np.zeros((image_h, image_w, image_c), dtype=np.uint8)
    red_ball_image[:] = (20, 20, 20)
    cy = image_h // 2
    cx = image_w // 2
    half = max(4, min(image_h, image_w) // 16)
    red_ball_image[cy - half : cy + half, cx - half : cx + half, 0] = 255
    red_ball_image[cy - half : cy + half, cx - half : cx + half, 1] = 0
    red_ball_image[cy - half : cy + half, cx - half : cx + half, 2] = 0

    random_state = state_mean_np.astype(np.float32, copy=True)
    random_state += np.random.normal(0.0, 0.25, size=(state_dim,)).astype(np.float32)
    random_image = np.random.randint(0, 256, size=(image_h, image_w, image_c), dtype=np.uint8)

    print(f"policy dir: {args.policy_dir}")
    print(f"state dim: {state_dim}")
    print(f"image shape: {(image_h, image_w, image_c)}")
    print(f"action dim: {cfg.output_features['action'].shape[0]}")
    print(f"n_action_steps: {cfg.n_action_steps}")
    print(f"action mean (dataset): {action_mean.squeeze(0).cpu().numpy().tolist()}")
    print(f"action std (dataset):  {action_std.squeeze(0).cpu().numpy().tolist()}")

    run_case("zero_state + black_image", zero_state, black_image)
    run_case("mean_state + mean_image", mean_state, mean_image)
    run_case("forward_state + red_ball_image", forward_state, red_ball_image)
    run_case("random_state + random_image", random_state, random_image)


if __name__ == "__main__":
    main()
