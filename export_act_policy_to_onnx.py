from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a LeRobot ACT policy to ONNX for C++ deployment.")
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=Path("/home/Actdog/policy/checkpoints/last/pretrained_model"),
        help="Directory containing config.json, model.safetensors and pre/post processors.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("/home/Actdog/onnx_policy/model_deploy.onnx"),
        help="Output ONNX filename.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("/home/Actdog/onnx_policy/model_deploy.json"),
        help="Output deployment metadata filename.",
    )
    parser.add_argument(
        "--lerobot-src",
        type=Path,
        default=Path("/home/zhangzhong/lerobot/src"),
        help="Path to the local lerobot source tree.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Control frequency used during deployment. If omitted, try to infer from dataset metadata.",
    )
    return parser.parse_args()


def add_lerobot_to_path(lerobot_src: Path) -> None:
    lerobot_src = lerobot_src.resolve()
    if not lerobot_src.exists():
        raise FileNotFoundError(f"LeRobot source directory not found: {lerobot_src}")
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))


def infer_dataset_fps(policy_dir: Path) -> float | None:
    train_cfg_path = policy_dir / "train_config.json"
    if not train_cfg_path.exists():
        return None

    with train_cfg_path.open("r", encoding="utf-8") as f:
        train_cfg = json.load(f)

    dataset_root = train_cfg.get("dataset", {}).get("root")
    if not dataset_root:
        return None

    info_path = Path(dataset_root) / "meta" / "info.json"
    if not info_path.exists():
        return None

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    image_info = info.get("observation.images.front", {}).get("info", {})
    fps = image_info.get("video.fps")
    if fps is None:
        return None
    return float(fps)


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


class DeployableActWrapper(nn.Module):
    """
    ONNX-friendly wrapper around the trained ACT core model.

    Important design choice:
    - We export the pure action-chunk predictor, not the stateful `select_action()` queue logic.
    - We DO include observation normalization and action de-normalization here, so the C++ side can feed:
      1. raw state values
      2. raw RGB image values in [0, 255]
    - The C++ side only needs to manage the ACT action queue and the 20Hz sample-and-hold rhythm.
    - `state_raw` is whatever the trained checkpoint expects for `observation.state`.
      For your newer dataset format, that means the merged state:
      [vx, vy, wz, lidar_0, ..., lidar_89] -> 93 dimensions.
    """

    def __init__(
        self,
        policy,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        image_mean: torch.Tensor,
        image_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.policy_model = policy.model
        self.n_action_steps = policy.config.n_action_steps
        self.eps = 1e-8

        self.register_buffer("state_mean", reshape_state_stats(state_mean))
        self.register_buffer("state_std", reshape_state_stats(state_std))
        self.register_buffer("image_mean", reshape_image_stats(image_mean))
        self.register_buffer("image_std", reshape_image_stats(image_std))
        self.register_buffer("action_mean", reshape_state_stats(action_mean))
        self.register_buffer("action_std", reshape_state_stats(action_std))

    def forward(self, state_raw: torch.Tensor, image_raw: torch.Tensor) -> torch.Tensor:
        # state_raw: (B, state_dim)
        # image_raw: (B, 3, H, W), raw RGB values in [0, 255]
        state_norm = (state_raw - self.state_mean) / (self.state_std + self.eps)

        image_01 = image_raw / 255.0
        image_norm = (image_01 - self.image_mean) / (self.image_std + self.eps)

        batch = {
            "observation.state": state_norm,
            "observation.images": [image_norm],
        }

        actions_norm = self.policy_model(batch)[0][:, : self.n_action_steps, :]
        actions = actions_norm * self.action_std.unsqueeze(1) + self.action_mean.unsqueeze(1)
        return actions


def export_to_onnx(args: argparse.Namespace) -> None:
    if importlib.util.find_spec("onnx") is None:
        raise ModuleNotFoundError(
            "Python package 'onnx' is required for export. "
            "Please install it in the environment that runs this script, "
            "for example: pip install onnx"
        )

    add_lerobot_to_path(args.lerobot_src)

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    policy_dir = args.policy_dir.resolve()
    onnx_path = args.onnx_path.resolve()
    metadata_path = args.metadata_path.resolve()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = PreTrainedConfig.from_pretrained(policy_dir)
    cfg.device = "cpu"

    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(policy_dir, config=cfg)
    policy.eval()
    policy.to("cpu")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_dir,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    state_mean, state_std = find_stats(preprocessor, "observation.state")
    image_mean, image_std = find_stats(preprocessor, "observation.images.front")
    action_mean, action_std = find_stats(postprocessor, "action")

    wrapper = DeployableActWrapper(
        policy=policy,
        state_mean=state_mean,
        state_std=state_std,
        image_mean=image_mean,
        image_std=image_std,
        action_mean=action_mean,
        action_std=action_std,
    ).eval()

    state_dim = int(cfg.robot_state_feature.shape[0]) if cfg.robot_state_feature else 0
    if state_dim <= 0:
        raise ValueError(
            "The checkpoint does not expose a valid observation.state feature. "
            "For ACT deployment in this project, observation.state must exist."
        )
    image_key = "observation.images.front"
    image_shape = cfg.input_features[image_key].shape
    image_c, image_h, image_w = image_shape
    action_dim = int(cfg.output_features["action"].shape[0])

    dummy_state = torch.zeros(1, state_dim, dtype=torch.float32)
    dummy_image = torch.zeros(1, image_c, image_h, image_w, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_state, dummy_image),
            str(onnx_path),
            input_names=["state", "image"],
            output_names=["action_chunk"],
            dynamic_axes={
                "state": {0: "batch"},
                "image": {0: "batch"},
                "action_chunk": {0: "batch"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    declared_radar = cfg.input_features.get("observation.radar")
    radar_declared_dim = int(declared_radar.shape[0]) if declared_radar else 0
    radar_used = False
    warning = None
    if declared_radar is not None:
        warning = (
            "The trained ACT core only consumes observation.state and visual inputs. "
            "observation.radar is declared in config but is not wired into the ACT forward path "
            "for this checkpoint, so the exported ONNX model does not use radar."
        )

    deployment_fps = args.fps if args.fps is not None else infer_dataset_fps(policy_dir)
    if deployment_fps is None:
        deployment_fps = 20.0

    metadata = {
        "policy_dir": str(policy_dir),
        "onnx_path": str(onnx_path),
        "type": cfg.type,
        "state_dim": state_dim,
        "state_layout_hint": "observation.state as stored in the checkpoint config; "
        "for the merged-state dataset this is [vx, vy, wz, lidar_0, ..., lidar_89]",
        "image_shape_chw": [int(image_c), int(image_h), int(image_w)],
        "action_dim": action_dim,
        "n_action_steps": int(cfg.n_action_steps),
        "chunk_size": int(cfg.chunk_size),
        "deployment_fps": float(deployment_fps),
        "radar_declared_dim": radar_declared_dim,
        "radar_used": radar_used,
        "warning": warning,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported ONNX model to: {onnx_path}")
    print(f"Saved deployment metadata to: {metadata_path}")
    if warning:
        print(f"WARNING: {warning}")


def main() -> None:
    args = parse_args()
    export_to_onnx(args)


if __name__ == "__main__":
    main()
