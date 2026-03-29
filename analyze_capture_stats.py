from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


STATE_KEY = "observation.state"
RADAR_KEY = "observation.radar"
ACTION_KEY = "action"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze action distribution and state-action correlation in captured episodes."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/home/Actdog/capture"),
        help="Directory containing episode_xxxx folders.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Threshold used when reporting nonzero action ratios.",
    )
    return parser.parse_args()


def find_episode_jsons(input_root: Path) -> list[Path]:
    episode_jsons = sorted(input_root.glob("episode_*/*.json"))
    if not episode_jsons:
        raise FileNotFoundError(f"No episode json files found under: {input_root}")
    return episode_jsons


def load_episode(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        episode = json.load(f)

    if STATE_KEY not in episode or ACTION_KEY not in episode or RADAR_KEY not in episode:
        raise KeyError(f"{json_path} is missing required keys.")

    if len(episode[STATE_KEY]) != len(episode[ACTION_KEY]) or len(episode[RADAR_KEY]) != len(episode[ACTION_KEY]):
        raise ValueError(
            f"{json_path} has mismatched lengths: "
            f"state={len(episode[STATE_KEY])}, "
            f"radar={len(episode[RADAR_KEY])}, "
            f"action={len(episode[ACTION_KEY])}"
        )
    return episode


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    # 如果某一维全是常数，numpy 的 corrcoef 会给 nan。
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summarize_dim(name: str, values: np.ndarray, threshold: float) -> None:
    abs_values = np.abs(values)
    quantiles = np.quantile(values, [0.1, 0.5, 0.9])
    print(name)
    print(f"  mean {float(np.mean(values)):.6f}")
    print(f"  std  {float(np.std(values)):.6f}")
    print(f"  min  {float(np.min(values)):.6f}")
    print(f"  max  {float(np.max(values)):.6f}")
    print(f"  abs<{threshold:.2f} {float(np.mean(abs_values < threshold)):.6f}")
    print(f"  abs<0.05 {float(np.mean(abs_values < 0.05)):.6f}")
    print(f"  abs>0.10 {float(np.mean(abs_values > 0.10)):.6f}")
    print(f"  abs>0.20 {float(np.mean(abs_values > 0.20)):.6f}")
    print(
        "  q10 q50 q90 "
        f"[{float(quantiles[0]):.6f}, {float(quantiles[1]):.6f}, {float(quantiles[2]):.6f}]"
    )


def summarize_radar_action_correlation(radars: np.ndarray, actions: np.ndarray) -> None:
    # 这里统计两类信息：
    # 1. 单根 beam 和动作的相关性，看看哪几根最“有信息”
    # 2. 按前/左/后/右四个扇区取平均距离，再看和动作的相关性
    #
    # 注意这里把 no-hit 的 -1 在调用前已经替换成 13.0，
    # 避免“没打到东西”被误当成一个极近距离。
    print("=== Radar-Action Correlation ===")

    for action_idx, action_name in enumerate(["vx", "vy", "wz"]):
        corrs = []
        for beam_idx in range(radars.shape[1]):
            corrs.append(safe_corr(radars[:, beam_idx], actions[:, action_idx]))

        corrs = np.asarray(corrs, dtype=np.float64)
        abs_corrs = np.abs(corrs)
        top_indices = np.argsort(-abs_corrs)[:10]

        print(action_name)
        print(f"  mean_abs_corr {float(np.nanmean(abs_corrs)):.6f}")
        print(f"  max_abs_corr  {float(np.nanmax(abs_corrs)):.6f}")
        print("  top10 beams (beam_index: corr)")
        for beam_idx in top_indices:
            print(f"    {int(beam_idx)}: {float(corrs[beam_idx]):.6f}")

    angles = np.arange(radars.shape[1]) * 4.0
    sectors = {
        "front": (angles <= 40) | (angles >= 320),
        "left": (angles >= 50) & (angles <= 130),
        "back": (angles >= 140) & (angles <= 220),
        "right": (angles >= 230) & (angles <= 310),
    }

    print()
    print("  sector mean-distance correlation")
    for sector_name, mask in sectors.items():
        sector_values = radars[:, mask].mean(axis=1)
        print(f"  {sector_name}")
        print(f"    vx {safe_corr(sector_values, actions[:, 0]):.6f}")
        print(f"    vy {safe_corr(sector_values, actions[:, 1]):.6f}")
        print(f"    wz {safe_corr(sector_values, actions[:, 2]):.6f}")


def main() -> None:
    args = parse_args()
    episode_jsons = find_episode_jsons(args.input_root)

    all_states: list[np.ndarray] = []
    all_radars: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    per_episode_nonzero: list[np.ndarray] = []

    for json_path in episode_jsons:
        episode = load_episode(json_path)
        states = np.asarray(episode[STATE_KEY], dtype=np.float32)
        radars = np.asarray(episode[RADAR_KEY], dtype=np.float32)
        actions = np.asarray(episode[ACTION_KEY], dtype=np.float32)

        # 这里只分析前三维 state，因为你原始录制格式里这三维就是 [vx, vy, wz]。
        # 后面如果你把录制格式改掉了，这里也能及时暴露问题。
        if states.ndim != 2 or states.shape[1] < 3:
            raise ValueError(f"{json_path} has invalid state shape: {states.shape}")
        if radars.ndim != 2 or radars.shape[1] != 90:
            raise ValueError(f"{json_path} has invalid radar shape: {radars.shape}")
        if actions.ndim != 2 or actions.shape[1] != 3:
            raise ValueError(f"{json_path} has invalid action shape: {actions.shape}")

        states_3 = states[:, :3]
        all_states.append(states_3)
        all_radars.append(np.where(radars < 0.0, 13.0, radars).astype(np.float32, copy=False))
        all_actions.append(actions)
        per_episode_nonzero.append(np.mean(np.abs(actions) > args.threshold, axis=0))

    states = np.concatenate(all_states, axis=0)
    radars = np.concatenate(all_radars, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    per_episode_nonzero_arr = np.stack(per_episode_nonzero, axis=0)

    print(f"episodes: {len(episode_jsons)}")
    print(f"frames: {states.shape[0]}")
    print()
    print("=== Action Distribution ===")
    summarize_dim("vx", actions[:, 0], args.threshold)
    summarize_dim("vy", actions[:, 1], args.threshold)
    summarize_dim("wz", actions[:, 2], args.threshold)
    print()
    print("=== Mean Per-Episode Nonzero Ratio ===")
    print(f"vx {float(np.mean(per_episode_nonzero_arr[:, 0])):.6f}")
    print(f"vy {float(np.mean(per_episode_nonzero_arr[:, 1])):.6f}")
    print(f"wz {float(np.mean(per_episode_nonzero_arr[:, 2])):.6f}")
    print()
    print("=== State-Action Correlation ===")
    print(f"vx {safe_corr(states[:, 0], actions[:, 0]):.6f}")
    print(f"vy {safe_corr(states[:, 1], actions[:, 1]):.6f}")
    print(f"wz {safe_corr(states[:, 2], actions[:, 2]):.6f}")
    print()
    summarize_radar_action_correlation(radars, actions)


if __name__ == "__main__":
    main()
