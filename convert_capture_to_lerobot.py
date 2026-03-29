from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# 这里统一定义我们在 JSON 里使用的字段名。
# 这样后面如果你修改录制格式，只需要改这里，不用全文件到处搜字符串。
IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
RADAR_KEY = "observation.radar"
ACTION_KEY = "action"
TIMESTAMP_KEY = "timestamp"


def parse_args() -> argparse.Namespace:
    # 这个函数负责解析命令行参数。
    # 这样你在运行脚本时可以灵活指定：
    # 1. 原始 episode 数据放在哪里
    # 2. LeRobot 数据集输出到哪里
    # 3. fps / repo_id / robot_type 等元信息
    parser = argparse.ArgumentParser(
        description="Convert MuJoCo capture episodes into a LeRobotDataset."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/home/Actdog/capture"),
        help="Directory containing episode_xxxx folders.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/go2_follow_ball",
        help="LeRobot repo id written into metadata.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/Actdog/lerobot_dataset"),
        help="Where the LeRobotDataset will be created.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frame rate used during capture.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="quadruped",
        help="robot_type metadata for LeRobotDataset.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="follow the red ball while avoiding cylindrical obstacles",
        help="Task string stored for every frame.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=120,
        help="Expected image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=160,
        help="Expected image width.",
    )
    parser.add_argument(
        "--near-zero-threshold",
        type=float,
        default=0.02,
        help="If all action dimensions have abs(value) below this threshold, treat the frame as near-zero.",
    )
    parser.add_argument(
        "--keep-near-zero-every",
        type=int,
        default=10,
        help="Keep one out of every N near-zero frames. Set to 1 to keep all near-zero frames.",
    )
    parser.add_argument(
        "--min-frames-per-episode",
        type=int,
        default=10,
        help="If an episode has fewer kept frames than this after filtering, skip the episode.",
    )
    parser.add_argument(
        "--linear-active-threshold",
        type=float,
        default=0.1,
        help="If |vx| or |vy| is above this threshold, treat the frame as an active linear-motion frame.",
    )
    parser.add_argument(
        "--vxvy-boost-repeats",
        type=int,
        default=1,
        help="Duplicate frames with active vx/vy this many times. Set to 1 to disable boosting.",
    )
    return parser.parse_args()



def find_episode_jsons(input_root: Path) -> list[Path]:
    # 在输入目录下查找所有形如：
    # capture/episode_xxxx/episode_xxxx.json
    # 的 episode 描述文件。
    episode_jsons = sorted(input_root.glob("episode_*/*.json"))
    if not episode_jsons:
        raise FileNotFoundError(f"No episode json files found under: {input_root}")
    return episode_jsons


def load_episode(json_path: Path) -> dict:
    # 读取一个 episode 的 JSON 文件。
    # 这个 JSON 是你当前 C++ 录制逻辑导出的结果。
    with json_path.open("r", encoding="utf-8") as f:
        episode = json.load(f)

    # 检查关键字段是否都存在。
    # 如果缺字段，说明录制格式变了，或者某次录制不完整。
    required_keys = [STATE_KEY, RADAR_KEY, IMAGE_KEY, ACTION_KEY, TIMESTAMP_KEY]
    for key in required_keys:
        if key not in episode:
            raise KeyError(f"{json_path} is missing key: {key}")

    # 检查所有字段的帧数是否一致。
    # 例如：
    # state 有 300 帧
    # radar 也应该是 300 帧
    # image/action/timestamp 也都应该是 300 帧
    # 如果长度不一致，说明这一回合数据已经不对齐，不能安全转换。
    lengths = [len(episode[key]) for key in required_keys]
    if len(set(lengths)) != 1:
        raise ValueError(f"{json_path} has unaligned frame counts: {dict(zip(required_keys, lengths))}")

    return episode


def infer_shapes(first_episode: dict) -> tuple[int, int, int]:
    # 从第一个 episode 自动推断维度：
    # state 维度，例如 [vx, vy, wz] -> 3
    # radar 维度，例如 90 线雷达 -> 90
    # action 维度，例如 [cmd_vx, cmd_vy, cmd_wz] -> 3
    #
    # 这样脚本不用把这些数字写死，后面你扩字段会更方便。
    state_dim = len(first_episode[STATE_KEY][0])
    radar_dim = len(first_episode[RADAR_KEY][0])
    action_dim = len(first_episode[ACTION_KEY][0])
    return state_dim, radar_dim, action_dim


def create_dataset(
    repo_id: str,
    output_root: Path,
    fps: int,
    robot_type: str,
    image_height: int,
    image_width: int,
    state_dim: int,
    radar_dim: int,
    action_dim: int,
) -> LeRobotDataset:
    # 按照 LeRobot 官方库的方式创建一个标准数据集对象。
    #
    # 这里的 features 很关键：
    # 它定义了每个字段在最终数据集里的类型和形状。
    #
    # 你现在新的目标是：
    # 1. 把原始的 observation.state 和 observation.radar 拼成一个更长的 observation.state
    # 2. 图像仍然保留在 observation.images.front
    # 3. action 仍然保留为控制指令
    #
    # 这样做的原因是：
    # 当前 LeRobot ACT 实现只会真正消费名为 observation.state 的低维状态，
    # 不会自动把 observation.radar 当成额外 state token 送进模型。
    # 所以这里直接在数据集转换阶段把两者拼起来，后面训练时 ACT 就能真的用上雷达。
    #
    # 注意：
    # 你原始 JSON 里虽然有 timestamp，
    # 但当前版本的 LeRobotDataset 不接受你手动往 frame 里塞 "timestamp"。
    # 它会自己维护 episode/frame 的时间元数据。
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        fps=fps,
        robot_type=robot_type,
        features={
            STATE_KEY: {"dtype": "float32", "shape": (state_dim + radar_dim,)},
            IMAGE_KEY: {
                "dtype": "video",
                "shape": (image_height, image_width, 3),
                "names": ["height", "width", "channel"],
            },
            ACTION_KEY: {"dtype": "float32", "shape": (action_dim,)}
           
        },
        use_videos=True,
    )


def is_near_zero_action(action: np.ndarray, threshold: float) -> bool:
    # 如果这一帧动作的三个维度绝对值都很小，
    # 我们把它视为“近零动作帧”。
    #
    # 这类帧通常对应：
    # 1. 基本静止
    # 2. 很弱的修正
    #
    # 数据里这类帧过多时，行为克隆很容易学成“输出一个接近 0 的安全平均值”。
    return bool(np.all(np.abs(action) < threshold))


def build_kept_frame_indices(
    episode: dict,
    near_zero_threshold: float,
    keep_near_zero_every: int,
) -> tuple[list[int], int, int]:
    # 先扫描一遍 episode，决定每一帧保不保留。
    #
    # 策略：
    # 1. 非近零动作帧：全部保留
    # 2. 近零动作帧：只保留每 N 帧中的 1 帧
    #
    # 这样做的目标不是“删掉所有静止帧”，
    # 而是把它们的占比压下来，避免模型塌缩到接近常数输出。
    kept_indices: list[int] = []
    near_zero_total = 0
    near_zero_kept = 0

    for frame_idx, action_raw in enumerate(episode[ACTION_KEY]):
        action = np.asarray(action_raw, dtype=np.float32)
        if is_near_zero_action(action, near_zero_threshold):
            near_zero_total += 1
            if keep_near_zero_every <= 1 or (near_zero_total - 1) % keep_near_zero_every == 0:
                kept_indices.append(frame_idx)
                near_zero_kept += 1
        else:
            kept_indices.append(frame_idx)

    return kept_indices, near_zero_total, near_zero_kept


def is_active_linear_motion(
    action: np.ndarray,
    linear_active_threshold: float,
) -> bool:
    # 如果动作里的 vx 或 vy 足够大，
    # 我们把它视为“有效的平移动作帧”。
    #
    # 你当前数据里：
    # 1. vx/vy 的非零帧本来就偏少
    # 2. 尤其 vy 更稀疏
    #
    # 所以训练时可以通过“重复看到这些帧”的方式，
    # 让模型更关注真正发生平移控制的样本。
    return bool(
        abs(float(action[0])) >= linear_active_threshold
        or abs(float(action[1])) >= linear_active_threshold
    )


def convert_episode(
    dataset: LeRobotDataset,
    episode: dict,
    episode_dir: Path,
    task: str,
    near_zero_threshold: float,
    keep_near_zero_every: int,
    min_frames_per_episode: int,
    linear_active_threshold: float,
    vxvy_boost_repeats: int,
) -> tuple[int, int, int, int]:
    # 把一个 episode 的 JSON + 图片目录，逐帧写入 LeRobotDataset。
    #
    # episode_dir/
    #   episode_xxxx.json
    #   images/
    #     000000.png
    #     000001.png

    image_dir = episode_dir / "images"
    num_frames = len(episode[ACTION_KEY])
    kept_indices, near_zero_total, near_zero_kept = build_kept_frame_indices(
        episode,
        near_zero_threshold=near_zero_threshold,
        keep_near_zero_every=keep_near_zero_every,
    )

    # 过滤后如果这一回合只剩很少几帧，就干脆跳过。
    # 过短的 episode 对训练帮助很小，还可能让数据组织变得很碎。
    if len(kept_indices) < min_frames_per_episode:
        return 0, near_zero_total, near_zero_kept, 0

    boosted_frame_copies = 0

    for frame_idx in kept_indices:
        # 根据 JSON 里记录的图片文件名，找到这一帧对应的 PNG。
        image_name = episode[IMAGE_KEY][frame_idx]
        image_path = image_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for frame {frame_idx}: {image_path}")

        # LeRobotDataset 接受 PIL.Image / numpy array 形式的图像。
        # 这里把 PNG 读成 RGB 图像。
        image = Image.open(image_path).convert("RGB")

        # 先取出这一帧的低维状态和雷达。
        # 原始录制格式里它们是两个字段：
        #   observation.state = [vx, vy, wz]
        #   observation.radar = [r0, r1, ..., r89]
        #
        # 这里把它们拼成：
        #   observation.state = [vx, vy, wz, r0, r1, ..., r89]
        #
        # 这样后面的 ACT 训练就会把雷达也当成 state 的一部分真正送进模型。
        state = np.asarray(episode[STATE_KEY][frame_idx], dtype=np.float32)
        radar = np.asarray(episode[RADAR_KEY][frame_idx], dtype=np.float32)
        radar = np.where(radar < 0.0, 13.0, radar).astype(np.float32, copy=False)
        merged_state = np.concatenate([state, radar], axis=0)

        # 组装这一帧的载荷。
        #
        # 这里做了几件事：
        # 1. 把拼接后的 merged_state 转成 observation.state
        # 2. 把 image 放进 observation.images.front
        # 3. 把 action 转成 float32
        # 4. 加一个 task 字符串，后面训练时可以作为任务描述
        #
        # 这里不再把 timestamp 写进 frame_payload，
        # 因为当前 LeRobotDataset 版本会把它判定为“额外字段”。
        frame_payload = {
            STATE_KEY: merged_state,
            IMAGE_KEY: image,
            ACTION_KEY: np.asarray(episode[ACTION_KEY][frame_idx], dtype=np.float32),
            "task": task,
        }

        # 如果当前帧包含明显的 vx 或 vy 平移动作，
        # 就把它重复写入多次，起到“过采样增强”的效果。
        #
        # 这样做不会改变单帧内容，
        # 但会让训练时这类样本在一个 epoch 里出现得更频繁。
        action = np.asarray(episode[ACTION_KEY][frame_idx], dtype=np.float32)
        repeat_count = 1
        if vxvy_boost_repeats > 1 and is_active_linear_motion(action, linear_active_threshold):
            repeat_count = vxvy_boost_repeats
            boosted_frame_copies += repeat_count - 1

        # 把这一帧加入当前 episode。
        for _ in range(repeat_count):
            dataset.add_frame(frame_payload)

    # 一个 episode 的所有帧都加完之后，调用 save_episode()
    # 告诉 LeRobot：这一回合结束了，可以把这一回合整理并落盘。
    dataset.save_episode()
    return len(kept_indices) + boosted_frame_copies, near_zero_total, near_zero_kept, boosted_frame_copies


def main() -> None:
    # 主流程：
    # 1. 解析参数
    # 2. 找到所有 episode 的 JSON
    # 3. 用第一个 episode 自动推断 state/radar/action 的维度
    # 4. 创建 LeRobotDataset
    # 5. 逐回合、逐帧写入
    # 6. finalize() 生成最终元数据
    args = parse_args()

    # 找到所有可转换的 episode。
    episode_jsons = find_episode_jsons(args.input_root)

    # 读取第一个 episode，用来推断数据维度。
    first_episode = load_episode(episode_jsons[0])
    state_dim, radar_dim, action_dim = infer_shapes(first_episode)

    # 创建最终的数据集对象。
    dataset = create_dataset(
        repo_id=args.repo_id,
        output_root=args.output_root,
        fps=args.fps,
        robot_type=args.robot_type,
        image_height=args.height,
        image_width=args.width,
        state_dim=state_dim,
        radar_dim=radar_dim,
        action_dim=action_dim,
    )

    total_original_frames = 0
    total_kept_frames = 0
    total_near_zero_frames = 0
    total_near_zero_kept = 0
    total_boosted_copies = 0
    saved_episode_count = 0

    # 逐个 episode 转换。
    for episode_idx, json_path in enumerate(episode_jsons):
        episode = load_episode(json_path)
        total_original_frames += len(episode[ACTION_KEY])

        kept_frames, near_zero_total, near_zero_kept, boosted_copies = convert_episode(
            dataset,
            episode,
            json_path.parent,
            args.task,
            near_zero_threshold=args.near_zero_threshold,
            keep_near_zero_every=args.keep_near_zero_every,
            min_frames_per_episode=args.min_frames_per_episode,
            linear_active_threshold=args.linear_active_threshold,
            vxvy_boost_repeats=args.vxvy_boost_repeats,
        )

        total_kept_frames += kept_frames
        total_near_zero_frames += near_zero_total
        total_near_zero_kept += near_zero_kept
        total_boosted_copies += boosted_copies

        if kept_frames > 0:
            saved_episode_count += 1
            print(
                f"saved episode {episode_idx}: {json_path} "
                f"(kept {kept_frames}/{len(episode[ACTION_KEY])} frames, "
                f"near-zero kept {near_zero_kept}/{near_zero_total}, "
                f"vx/vy boosted +{boosted_copies})"
            )
        else:
            print(
                f"skipped episode {episode_idx}: {json_path} "
                f"(kept 0/{len(episode[ACTION_KEY])} frames after filtering)"
            )

    # finalize() 会整理 LeRobotDataset 的元数据、索引和统计信息。
    # 这是转换完成时必须调用的最后一步。
    dataset.finalize()
    print(f"LeRobot dataset saved to: {args.output_root}")
    print(f"episodes saved: {saved_episode_count}/{len(episode_jsons)}")
    print(f"frames kept: {total_kept_frames}/{total_original_frames}")
    print(f"near-zero frames kept: {total_near_zero_kept}/{total_near_zero_frames}")
    print(f"extra vx/vy boosted copies: {total_boosted_copies}")


if __name__ == "__main__":
    main()


# visualize
# lerobot-dataset-viz   \
# --root /home/Actdog/lerobot_dataset \
# --repo-id lerobot/go2_follow_ball   \
# --episode-index 5

# lerobot-train   --dataset.root=/home/Actdog/lerobot_dataset   \
# --dataset.repo_id=lerobot/go2_follow_ball   \
# --policy.type=act   \
# --policy.repo_id=zhangzhong/myrobot_act_policy   \
# --output_dir=/home/Actdog/policy2   \
# --policy.push_to_hub=False  \
# --policy.device=cuda \
# --policy.kl_weight=1.0 \
# --policy.chunk_size=10 \
# --policy.n_action_steps=10 \
# --policy.use_vae=False 




# # /root/gpufree-share/lerobot_dataset


# lerobot-train   --dataset.root=/root/gpufree-share/lerobot_dataset   \
# --dataset.repo_id=lerobot/go2_follow_ball   \
# --policy.type=act   \
# --policy.repo_id=zhangzhong/myrobot_act_policy   \
# --output_dir=/root/gpufree-share/policy   \
# --policy.push_to_hub=False  \
#  --policy.device=cuda
