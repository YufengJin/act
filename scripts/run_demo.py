#!/usr/bin/env python3
"""
run_demo.py -- act demo client (WebSocket).

Runs a single act bimanual task in simulation, delegates action inference to a
Policy Server over WebSocket via `policy-websocket`. Demo only — no success
tracking, no aggregate metrics. Default: headless, saves one MP4 per reset
under `demo_log/`. Pass `--gui` for an on-screen GLFW window (X11 compose).

act has two bimanual sim tasks:

    Task                | task_name (scripted / human data) | XML                               | max_reward
    --------------------|-----------------------------------|-----------------------------------|-----------
    Transfer Cube       | sim_transfer_cube_{scripted,human}| bimanual_viperx_transfer_cube.xml | 4
    Bimanual Insertion  | sim_insertion_{scripted,human}    | bimanual_viperx_insertion.xml     | 4

Action layout (14-D, absolute joint positions + normalized gripper):
    [ left_arm_qpos(6), left_gripper(1),
      right_arm_qpos(6), right_gripper(1) ]

Usage
-----
    # host: start a random 14-D policy server
    python tests/test_random_policy_server.py --port 8000

    # container: record 3 rollouts of Transfer Cube
    python scripts/run_demo.py \\
        --task_name sim_transfer_cube_scripted \\
        --policy_server_addr localhost:8000 --num_resets 3

    # container: record 1 rollout of Insertion with on-screen GUI (X11 mode)
    python scripts/run_demo.py --gui \\
        --task_name sim_insertion_scripted \\
        --policy_server_addr localhost:8000 --num_resets 1
"""

import argparse
import atexit
import gc
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constants import SIM_TASK_CONFIGS
from policy_websocket import WebsocketClientPolicy
from sim_env import BOX_POSE, make_sim_env
from utils import sample_box_pose, sample_insertion_pose, set_seed


ACTION_DIM = 14


def set_initial_pose(task_name: str):
    if "sim_transfer_cube" in task_name:
        BOX_POSE[0] = sample_box_pose()
    elif "sim_insertion" in task_name:
        BOX_POSE[0] = np.concatenate(sample_insertion_pose())


def extract_images(obs) -> dict:
    if "images" in obs:
        return {name: np.asarray(img) for name, img in obs["images"].items()}
    if "image" in obs:
        return {"main": np.asarray(obs["image"])}
    return {}


def save_rollout_video(frames: list, reset_idx: int, task_name: str, output_dir: str):
    """Save an MP4 of the first (top) camera stream over one reset."""
    os.makedirs(output_dir, exist_ok=True)
    if not frames:
        return None
    cam_name = next(iter(frames[0]))
    stream = [f[cam_name] for f in frames if cam_name in f]
    if not stream:
        return None
    mp4_path = os.path.join(output_dir, f"reset={reset_idx}--task={task_name}--cam={cam_name}.mp4")
    writer = imageio.get_writer(mp4_path, fps=50, format="FFMPEG", codec="libx264")
    for f in stream:
        writer.append_data(f)
    writer.close()
    print(f"Saved demo video: {mp4_path}")
    return mp4_path


def run_reset(env, policy, task_name, max_timesteps, use_gui, save_video):
    import matplotlib.pyplot as plt

    set_initial_pose(task_name)
    ts = env.reset()
    policy.reset()
    env_max_reward = env.task.max_reward

    if use_gui:
        ax = plt.subplot()
        plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id="angle"))
        plt.ion()

    frames = []
    highest_reward = 0.0

    for t in range(max_timesteps):
        if use_gui:
            plt_img.set_data(env._physics.render(height=480, width=640, camera_id="angle"))
            plt.pause(0.001)

        obs = ts.observation
        images = extract_images(obs)
        if save_video:
            frames.append(images)

        observation = {
            "qpos": np.asarray(obs["qpos"], dtype=np.float32),
            "qvel": np.asarray(obs.get("qvel", []), dtype=np.float32),
            "images": images,
            "task_description": task_name,
        }
        start = time.time()
        result = policy.infer(observation)
        action = np.asarray(result["actions"], dtype=np.float32)
        if action.ndim > 1:
            action = action[0]
        if action.shape[0] != ACTION_DIM:
            raise ValueError(f"Policy returned action of shape {action.shape}; expected ({ACTION_DIM},)")

        if t % 50 == 0:
            print(f"  t={t}: infer {(time.time() - start) * 1000:.1f}ms, action[:4]={action[:4]}")

        ts = env.step(action)
        r = ts.reward if ts.reward is not None else 0.0
        if r > highest_reward:
            highest_reward = float(r)
        if highest_reward >= env_max_reward:
            print(f"  Success at t={t} (reward={highest_reward}).")
            break

    if use_gui:
        plt.close()

    return highest_reward >= env_max_reward, len(frames), frames, highest_reward


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "act demo client: run a policy in sim via WebSocket, no eval.\n\n"
            "Two tasks supported:\n"
            "  * sim_transfer_cube_{scripted,human}  -- left arm passes cube to right arm\n"
            "  * sim_insertion_{scripted,human}      -- insert peg into socket (bimanual)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--policy_server_addr", type=str, default="localhost:8000",
        help="WebSocket policy server host:port",
    )
    parser.add_argument("--policy", type=str, default="randomPolicy",
                        help="Policy name (logging only)")
    parser.add_argument(
        "--task_name", type=str, default="sim_transfer_cube_scripted",
        choices=list(SIM_TASK_CONFIGS.keys()),
        help="Which act task to roll out (see docstring for the 2-task taxonomy)",
    )
    parser.add_argument("--num_resets", type=int, default=3,
                        help="Number of rollouts (env.reset cycles) to record")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true",
                        help="On-screen GLFW rendering (requires X11 compose profile)")
    parser.add_argument("--demo_log_dir", type=str, default="./demo_log",
                        help="Where to save MP4s (headless mode only)")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    task_cfg = SIM_TASK_CONFIGS[args.task_name]
    max_timesteps = task_cfg["episode_len"]

    addr = args.policy_server_addr
    host, port = (addr.rsplit(":", 1) if ":" in addr else (addr, "8000"))
    port = int(port)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.demo_log_dir, f"{args.task_name}--{date_str}") if not args.gui else ""

    print("=" * 60)
    print("act Demo (run policy in sim, no eval)")
    print("=" * 60)
    print(f"  task_name:      {args.task_name}")
    print(f"  episode_len:    {max_timesteps}")
    print(f"  num_resets:     {args.num_resets}")
    print(f"  policy:         {args.policy}")
    print(f"  policy_server:  ws://{host}:{port}")
    print(f"  GUI:            {'on (--gui)' if args.gui else 'off (headless, videos saved)'}")
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        print(f"  demo_log_dir:   {run_dir}")
    print("=" * 60)

    policy = WebsocketClientPolicy(host=host, port=port)
    metadata = policy.get_server_metadata()
    print(f"Server metadata: {metadata}")
    policy_dim = metadata.get("action_dim") if isinstance(metadata, dict) else None
    if policy_dim is not None and int(policy_dim) != ACTION_DIM:
        raise ValueError(
            f"task_name={args.task_name} expects action_dim={ACTION_DIM}, "
            f"but server returned {policy_dim}."
        )

    env = None

    def _cleanup(signum=None, frame=None):
        print("\nCleaning up ...", flush=True)
        try:
            policy.close()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        if signum is not None:
            os._exit(1)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
    atexit.register(lambda: policy.close())

    try:
        env = make_sim_env(args.task_name)
        save_video = not args.gui

        for ep_idx in range(args.num_resets):
            print(f"\n--- Reset {ep_idx + 1}/{args.num_resets} ---")
            success, length, frames, highest_reward = run_reset(
                env=env,
                policy=policy,
                task_name=args.task_name,
                max_timesteps=max_timesteps,
                use_gui=args.gui,
                save_video=save_video,
            )
            print(
                f"  Reset {ep_idx}: {'SUCCESS' if success else 'FAILURE'} "
                f"(length={length}, highest_reward={highest_reward}/{env.task.max_reward})"
            )
            if save_video:
                save_rollout_video(frames, ep_idx, args.task_name, run_dir)
            del frames
            gc.collect()
    finally:
        try:
            policy.close()
        except Exception:
            pass

    print("\nDone.")


if __name__ == "__main__":
    main()
