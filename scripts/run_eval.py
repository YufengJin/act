#!/usr/bin/env python3
"""
run_eval.py -- act evaluation client (WebSocket).

Runs the act bimanual sim loop across N rollouts, delegates action inference
to a Policy Server over WebSocket via `policy-websocket`. The client sends
raw observations (qpos, images, task description); the server handles
remapping/chunking and returns a 14-dim action per query. Reports per-rollout
and aggregate success rate (reward == env.task.max_reward at any step).

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
    # host (spins up a random-policy server that advertises action_dim=14)
    python tests/test_random_policy_server.py --port 8000

    # container: eval Transfer Cube, 10 rollouts
    python scripts/run_eval.py \\
        --task_name sim_transfer_cube_scripted \\
        --policy_server_addr localhost:8000 --num_rollouts 10

    # container: eval Bimanual Insertion, 10 rollouts
    python scripts/run_eval.py \\
        --task_name sim_insertion_scripted \\
        --policy_server_addr localhost:8000 --num_rollouts 10
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


def log(msg: str, log_file=None):
    print(msg, flush=True)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


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


def save_rollout_video(frames_by_cam: list, episode_idx: int, success: bool,
                       task_name: str, output_dir: str):
    """Save an MP4 of the first camera (top), one frame per env step."""
    os.makedirs(output_dir, exist_ok=True)
    if not frames_by_cam:
        return None
    cam_name = next(iter(frames_by_cam[0]))
    frames = [f[cam_name] for f in frames_by_cam if cam_name in f]
    if not frames:
        return None
    mp4_path = os.path.join(
        output_dir,
        f"episode={episode_idx}--success={success}--task={task_name}.mp4",
    )
    writer = imageio.get_writer(mp4_path, fps=50, format="FFMPEG", codec="libx264")
    for f in frames:
        writer.append_data(f)
    writer.close()
    return mp4_path


def run_episode(env, policy, task_name, max_timesteps, onscreen_render, log_file=None):
    """Run a single evaluation episode. Returns (success, length, frames, rewards)."""
    import matplotlib.pyplot as plt

    set_initial_pose(task_name)
    ts = env.reset()
    policy.reset()

    env_max_reward = env.task.max_reward

    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id="angle"))
        plt.ion()

    frames = []
    rewards = []
    highest_reward = 0.0

    for t in range(max_timesteps):
        if onscreen_render:
            plt_img.set_data(env._physics.render(height=480, width=640, camera_id="angle"))
            plt.pause(0.001)

        obs = ts.observation
        images = extract_images(obs)
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
        query_time = time.time() - start

        if action.shape[0] != ACTION_DIM:
            raise ValueError(
                f"Policy returned action of shape {action.shape}; expected ({ACTION_DIM},)"
            )

        if t % 50 == 0:
            log(f"  t={t}: infer {query_time*1000:.1f}ms, action[:4]={action[:4]}", log_file)

        ts = env.step(action)
        r = ts.reward if ts.reward is not None else 0.0
        rewards.append(float(r))
        if r > highest_reward:
            highest_reward = float(r)
        # success = reached the maximum task reward at any step of the episode
        if highest_reward >= env_max_reward:
            log(f"  Success at t={t} (reward={highest_reward}).", log_file)
            break

    if onscreen_render:
        plt.close()

    success = highest_reward >= env_max_reward
    return success, len(rewards), frames, rewards, highest_reward


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "act WebSocket evaluation client.\n\n"
            "Two tasks supported:\n"
            "  * sim_transfer_cube_{scripted,human}  -- left arm passes cube to right arm\n"
            "  * sim_insertion_{scripted,human}      -- insert peg into socket (bimanual)\n"
            "Success = reach env.task.max_reward at any step of the episode."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--policy_server_addr", type=str, default="localhost:8000",
        help="WebSocket policy server host:port",
    )
    parser.add_argument("--policy", type=str, default="randomPolicy",
                        help="Policy name (for logging only)")
    parser.add_argument(
        "--task_name", type=str, default="sim_transfer_cube_scripted",
        choices=list(SIM_TASK_CONFIGS.keys()),
        help="Which act task to evaluate (see docstring for the 2-task taxonomy)",
    )
    parser.add_argument("--num_rollouts", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--log_dir", type=str, default="./eval_logs")
    parser.add_argument("--save_video", action="store_true", default=True)
    parser.add_argument("--no_save_video", dest="save_video", action="store_false")
    parser.add_argument("--onscreen_render", action="store_true",
                        help="Requires X11 compose profile")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    task_cfg = SIM_TASK_CONFIGS[args.task_name]
    max_timesteps = task_cfg["episode_len"]

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.log_dir, f"{args.task_name}--{date_str}")
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "eval.log")
    log_file = open(log_path, "w")
    log("=" * 60, log_file)
    log("act WebSocket Eval Run", log_file)
    log("=" * 60, log_file)
    log(f"  policy:           {args.policy}", log_file)
    log(f"  task_name:        {args.task_name}", log_file)
    log(f"  num_rollouts:     {args.num_rollouts}", log_file)
    log(f"  episode_len:      {max_timesteps}", log_file)
    log(f"  policy_server:    {args.policy_server_addr}", log_file)
    log(f"  run_dir:          {run_dir}", log_file)
    log(f"  save_video:       {args.save_video}", log_file)
    log("=" * 60, log_file)

    addr = args.policy_server_addr
    host, port = (addr.rsplit(":", 1) if ":" in addr else (addr, "8000"))
    port = int(port)
    log(f"Connecting to policy server at ws://{host}:{port} ...", log_file)

    policy = WebsocketClientPolicy(host=host, port=port)
    metadata = policy.get_server_metadata()
    log(f"Server metadata: {metadata}", log_file)
    policy_dim = metadata.get("action_dim") if isinstance(metadata, dict) else None
    if policy_dim is not None and int(policy_dim) != ACTION_DIM:
        raise ValueError(
            f"task_name={args.task_name} expects action_dim={ACTION_DIM}, "
            f"but server returned {policy_dim}."
        )

    def _cleanup(signum=None, frame=None):
        print("\nCleaning up ...", flush=True)
        try:
            policy.close()
        except Exception:
            pass
        if not log_file.closed:
            log_file.close()
        sys.stdout.flush()
        sys.stderr.flush()
        if signum is not None:
            os._exit(1)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
    atexit.register(lambda: policy.close())

    env = make_sim_env(args.task_name)
    env_max_reward = env.task.max_reward
    log(f"env_max_reward = {env_max_reward}", log_file)

    successes = []
    lengths = []
    highest_rewards = []

    try:
        for rollout_id in range(args.num_rollouts):
            log(f"\n--- Rollout {rollout_id + 1}/{args.num_rollouts} ---", log_file)
            success, length, frames, rewards, high_r = run_episode(
                env=env,
                policy=policy,
                task_name=args.task_name,
                max_timesteps=max_timesteps,
                onscreen_render=args.onscreen_render,
                log_file=log_file,
            )
            successes.append(success)
            lengths.append(length)
            highest_rewards.append(high_r)

            log(
                f"  Episode {rollout_id}: {'SUCCESS' if success else 'FAILURE'} "
                f"(length={length}, highest_reward={high_r}/{env_max_reward})",
                log_file,
            )

            if args.save_video:
                mp4 = save_rollout_video(frames, rollout_id, success, args.task_name, run_dir)
                if mp4:
                    log(f"  Saved {mp4}", log_file)
            del frames
            gc.collect()

        success_rate = float(np.mean(successes)) if successes else 0.0
        avg_length = float(np.mean(lengths)) if lengths else 0.0

        log("\n" + "=" * 60, log_file)
        log("FINAL RESULTS", log_file)
        log("=" * 60, log_file)
        log(f"Policy:               {args.policy}", log_file)
        log(f"Task:                 {args.task_name}", log_file)
        log(f"Rollouts:             {args.num_rollouts}", log_file)
        log(f"Success rate:         {success_rate*100:.1f}%", log_file)
        log(f"Avg episode length:   {avg_length:.1f}", log_file)
        for r in range(int(env_max_reward) + 1):
            hits = int(sum(1 for h in highest_rewards if h >= r))
            log(f"reward >= {r}: {hits}/{args.num_rollouts} = {hits / args.num_rollouts * 100:.1f}%", log_file)
        log("=" * 60, log_file)
        log(f"Log saved to: {log_path}", log_file)
        return success_rate
    finally:
        try:
            policy.close()
        except Exception:
            pass
        if not log_file.closed:
            log_file.close()


if __name__ == "__main__":
    main()
