#!/usr/bin/env python3
"""
Test policy server for act — returns random 14-D bimanual actions via WebSocket.

Usage
-----
    python tests/test_random_policy_server.py --port 8765

Then from the client:
    python scripts/run_eval.py --task_name sim_transfer_cube_scripted \\
        --policy_server_addr localhost:8765 --n-episodes 2

Action layout (14-D):
    [ left_arm_qpos(6), left_gripper(1),
      right_arm_qpos(6), right_gripper(1) ]
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from policy_websocket import BasePolicy, WebsocketPolicyServer


logger = logging.getLogger(__name__)


START_LEFT = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0], dtype=np.float32)
START_RIGHT = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0], dtype=np.float32)
ACTION_DIM = 14


class RandomPolicy(BasePolicy):
    """Random small perturbations around the start pose. Grippers mid-range.

    This never solves the task — it exists to exercise the eval wiring
    (client handshake, obs serialization, action deserialization, env stepping).
    """

    def __init__(self, noise_scale: float = 0.02) -> None:
        self._noise_scale = float(noise_scale)

    def infer(self, obs: Dict) -> Dict:
        noise_l = np.random.uniform(-self._noise_scale, self._noise_scale, size=6).astype(np.float32)
        noise_r = np.random.uniform(-self._noise_scale, self._noise_scale, size=6).astype(np.float32)
        action = np.concatenate([
            START_LEFT + noise_l,  # left arm 6
            [0.5],                 # left gripper (normalized mid-open)
            START_RIGHT + noise_r, # right arm 6
            [0.5],                 # right gripper
        ]).astype(np.float32)
        assert action.shape[0] == ACTION_DIM, action.shape
        return {"actions": action}

    def reset(self) -> None:
        pass


def main():
    parser = argparse.ArgumentParser(description="act test policy server (random actions)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--noise_scale", type=float, default=0.02,
                        help="Uniform noise magnitude on arm qpos around START pose")
    args = parser.parse_args()

    policy = RandomPolicy(noise_scale=args.noise_scale)
    metadata = {"policy_name": "RandomPolicy(act)", "action_dim": ACTION_DIM}

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    print(f"Starting act RandomPolicy server on ws://{args.host}:{args.port}")
    print(f"Advertising action_dim={ACTION_DIM}. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print("Server stopped, port released.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
