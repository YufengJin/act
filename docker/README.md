# act Docker Guide

Build and run act (ACT policy + 2 bimanual MuJoCo tasks) in a GPU container with an editable uv install. Evaluation delegates action inference to a host-side WebSocket policy server via `policy-websocket`.

## Prerequisites

- Docker + Docker Compose (v2)
- NVIDIA Container Toolkit
- For X11 mode: an X server on the host

## Python environment

The image uses [uv](https://github.com/astral-sh/uv) to install dependencies into `/opt/venv` (at build time via `uv sync --frozen`, matching the repo root `pyproject.toml` / `uv.lock`). `PATH` includes `/opt/venv/bin`; inside the container you can run `python` directly without activating conda or mamba.

Notable pins (from `pyproject.toml`): Python 3.9, torch 2.0.0+cu118, torchvision 0.15.0+cu118, mujoco 2.3.3, dm_control 1.0.9, `policy-websocket @ git+https://github.com/YufengJin/policy_websocket.git`. The `detr` subpackage is a uv workspace member.

## Build

From the project root:

```bash
cd docker
docker compose -f docker-compose.headless.yaml build
# or x11
docker compose -f docker-compose.x11.yaml build
```

Custom image tag:

```bash
IMAGE=act:dev docker compose -f docker-compose.headless.yaml build
```

## Start

### Headless (batch eval, CI)

```bash
cd docker
docker compose -f docker-compose.headless.yaml up -d
```

Uses `MUJOCO_GL=osmesa` for offscreen rendering — no X server required.

### X11 (GUI visualization with `--onscreen_render`)

```bash
cd docker
xhost +local:docker          # one-time per boot
docker compose -f docker-compose.x11.yaml up -d
```

Headless and X11 share `container_name: act_container`, so only one can run at a time. Stop the other first with `docker compose down`.

## Attach

```bash
docker exec -it act_container bash
```

## Stop

```bash
cd docker
docker compose -f docker-compose.headless.yaml down
```

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `IMAGE` | `act:latest` | Override image tag at build/run |
| `GPU` | `all` | GPU selection (`0`, `0,1`, ...) |
| `DISPLAY` | `$DISPLAY` | X11 display (x11 mode only) |
| `MUJOCO_GL` | `osmesa` (headless) / `glfw` (x11) | MuJoCo rendering backend |

## Verify runtime

Inside the container:

```bash
nvidia-smi | head -5
python -c "import detr; print(detr.__file__)"                 # /workspace/detr/...
python -c "import sim_env, constants; print(sim_env.__file__)"  # /workspace/sim_env.py
python -c "import policy_websocket; print(policy_websocket.__version__)"
```

## Smoke test

Headless-safe end-to-end sanity check — builds a plain env, resets, steps 10 times, prints `smoke test OK`. No dataset, no GUI, no network.

```bash
docker exec -it act_container python -c "
import numpy as np
from sim_env import make_sim_env, BOX_POSE
BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]
env = make_sim_env('sim_transfer_cube_scripted')
env.reset()
for _ in range(10):
    action = np.concatenate([np.zeros(6), [0.5], np.zeros(6), [0.5]])
    env.step(action)
print('smoke test OK')
"
```

If this prints `smoke test OK`, the container is runnable. If it errors, the image built but the runtime is broken — do not ship.

## Tasks

act ships **two bimanual MuJoCo tasks**, each with two dataset variants (scripted / human). The env geometry is identical; `scripted` vs `human` only changes the demo distribution you train on.

| Task | `task_name` values | XML | `episode_len` | `max_reward` | Goal |
|---|---|---|---|---|---|
| **Transfer Cube** | `sim_transfer_cube_scripted`, `sim_transfer_cube_human` | `assets/bimanual_viperx_transfer_cube.xml` | 400 | 4 | Left arm picks up the cube and hands it off to the right arm. |
| **Bimanual Insertion** | `sim_insertion_scripted`, `sim_insertion_human` | `assets/bimanual_viperx_insertion.xml` | 400 / 500 | 4 | Left arm holds socket, right arm holds peg, peg inserted into socket. |

Both expose a 14-D action: `[left_arm_qpos(6), left_gripper(1), right_arm_qpos(6), right_gripper(1)]`.

## Run demo / eval against a WebSocket policy server

Both `scripts/run_demo.py` and `scripts/run_eval.py` are WebSocket clients and target the same two tasks above. The shipped `tests/test_random_policy_server.py` is a 14-D random policy you can use to smoke-test the protocol:

```bash
# Terminal 1 — policy server (host, or inside another container process)
python tests/test_random_policy_server.py --port 8000

# Terminal 2 — record a 3-rollout Transfer Cube demo (saves MP4s to demo_log/)
docker exec -it act_container python scripts/run_demo.py \
    --task_name sim_transfer_cube_scripted \
    --policy_server_addr localhost:8000 --num_resets 3

# Terminal 2 — evaluate 10 rollouts on Bimanual Insertion (writes eval.log)
docker exec -it act_container python scripts/run_eval.py \
    --task_name sim_insertion_scripted \
    --policy_server_addr localhost:8000 --num_rollouts 10
```

Expected on a successful wiring: `Server metadata: {... "action_dim": 14}`, rollouts complete, per-rollout SUCCESS/FAILURE printed. `run_eval.py` additionally prints final success-rate bucketed by reward level and writes everything to `eval_logs/<task>--<ts>/eval.log`.

`network_mode: host` in compose is why `localhost:<port>` reaches a host server from inside the container. See `scripts/run_demo.py --help` / `scripts/run_eval.py --help` for all flags.

## Troubleshooting

### GPU not detected
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 nvidia-smi
```

### Container name conflict
```bash
docker stop act_container && docker rm act_container
```

### MuJoCo rendering errors
Headless: ensure `MUJOCO_GL=osmesa` is set (the compose file does this; `libosmesa6` is in the apt list).
X11: ensure `xhost +local:docker` ran on the host and `DISPLAY` is exported before `docker compose up`.

### `uv.lock` out of sync after editing `pyproject.toml`
Host side: `uv lock` to regenerate, then rebuild the image (the dep layer is cache-invalidated).

## Notes

- `/workspace` bind-mounts the repo root; host edits reflect immediately (editable install).
- The image ships no training data or checkpoints — mount them into the container or place them under the repo root.
- The WebSocket protocol is the only eval entry point in this image. If you want to eval ACT's own `policy_best.ckpt`, either write a thin server that wraps it or use the upstream `python imitate_episodes.py --eval --ckpt_dir ...` path directly.
