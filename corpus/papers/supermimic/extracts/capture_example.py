"""Roll out the local GenCoord actor and write a frame-aligned example sidecar.

Frame i shows the sim RGB before the step, the actor's 29-d action, and the
logged full state after that step. auto_reset is off so a terminal step keeps
the post-step state.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import mjlab.tasks  # noqa: F401
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer.viewer_config import ViewerConfig

import supermimic.tasks  # noqa: F401
from supermimic.representations import extract_actor_goal
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.tasks.task_spec import build_actor_obs_td

PAPER = Path("/mnt/d/robotic_research/corpus/papers/supermimic")
MOTION = PAPER / "extracts/g1_random_clip.npz"
CFG = PAPER / "repo/logs/SuperMimic-GenCoord-G1/2026-11-22_15-43-40/runner_cfg.json"
WEIGHTS = PAPER / "repo/logs/SuperMimic-GenCoord-G1/2026-11-21_01-24-36"
OUT_DIR = PAPER / "extracts/replay_buffer_example_frames"
SIDECAR = PAPER / "extracts/replay_buffer_example.json"
MAX_FRAMES = 64


def save_png(path: Path, rgb: np.ndarray) -> None:
    image = np.ascontiguousarray(rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    height, width = image.shape[:2]
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-i", "pipe:0", str(path),
        ],
        input=image.tobytes(),
        check=True,
    )


def main() -> None:
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device.startswith("cuda"):
        print("gpu:", torch.cuda.get_device_name(0))

    env_cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=False)
    env_cfg.scene.num_envs = 1
    env_cfg.commands["motion"].motion_file = str(MOTION)
    env_cfg.auto_reset = False
    env_cfg.viewer.width = 640
    env_cfg.viewer.height = 540
    env_cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
    env_cfg.viewer.distance = 3.0

    raw = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(raw)

    runner_cfg = load_runner_cfg(json.loads(CFG.read_text()))
    wm = build_world_model(runner_cfg, env, device)
    wm.load_bundle(torch.load(WEIGHTS / "world_model.pt", map_location=device, weights_only=False))
    actor = build_actor(runner_cfg, wm, device)
    actor.load_state_dict(torch.load(WEIGHTS / "actor.pt", map_location=device, weights_only=False))
    actor.eval()
    representation = wm.representation

    obs, _ = env.reset()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[dict] = []
    for t in range(MAX_FRAMES):
        rgb = raw.render()
        image_name = f"replay_buffer_example_frames/{t:03d}.png"
        save_png(OUT_DIR / f"{t:03d}.png", rgb)
        full = obs["full_state"]
        with torch.no_grad():
            aux = representation.extract_aux_state(full)
            goal = extract_actor_goal(obs["goal"])
            rep = representation.represent(full)
            sim_ctx = wm.sim_context.read(raw)
            sensor = representation.compute_actor_sensor(rep, sim_ctx)
            history = sensor.unsqueeze(1)
            action = actor(build_actor_obs_td({"motion": goal}, history))
        next_obs, _, dones, _ = env.step(action)
        root = full[0, :3].detach().float().cpu().tolist()
        action_row = action[0].detach().float().cpu().tolist()
        nxt = next_obs["full_state"][0].detach().float().cpu().tolist()
        frames.append(
            {
                "t": t,
                "time_s": round(t * float(raw.step_dt), 4),
                "image": image_name,
                "observation": (
                    f"GenCoord full state before the step, dim {int(full.shape[-1])}, "
                    f"root_pos {[round(v, 4) for v in root]}"
                ),
                "trained_checkpoint_action": action_row,
                "golden": {
                    "kind": "next_state",
                    "summary": f"Logged full state after the step, dim {len(nxt)}.",
                    "values": nxt,
                },
            }
        )
        print(f"frame {t} action_dim {len(action_row)} next_state {len(nxt)} done {bool(dones[0])}")
        obs = next_obs
        if bool(dones[0]):
            break

    SIDECAR.write_text(json.dumps({"fps": int(round(1.0 / float(raw.step_dt))), "frames": frames}) + "\n")
    print(f"wrote {SIDECAR} frames {len(frames)}")
    env.close()


if __name__ == "__main__":
    main()
