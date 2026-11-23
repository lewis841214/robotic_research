"""Run a local MicroVLA checkpoint on consecutive LIBERO-Spatial frames.

Each frame stores the agent camera, the unnormalized 16x7 action chunk from
MicroVLA.sample, and the logged future actions from the cache.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/mnt/d/robotic_research/corpus/papers/microvla/repo")
sys.path.insert(0, str(REPO))

from train import Batches, MicroVLA, Config, unnormalize  # noqa: E402

PAPER = REPO.parent
CACHE = REPO / "runs/libero_cache_96"
CKPT = REPO / "runs/paper-card/latest.pt"
OUT = PAPER / "extracts/microvla_example_frames"
SIDECAR = PAPER / "extracts/microvla_example.json"
N = 16


def save_png(path: Path, rgb: np.ndarray) -> None:
    image = np.ascontiguousarray(rgb)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    if device.type == "cuda":
        print("gpu", torch.cuda.get_device_name(0))
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    data = Batches(CACHE, horizon=ckpt["config"]["horizon"], seed=0)
    cfg = Config(**ckpt["config"])
    model = MicroVLA(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    episode = int(data.arrays["episode"][data.indices["train"][0]])
    positions = np.flatnonzero(data.arrays["episode"] == episode)
    positions = positions[:N]
    if len(positions) < 2:
        raise SystemExit(f"episode {episode} has {len(positions)} frames")
    frames = []
    OUT.mkdir(parents=True, exist_ok=True)
    ends = data.ends
    for t, index in enumerate(positions):
        image = data.arrays["images"][index, 0].transpose(1, 2, 0)
        save_png(OUT / f"{t:03d}.png", image)
        future = np.minimum(index + np.arange(data.horizon), ends[index])
        images_t = torch.from_numpy(np.ascontiguousarray(data.arrays["images"][index][None])).to(device)
        state_t = torch.from_numpy(np.ascontiguousarray(data.arrays["state"][index][None])).to(device)
        state_t = (state_t - state_t.new_tensor(data.meta["norm"]["state"]["center"])) / state_t.new_tensor(data.meta["norm"]["state"]["scale"])
        instr = np.ascontiguousarray(data.instructions[int(data.arrays["task"][index])][None])
        instr_t = torch.from_numpy(instr).to(device)
        golden = data.arrays["actions"][future]
        with torch.no_grad():
            pred = model.sample(images_t, state_t, instr_t, num_steps=10)
        pred = unnormalize(pred, data.meta["norm"]["action"])[0].detach().float().cpu().reshape(-1).tolist()
        gold = golden.reshape(-1).astype(float).tolist()
        task = data.meta["identity"]["tasks"][int(data.arrays["task"][index])]
        frames.append({
            "t": t,
            "time_s": round(t / 20, 4),
            "image": f"microvla_example_frames/{t:03d}.png",
            "observation": f"LIBERO-Spatial agent camera, episode {episode}, frame {int(data.arrays['frame'][index])}, task {task}",
            "trained_checkpoint_action": pred,
            "golden": {
                "kind": "action_chunk",
                "summary": "Logged 16 by 7 future actions, flattened row-major, in the dataset action units.",
                "values": gold,
            },
        })
        print(f"frame {t} pred {len(pred)} golden {len(gold)}")
    SIDECAR.write_text(json.dumps({"fps": 20, "frames": frames}) + "\n")
    print("wrote", SIDECAR, "n", len(frames))


if __name__ == "__main__":
    main()
