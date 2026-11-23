#!/usr/bin/env python3
"""Tools for the paper-card corpus.

Commands:
  fetch-pdf URL DEST
  clone URL DEST
  index REPO DEST_DIR
  validate CARD
  render-example SIDECAR DEST_MP4
  site CORPUS_DIR
"""

from __future__ import annotations

import ast
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schema.json"

STATUS_OK = {"verified", "inferred", "missing"}


def die(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def fetch_pdf(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    if dest.stat().st_size < 1000 or dest.read_bytes()[:4] != b"%PDF":
        die(f"{dest} is not a PDF")
    print(dest)


def clone_repo(url: str, dest: Path) -> str:
    if dest.exists():
        die(f"{dest} already exists")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
    sha = subprocess.check_output(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], text=True
    ).strip()
    print(sha)
    return sha


def _bases(node: ast.ClassDef) -> list[str]:
    return [ast.unparse(base) for base in node.bases]


def _is_module(bases: list[str]) -> bool:
    text = " ".join(bases)
    return "Module" in text or "PreTrainedModel" in text


def _init_assigns(node: ast.ClassDef) -> list[str]:
    lines: list[str] = []
    for item in node.body:
        if not isinstance(item, ast.FunctionDef) or item.name != "__init__":
            continue
        for stmt in item.body:
            if not isinstance(stmt, ast.Assign):
                continue
            target = stmt.targets[0]
            rendered = ast.unparse(target)
            if rendered.startswith("self."):
                lines.append(f"    {rendered} = {ast.unparse(stmt.value)[:160]}")
    return lines[:12]


def index_repo(repo: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    modules: list[dict] = []
    outline: list[str] = [
        "AST outline of classes whose base names contain Module or PreTrainedModel.",
        "This is not a live torch.nn.Module print and does not count parameters.",
        "",
    ]
    for path in sorted(repo.rglob("*.py")):
        if any(part in {".git", "tests", "__pycache__"} for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        rel = path.relative_to(repo).as_posix()
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases = _bases(node)
            if not _is_module(bases):
                continue
            record = {
                "name": node.name,
                "path": rel,
                "line": node.lineno,
                "bases": bases,
                "assigns": _init_assigns(node),
            }
            modules.append(record)
            outline.append(f"{node.name}  {rel}:{node.lineno}")
            if bases:
                outline.append(f"  bases: {', '.join(bases)}")
            outline.extend(record["assigns"])
            outline.append("")
    (dest / "modules.json").write_text(json.dumps(modules, indent=2) + "\n")
    (dest / "model_outline.txt").write_text("\n".join(outline))
    print(f"{len(modules)} module classes -> {dest}")


SLOTS = [
    "observation_encoder",
    "language_or_goal",
    "memory",
    "action_representation",
    "controller",
    "objective",
]
SLOT_LABELS = {
    "observation_encoder": "Observation encoder",
    "language_or_goal": "Language or goal",
    "memory": "Memory",
    "action_representation": "Action representation",
    "controller": "Controller",
    "objective": "Objective",
}


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _check_types(node, schema, path: str, errors: list[str]) -> None:
    """A small checker for this schema: types, required keys, enums, refs."""
    if "$ref" in schema:
        ref = schema["$ref"].split("/")[-1]
        _check_types(node, _load_schema()["$defs"][ref], path, errors)
        return
    if "enum" in schema and node not in schema["enum"]:
        errors.append(f"{path}: {node!r} not in {schema['enum']}")
        return
    expected = schema.get("type")
    if isinstance(expected, list):
        if node is None and "null" in expected:
            return
        expected = next(item for item in expected if item != "null")
    if expected == "boolean":
        if not isinstance(node, bool):
            errors.append(f"{path}: expected boolean")
        return
    py = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
    }.get(expected)
    if py and (not isinstance(node, py) or isinstance(node, bool)):
        errors.append(f"{path}: expected {expected}")
        return
    if isinstance(node, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in node:
                errors.append(f"{path}.{key}: missing")
        if schema.get("additionalProperties") is False:
            allowed = set(schema.get("properties", {}))
            for key in node:
                if key not in allowed:
                    errors.append(f"{path}.{key}: unexpected")
        for key, sub in schema.get("properties", {}).items():
            if key in node:
                _check_types(node[key], sub, f"{path}.{key}", errors)
    if isinstance(node, list):
        item = schema.get("items")
        if item:
            for i, value in enumerate(node):
                _check_types(value, item, f"{path}[{i}]", errors)
    if isinstance(node, str) and "minLength" in schema and len(node) < schema["minLength"]:
        errors.append(f"{path}: empty")
    if isinstance(node, str) and "pattern" in schema and not re.search(schema["pattern"], node):
        errors.append(f"{path}: does not match {schema['pattern']}")


def _evidence_errors(card: dict) -> list[str]:
    errors: list[str] = []

    def check(status: str, sources: list, code: list, path: str) -> None:
        if status == "verified" and not sources and not code:
            errors.append(f"{path}: verified claim has no source or code anchor")
        if status == "missing" and sources:
            errors.append(f"{path}: missing claim still has sources")
        for anchor in code:
            if anchor["end"] < anchor["start"]:
                errors.append(f"{path}: code range {anchor['path']} is reversed")

    provenance = card["data"]["training_provenance"]
    check(provenance["status"], provenance["sources"], [], "data.training_provenance")
    for i, dataset in enumerate(card["data"]["datasets"]):
        check(dataset["status"], dataset["sources"], dataset.get("code", []), f"data[{i}]")
        check(dataset["golden"]["status"], dataset["golden"]["sources"], [], f"data[{i}].golden")
        if "simulator" in dataset:
            sim = dataset["simulator"]
            check(sim["status"], sim["sources"], [], f"data[{i}].simulator")
        if "recorded" in dataset:
            recorded = dataset["recorded"]
            check(recorded["status"], recorded["sources"], [], f"data[{i}].recorded")
        if "example" in dataset:
            example = dataset["example"]
            check(example["status"], example["sources"], [], f"data[{i}].example")
    check(
        card["model"]["total_params"]["status"],
        card["model"]["total_params"]["sources"],
        [],
        "model.total_params",
    )
    for i, module in enumerate(card["model"]["modules"]):
        check(module["status"], module["sources"], module.get("code", []), f"modules[{i}]")
    for i, fact in enumerate(card["compute"]):
        check(fact["status"], fact["sources"], [], f"compute[{i}]")
    for i, obj in enumerate(card["objectives"]):
        check(obj["status"], obj["sources"], obj.get("code", []), f"objectives[{i}]")
    for split in ("train", "infer"):
        for i, step in enumerate(card["procedures"][split]):
            check(
                step["status"],
                step.get("sources", []),
                step.get("code", []),
                f"{split}[{i}]",
            )
    check(card["task"]["status"], card["task"]["sources"], [], "task")
    slots = [part["slot"] for part in card["parts"]]
    if slots != SLOTS:
        errors.append(f"parts: expected slots {SLOTS}, got {slots}")
    for i, part in enumerate(card["parts"]):
        check(part["status"], part["sources"], part.get("code", []), f"parts[{i}]")
        evidence = part["evidence"]
        check(evidence["status"], evidence["sources"], [], f"parts[{i}].evidence")
    return errors


EXAMPLE_SHOWS = ["observation", "trained_checkpoint_action", "golden"]
MAX_EXAMPLE_FRAMES = 128


def _data_errors(card: dict, paper_dir: Path) -> list[str]:
    errors: list[str] = []
    datasets = card["data"]["datasets"]
    kinds = {item["stream_kind"] for item in datasets}
    mode = card["data"]["training_provenance"]["mode"]
    if mode == "sim_policy_rollout" and "simulation" not in kinds:
        errors.append("data.training_provenance: sim_policy_rollout needs a simulation dataset")
    if mode == "recorded_success" and "recorded" not in kinds and "mixed" not in kinds:
        errors.append("data.training_provenance: recorded_success needs a recorded dataset")
    if mode == "both" and "mixed" not in kinds and not ({"simulation", "recorded"} <= kinds):
        errors.append("data.training_provenance: both needs a simulation dataset and a recorded dataset, or one mixed dataset")
    for i, dataset in enumerate(datasets):
        kind = dataset["stream_kind"]
        if kind == "simulation" and "simulator" not in dataset:
            errors.append(f"data[{i}]: simulation dataset needs simulator")
        if kind == "recorded" and "recorded" not in dataset:
            errors.append(f"data[{i}]: recorded dataset needs recorded")
        if kind == "mixed" and "mixture" not in dataset:
            errors.append(f"data[{i}]: mixed dataset needs mixture")
        if dataset["feeds_gradient"] and "example" not in dataset:
            errors.append(f"data[{i}]: a dataset that feeds a loss needs example")
        if "example" in dataset:
            errors.extend(_example_errors(dataset["example"], paper_dir, f"data[{i}].example"))
    return errors


def _example_errors(example: dict, paper_dir: Path, path: str) -> list[str]:
    errors: list[str] = []
    if example["shows"] != EXAMPLE_SHOWS:
        errors.append(f"{path}.shows: expected {EXAMPLE_SHOWS}")
    checkpoint = example["trained_checkpoint"]
    if example["status"] == "inferred":
        errors.append(f"{path}: a checkpoint clip is verified or missing")
        return errors
    if example["status"] == "missing":
        if example["view"] != "none" or example["fps"] is not None or example["num_frames"] is not None:
            errors.append(f"{path}: a missing clip has view none and null fps and num_frames")
        if example["video_path"] is not None or example["sidecar_path"] is not None:
            errors.append(f"{path}: a missing clip has null video_path and sidecar_path")
        if checkpoint["checkpoint_path"] is not None:
            errors.append(f"{path}: a missing clip has null checkpoint_path")
        return errors
    if example["view"] == "none":
        errors.append(f"{path}: a verified clip names sim_rgb, camera, or plot")
    fps = example["fps"]
    count = example["num_frames"]
    if not isinstance(fps, int) or isinstance(fps, bool) or fps < 1:
        errors.append(f"{path}.fps: expected a positive integer")
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= MAX_EXAMPLE_FRAMES:
        errors.append(f"{path}.num_frames: expected 1..{MAX_EXAMPLE_FRAMES}")
    dim = checkpoint["dim"]
    if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
        errors.append(f"{path}.trained_checkpoint.dim: expected a positive integer")
    video = _extracts_file(example["video_path"], paper_dir, f"{path}.video_path", errors)
    sidecar_path = _extracts_file(example["sidecar_path"], paper_dir, f"{path}.sidecar_path", errors)
    weights = _paper_file(checkpoint["checkpoint_path"], paper_dir, f"{path}.trained_checkpoint.checkpoint_path", errors)
    if sidecar_path is None or video is None or weights is None or errors:
        return errors
    errors.extend(_sidecar_errors(sidecar_path, fps, count, dim, example["view"]))
    probed = _video_frame_count(video)
    if probed != count:
        errors.append(f"{path}: video has {probed} frames, sidecar declares {count}")
    return errors


def _extracts_file(value: str | None, paper_dir: Path, path: str, errors: list[str]) -> Path | None:
    if not isinstance(value, str) or not value.startswith("extracts/") or ".." in Path(value).parts:
        errors.append(f"{path}: expected a path under extracts/")
        return None
    file = paper_dir / value
    if not file.is_file():
        errors.append(f"{path}: {value} is missing")
        return None
    return file


def _paper_file(value: str | None, paper_dir: Path, path: str, errors: list[str]) -> Path | None:
    if not isinstance(value, str) or not value or ".." in Path(value).parts:
        errors.append(f"{path}: expected a path inside the paper directory")
        return None
    file = paper_dir / value
    if not file.is_file():
        errors.append(f"{path}: {value} is missing")
        return None
    return file


def _sidecar_errors(path: Path, fps: int, count: int, dim: int, view: str) -> list[str]:
    errors: list[str] = []
    try:
        sidecar = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{path.name}: {exc}"]
    if sidecar.get("fps") != fps:
        errors.append(f"{path.name}: fps {sidecar.get('fps')} != example fps {fps}")
    frames = sidecar.get("frames")
    if not isinstance(frames, list) or len(frames) != count:
        errors.append(f"{path.name}: frames length must be {count}")
        return errors
    for i, frame in enumerate(frames):
        if not isinstance(frame, dict) or frame.get("t") != i:
            errors.append(f"{path.name}[{i}]: t must equal the frame index")
            continue
        action = frame.get("trained_checkpoint_action")
        if not isinstance(action, list) or len(action) != dim or not all(_is_number(item) for item in action):
            errors.append(f"{path.name}[{i}]: trained_checkpoint_action must be {dim} numbers")
        golden = frame.get("golden")
        if not isinstance(golden, dict) or not isinstance(golden.get("kind"), str) or not isinstance(golden.get("summary"), str):
            errors.append(f"{path.name}[{i}]: golden needs kind and summary")
        elif "values" in golden and (
            not isinstance(golden["values"], list) or not all(_is_number(item) for item in golden["values"])
        ):
            errors.append(f"{path.name}[{i}]: golden.values must be numbers")
        image = frame.get("image")
        if view in {"sim_rgb", "camera"}:
            if not isinstance(image, str) or not (path.parent / image).is_file():
                errors.append(f"{path.name}[{i}]: image file is missing")
        elif image is not None and not (path.parent / image).is_file():
            errors.append(f"{path.name}[{i}]: image file is missing")
    return errors


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _video_frame_count(path: Path) -> int | None:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        return None
    try:
        return int(probe.stdout.strip())
    except ValueError:
        return None


def _write_ppm(path: Path, width: int, height: int, rgb: bytearray) -> None:
    header = f"P6\n{width} {height}\n255\n".encode()
    path.write_bytes(header + rgb)


def _load_rgb(image: Path, width: int, height: int) -> bytes:
    probe = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(image), "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "pipe:1",
        ],
        capture_output=True, check=True,
    )
    return probe.stdout


def _fill(rgb: bytearray, width: int, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    for y in range(y0, y1):
        row = (y * width + x0) * 3
        for x in range(x0, x1):
            rgb[row:row + 3] = bytes(color)
            row += 3


def _bars(rgb: bytearray, width: int, height: int, left: int, series: list[float], color: tuple[int, int, int], y0: int, y1: int) -> None:
    if not series:
        return
    peak = max(1.0, max(abs(value) for value in series))
    slot = (width - left - 16) / len(series)
    mid = (y0 + y1) // 2
    for i, value in enumerate(series):
        x = left + 8 + int(i * slot)
        span = max(1, int((y1 - y0) * 0.45 * (value / peak)))
        y_top, y_bot = (mid - span, mid) if value >= 0 else (mid, mid + span)
        _fill(rgb, width, x, max(y0, y_top), min(width, x + max(1, int(slot) - 2)), min(y1, y_bot), color)


def render_example(sidecar_path: Path, dest: Path) -> None:
    """Compose one mp4 whose frame i is sidecar frame i.

    The left pane is the observation image, or a flat field when the sidecar
    has no image. The right pane draws the trained checkpoint action in green
    and golden.values in amber when that list is present.
    """
    sidecar = json.loads(sidecar_path.read_text())
    frames = sidecar["frames"]
    fps = int(sidecar["fps"])
    width, height, left = 960, 540, 640
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".frames")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    for i, frame in enumerate(frames):
        rgb = bytearray([28, 26, 23]) * (width * height)
        image = frame.get("image")
        if image:
            raw = _load_rgb(sidecar_path.parent / image, left, height)
            for y in range(height):
                src = y * left * 3
                dst = y * width * 3
                rgb[dst:dst + left * 3] = raw[src:src + left * 3]
        else:
            _fill(rgb, width, 0, 0, left, height, (42, 48, 56))
        _fill(rgb, width, left, int(height * i / max(1, len(frames))), width, int(height * (i + 1) / max(1, len(frames))), (18, 72, 64))
        action = frame.get("trained_checkpoint_action") or []
        golden_values = (frame.get("golden") or {}).get("values") or []
        _bars(rgb, width, height, left, action, (125, 206, 160), 24, height // 2 - 8)
        _bars(rgb, width, height, left, golden_values, (224, 177, 90), height // 2 + 8, height - 24)
        ppm = tmp / f"{i:04d}.ppm"
        _write_ppm(ppm, width, height, rgb)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
            "-i", str(tmp / "%04d.ppm"), "-pix_fmt", "yuv420p", str(dest),
        ],
        check=True,
    )
    shutil.rmtree(tmp)
    print(dest)


def validate_card(path: Path) -> None:
    card = json.loads(path.read_text())
    errors: list[str] = []
    _check_types(card, _load_schema(), "$", errors)
    if not errors:
        errors.extend(_evidence_errors(card))
        errors.extend(_data_errors(card, path.parent))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        die(f"{path} failed with {len(errors)} errors")
    print(f"ok {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebuild_index(corpus: Path) -> list[dict]:
    rows = []
    for card_path in sorted(corpus.glob("papers/*/card.json")):
        card = json.loads(card_path.read_text())
        pdf = card_path.parent / "paper.pdf"
        rows.append(
            {
                "id": card["id"],
                "title": card["identity"]["title"],
                "repo_role": card["identity"]["repo_role"],
                "repo_url": card["identity"].get("repo_url"),
                "total_params": card["model"]["total_params"]["value"],
                "paradigm": card["landscape"]["paradigm"],
                "hardware_fit": card["landscape"]["hardware_fit"],
                "pdf_sha256": _sha256(pdf) if pdf.exists() else None,
                "gaps": len(card["gaps"]),
                "parts": {part["slot"]: part["choice"] for part in card["parts"]},
            }
        )
    (corpus / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


CSS = """
:root {
  color-scheme: light dark;
  --bg: #f4f1ea;
  --ink: #1c1915;
  --muted: #5e584f;
  --line: #ddd6c8;
  --card: #fffdf8;
  --accent: #0f6e56;
  --warn: #8a4b08;
  --bad: #8d2f2f;
  --code: #efeae0;
  --link: #0b5cab;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #161410;
    --ink: #f3efe6;
    --muted: #b7b0a4;
    --line: #3a342c;
    --card: #221e19;
    --accent: #7dcea0;
    --warn: #e0b15a;
    --bad: #f0a0a0;
    --code: #2c271f;
    --link: #9dc7ff;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 16px/1.5 "Iowan Old Style", Palatino, Georgia, serif;
}
a { color: var(--link); }
header, main { width: min(1080px, calc(100% - 32px)); margin: 0 auto; }
main.wide { width: min(1440px, calc(100% - 24px)); }
.matrix { overflow-x: auto; }
.matrix td, .matrix th { min-width: 170px; }
header { padding: 28px 0 8px; }
h1 { font-size: 1.8rem; line-height: 1.2; margin: 0 0 8px; }
h2 { font-size: 1.25rem; margin: 28px 0 8px; }
.lede, .meta { color: var(--muted); }
.nav { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 20px; }
.nav a, .pill {
  display: inline-block;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 2px 10px;
  background: var(--card);
  text-decoration: none;
  font: 13px/1.6 ui-sans-serif, system-ui, sans-serif;
}
.pill.verified { color: var(--accent); }
.pill.inferred { color: var(--warn); }
.pill.missing, .pill.none { color: var(--bad); }
.section, a.paper-row {
  display: block;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 14px 16px;
  margin: 12px 0;
  color: inherit;
  text-decoration: none;
}
a.paper-row strong { color: var(--link); }
table { width: 100%; border-collapse: collapse; font: 14px/1.4 ui-sans-serif, system-ui, sans-serif; }
th, td { text-align: left; vertical-align: top; padding: 6px 8px; border-bottom: 1px solid var(--line); }
code, pre { font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
video { width: 100%; max-width: 960px; background: #000; }
pre {
  background: var(--code);
  padding: 12px;
  overflow: auto;
  max-height: 420px;
  border-radius: 8px;
}
.formula { overflow-x: auto; padding: 8px 0; }
.step { margin: 0 0 10px; }
.sources { color: var(--muted); font: 13px/1.4 ui-sans-serif, system-ui, sans-serif; }
ul.gaps { margin-top: 0; }
"""


def _pill(status: str) -> str:
    return f'<span class="pill {html.escape(status)}">{html.escape(status)}</span>'


def _sources(sources: list[dict], code: list[dict] | None = None) -> str:
    parts = []
    for source in sources or []:
        label = f"{source['kind']}: {source['locator']}"
        if source.get("url"):
            parts.append(f'<a href="{html.escape(source["url"])}">{html.escape(label)}</a>')
        else:
            parts.append(html.escape(label))
    for anchor in code or []:
        label = f"{anchor['path']}:{anchor['start']}-{anchor['end']}"
        if anchor.get("symbol"):
            label = f"{anchor['symbol']} · {label}"
        parts.append(f'<a href="{html.escape(anchor["url"])}">{html.escape(label)}</a>')
    if not parts:
        return ""
    return '<div class="sources">' + " · ".join(parts) + "</div>"


def _parts_section(card: dict) -> str:
    task = card["task"]
    contract = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{html.escape(task[key])}</td></tr>"
        for label, key in (
            ("Embodiment", "embodiment"),
            ("Action", "action"),
            ("Horizon", "horizon"),
            ("Contact", "contact"),
            ("Latency", "latency"),
            ("Data", "data"),
        )
    )
    blocks = []
    for part in card["parts"]:
        evidence = part["evidence"]
        blocks.append(
            f"""<h3>{html.escape(SLOT_LABELS[part["slot"]])} {_pill(part["status"])}</h3>
<p><strong>{html.escape(part["choice"])}</strong></p>
<p>In: {html.escape(part["inputs"])}</p>
<p>Out: {html.escape(part["outputs"])}</p>
<p>Frozen: {html.escape(part["frozen"])} Trained: {html.escape(part["trained"])}</p>
<p>Breaks if: {html.escape(part["breaks_if"])}</p>
<p>Evidence: {html.escape(evidence["claim"])} {_pill(evidence["status"])} {_sources(evidence["sources"])}</p>
{_sources(part["sources"], part.get("code"))}"""
        )
    return f"""<section class="section"><h2>Task contract</h2>
<p class="meta">What a new task has to supply before a slot from this method can move.</p>
<table><tbody>{contract}</tbody></table>
{_pill(task["status"])} {_sources(task["sources"])}
<h2>Parts</h2>
{''.join(blocks)}
</section>"""


def _compare_page(cards: list[dict]) -> str:
    head = "".join(
        f'<th><a href="papers/{html.escape(card["id"])}.html">{html.escape(card["id"])}</a></th>'
        for card in cards
    )
    task_rows = []
    for label, key in (
        ("Embodiment", "embodiment"),
        ("Action", "action"),
        ("Horizon", "horizon"),
        ("Contact", "contact"),
        ("Latency", "latency"),
        ("Data", "data"),
    ):
        cells = "".join(f"<td>{html.escape(card['task'][key])}</td>" for card in cards)
        task_rows.append(f"<tr><th>{label}</th>{cells}</tr>")
    part_rows = []
    for slot in SLOTS:
        cells = []
        for card in cards:
            part = next(item for item in card["parts"] if item["slot"] == slot)
            cells.append(
                "<td><strong>{}</strong><br><span class='meta'>{}</span><br>{} {}</td>".format(
                    html.escape(part["choice"]),
                    html.escape("Breaks if: " + part["breaks_if"]),
                    html.escape(part["evidence"]["claim"]),
                    _pill(part["evidence"]["status"]),
                )
            )
        part_rows.append(f"<tr><th>{html.escape(SLOT_LABELS[slot])}</th>{''.join(cells)}</tr>")
    body = f"""<header>
<p><a href="index.html">All papers</a></p>
<h1>Parts</h1>
<p class="lede">Match a new task to a row in the contract. Read the break condition before moving a slot. A verified evidence pill means the cited source compares that slot. A missing pill means the published number is for the whole system, or the comparison was not extracted.</p>
</header>
<main class="wide">
<section class="section matrix"><h2>Task contract</h2>
<table><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(task_rows)}</tbody></table></section>
<section class="section matrix"><h2>Slots</h2>
<table><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(part_rows)}</tbody></table></section>
</main>"""
    return _page("Parts", body, depth=".")


def _page(title: str, body: str, depth: str) -> str:
    katex_css = "https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css"
    katex_js = "https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.js"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="{katex_css}">
<style>{CSS}</style>
</head>
<body>
{body}
<script src="{katex_js}"></script>
<script>
document.querySelectorAll(".formula").forEach((el) => {{
  if (window.katex) {{
    katex.render(el.dataset.latex, el, {{displayMode: true, throwOnError: false}});
  }}
}});
</script>
</body>
</html>
"""


def _render_card(card: dict, paper_dir: Path) -> str:
    ident = card["identity"]
    links = []
    if ident.get("pdf_url"):
        links.append(f'<a href="{html.escape(ident["pdf_url"])}">PDF</a>')
    if ident.get("arxiv"):
        links.append(f'<a href="https://arxiv.org/abs/{html.escape(ident["arxiv"])}">arXiv {html.escape(ident["arxiv"])}</a>')
    if ident.get("project_url"):
        links.append(f'<a href="{html.escape(ident["project_url"])}">project</a>')
    if ident.get("repo_url"):
        commit = ident.get("repo_commit") or ""
        label = f'repo {commit[:8]}' if commit else "repo"
        links.append(f'<a href="{html.escape(ident["repo_url"])}">{html.escape(label)}</a>')
    conflicts = ""
    if ident.get("distinct_from"):
        items = "".join(
            f'<li><a href="{html.escape(item["url"])}">{html.escape(item["name"])}</a> — {html.escape(item["why"])}</li>'
            for item in ident["distinct_from"]
        )
        conflicts = f"<h2>Not the same work</h2><ul>{items}</ul>"
    data_blocks = []
    for dataset in card["data"]["datasets"]:
        rows = "".join(
            "<tr><td><code>{}</code></td><td>{}</td><td><code>{}</code></td><td>{}</td></tr>".format(
                html.escape(field["name"]),
                html.escape(field["dtype"]),
                html.escape(field["shape"]),
                html.escape(field.get("note", "")),
            )
            for field in dataset["fields"]
        )
        table = (
            "<table><thead><tr><th>Field</th><th>Type</th><th>Shape</th><th>Note</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            if rows
            else "<p class='meta'>No field-level schema was recorded.</p>"
        )
        example = dataset.get("example")
        example_html = ""
        if example:
            clip = example["trained_checkpoint"]
            video = example.get("video_path")
            player = ""
            if video:
                player = (
                    f'<video controls src="../../papers/{html.escape(card["id"])}/'
                    f'{html.escape(video)}"></video>'
                )
            example_html = (
                f"<p><strong>Example</strong> {_pill(example['status'])} "
                f"{html.escape(example['view'])}. "
                f"Checkpoint action <code>{html.escape(clip['action_name'])}</code> "
                f"dim {html.escape(str(clip['dim']))}. {html.escape(clip['note'])}</p>"
                f"{player}"
            )
        origin = ""
        if dataset.get("simulator"):
            sim = dataset["simulator"]
            origin = f"<p>Simulator {html.escape(sim['name'])}, env <code>{html.escape(sim['env_id'])}</code>, collector {html.escape(sim['collector'])}.</p>"
        if dataset.get("recorded"):
            rec = dataset["recorded"]
            origin = f"<p>Recorded {html.escape(rec['origin'])}: {html.escape(rec['dataset_name'])}. {html.escape(rec['success_filter'])}</p>"
        if dataset.get("mixture"):
            origin += f"<p>{html.escape(dataset['mixture'])}</p>"
        golden = dataset["golden"]
        data_blocks.append(
            f"""<h3>{html.escape(dataset["name"])} {_pill(dataset["tier"])} {_pill(dataset["status"])} {_pill(dataset["stream_kind"])}</h3>
<p>{html.escape(dataset["volume"])}</p>
<p>{html.escape(dataset.get("sampling") or "")}</p>
<p>{html.escape(dataset.get("augmentation") or "")}</p>
{origin}
<p>Feeds gradient: {str(dataset["feeds_gradient"]).lower()}. Golden {html.escape(golden["kind"])}: {html.escape(golden["description"])}</p>
{example_html}
{table}
{_sources(dataset["sources"], dataset.get("code"))}"""
        )
    modules = "".join(
        "<tr><td>{}</td><td>{}</td><td>{} {}</td></tr>".format(
            html.escape(module["name"]),
            html.escape(module["params"] or ""),
            html.escape(module["detail"]),
            _sources(module["sources"], module.get("code")),
        )
        for module in card["model"]["modules"]
    )
    outline_path = paper_dir / "extracts" / "model_outline.txt"
    outline = ""
    if outline_path.exists():
        text = outline_path.read_text()
        outline = (
            "<h3>AST module outline</h3>"
            "<p class='meta'>Class and constructor assignments parsed from the clone. "
            "Parameter totals come only from the fact above.</p>"
            f"<pre>{html.escape(text[:20000])}</pre>"
        )
    compute = "".join(
        "<tr><td>{}</td><td>{}</td><td>{} {}</td></tr>".format(
            html.escape(fact["name"]),
            html.escape(fact["value"]),
            _pill(fact["status"]),
            _sources(fact["sources"]),
        )
        for fact in card["compute"]
    )
    objectives = []
    for obj in card["objectives"]:
        symbols = "".join(
            f"<li><code>{html.escape(sym['sym'])}</code> — {html.escape(sym['meaning'])}</li>"
            for sym in obj["symbols"]
        )
        objectives.append(
            f"""<h3>{html.escape(obj["name"])} {_pill(obj["status"])}</h3>
<div class="formula" data-latex="{html.escape(obj["latex"], quote=True)}">{html.escape(obj["latex"])}</div>
<p>{html.escape(obj["explanation"])}</p>
<ul>{symbols}</ul>
{_sources(obj["sources"], obj.get("code"))}"""
        )
    def steps(split: str) -> str:
        items = []
        for i, step in enumerate(card["procedures"][split], 1):
            items.append(
                f'<div class="step"><strong>{i}.</strong> {html.escape(step["text"])} {_pill(step["status"])}'
                f'{_sources(step.get("sources", []), step.get("code"))}</div>'
            )
        return "\n".join(items)
    gaps = "".join(f"<li>{html.escape(gap)}</li>" for gap in card["gaps"])
    authors = ", ".join(html.escape(name) for name in ident["authors"])
    return f"""<header>
<p><a href="../index.html">All papers</a></p>
<h1>{html.escape(ident["title"])}</h1>
<p class="meta">{authors} · {html.escape(ident["venue"])} {ident["year"]}</p>
<p class="nav">{' '.join(links)} {_pill(ident["repo_role"])}</p>
<p>{html.escape(ident.get("repo_note") or "")}</p>
</header>
<main>
<section class="section"><p>{html.escape(card["summary"])}</p>
<p class="meta">{html.escape(card["landscape"]["paradigm"])} · {html.escape(card["landscape"]["bottleneck"])} · {html.escape(card["landscape"]["hardware_fit"])}</p>
{conflicts}
</section>
{_parts_section(card)}
<section class="section"><h2>Data</h2>
<p class="meta">{html.escape(card["data"]["note"])}</p>
<p>{_pill(card["data"]["training_provenance"]["mode"])} {html.escape(card["data"]["training_provenance"]["summary"])}</p>
{''.join(data_blocks)}
</section>
<section class="section"><h2>Model</h2>
<p>{html.escape(card["model"]["overview"])}</p>
<p><strong>{html.escape(card["model"]["total_params"]["value"])}</strong> {_pill(card["model"]["total_params"]["status"])} {_sources(card["model"]["total_params"]["sources"])}</p>
<p class="meta">{html.escape(card["model"]["runtime"])}</p>
<table><thead><tr><th>Module</th><th>Params</th><th>Detail</th></tr></thead><tbody>{modules}</tbody></table>
{outline}
</section>
<section class="section"><h2>Compute</h2>
<table><tbody>{compute}</tbody></table>
</section>
<section class="section"><h2>Objectives</h2>
{''.join(objectives)}
</section>
<section class="section"><h2>Train</h2>{steps("train")}</section>
<section class="section"><h2>Infer</h2>{steps("infer")}</section>
<section class="section"><h2>Gaps</h2><ul class="gaps">{gaps}</ul></section>
</main>"""


def build_site(corpus: Path) -> None:
    rows = rebuild_index(corpus)
    site = corpus / "site"
    if site.exists():
        shutil.rmtree(site)
    papers = site / "papers"
    papers.mkdir(parents=True)
    loaded = []
    listing = []
    for row in rows:
        card_path = corpus / "papers" / row["id"] / "card.json"
        card = json.loads(card_path.read_text())
        page = papers / f"{row['id']}.html"
        page.write_text(_page(card["identity"]["title"], _render_card(card, card_path.parent), depth=".."))
        loaded.append(card)
        listing.append(
            f"""<a class="paper-row" href="papers/{html.escape(row["id"])}.html">
<strong>{html.escape(row["title"])}</strong><br>
<span class="meta">{html.escape(row["paradigm"])}</span><br>
{_pill(row["repo_role"])} <span class="meta">{html.escape(row["total_params"])} · {row["gaps"]} gaps · {html.escape(row["hardware_fit"])}</span>
</a>"""
        )
    index = f"""<header><h1>Paper cards</h1>
<p class="lede">Structured reading notes for the CoRL 2026 frontier list. A number is on the page only when its card names a source. Cloned code is linked at the pinned commit.</p>
<p class="nav"><a href="parts.html">Parts comparison</a></p>
</header>
<main>{''.join(listing)}</main>"""
    (site / "index.html").write_text(_page("Paper cards", index, depth="."))
    order = ["microvla", "compvla", "robottt", "supermimic", "pi07", "dreamzero"]
    loaded.sort(key=lambda card: order.index(card["id"]) if card["id"] in order else len(order))
    (site / "parts.html").write_text(_compare_page(loaded))
    print(site / "index.html")


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        die(__doc__ or "missing command")
    command, rest = argv[1], argv[2:]
    if command == "fetch-pdf" and len(rest) == 2:
        fetch_pdf(rest[0], Path(rest[1]))
    elif command == "clone" and len(rest) == 2:
        clone_repo(rest[0], Path(rest[1]))
    elif command == "index" and len(rest) == 2:
        index_repo(Path(rest[0]), Path(rest[1]))
    elif command == "validate" and len(rest) == 1:
        validate_card(Path(rest[0]))
    elif command == "render-example" and len(rest) == 2:
        render_example(Path(rest[0]), Path(rest[1]))
    elif command == "site" and len(rest) == 1:
        build_site(Path(rest[0]))
    else:
        die(__doc__ or "bad command")


if __name__ == "__main__":
    main(sys.argv)
