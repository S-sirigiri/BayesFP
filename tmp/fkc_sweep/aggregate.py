"""Aggregate sweep results into one Markdown table sorted by (collision%, -success).

Discovers runs by scanning src/RoboLab/output/fkc_sweep_* and joining each with
its corresponding YAML config in tmp/fkc_sweep/{,sweep_configs/}.
"""
from __future__ import annotations

import json
import os
import sys
import yaml
import numpy as np
from pathlib import Path

ROOT = Path("/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion")
OUT = ROOT / "src" / "RoboLab" / "output"
SWEEP = ROOT / "tmp" / "fkc_sweep"
PLACEHOLDER = ROOT / "src" / "openpi" / "configs" / "fkc" / "placeholder.yaml"


def _find_cfg(name: str) -> dict | None:
    for cand in [SWEEP / f"cfg_{name}.yaml", SWEEP / "sweep_configs" / f"cfg_{name}.yaml"]:
        if cand.exists():
            return yaml.safe_load(cand.read_text())
    return None


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _collision_stats(npz_path: Path) -> tuple[int, int, float]:
    """(steps_in_collision, steps_total, min_clearance_m). NaN-safe."""
    if not npz_path.exists():
        return 0, 0, float("nan")
    with np.load(npz_path, allow_pickle=True) as f:
        in_coll = f["in_collision"]
        min_clr = f["min_clearance_m"]
    return int(in_coll.sum()), int(in_coll.shape[0]), float(min_clr.min())


def _row(run_dir: Path) -> dict | None:
    name = run_dir.name.replace("fkc_sweep_", "")
    res = _load_jsonl(run_dir / "episode_results.jsonl")
    if not res:
        return None
    n = len(res)
    succ = sum(r["success"] for r in res)
    # Collision npz lives under <task>/collisions_run_*.npz
    npz_paths = list(run_dir.glob("**/collisions_run_*.npz"))
    if npz_paths:
        cs, ct, mc = _collision_stats(npz_paths[0])
    else:
        cs, ct, mc = 0, 0, float("nan")
    cfg = _find_cfg(name) or {}
    coll = (cfg.get("collision") or {})
    return {
        "name": name,
        "n": n,
        "success": succ,
        "succ_rate": succ / n if n else 0.0,
        "coll_steps": cs,
        "total_steps": ct,
        "coll_pct": (cs / ct * 100) if ct else 0.0,
        "min_clear_mm": mc * 1000,
        "mode": cfg.get("mode"),
        "penalty": coll.get("penalty_type"),
        "margin": coll.get("safety_margin"),
        "w_grad": cfg.get("w_ineq_grad"),
        "w_value": cfg.get("w_ineq_value"),
        "beta": cfg.get("beta_strength"),
    }


def main():
    rows = []
    for d in sorted(OUT.glob("fkc_sweep_*")):
        if d.is_dir():
            r = _row(d)
            if r:
                rows.append(r)
    if not rows:
        print("no rows found", file=sys.stderr)
        return 1
    # Identify a vanilla baseline (max-n vanilla)
    vanilla = [r for r in rows if r["mode"] in (None, "vanilla")]
    vanilla_succ = max((v["succ_rate"] for v in vanilla), default=None)
    vanilla_coll = min((v["coll_pct"] for v in vanilla), default=None)

    rows_sorted = sorted(rows, key=lambda r: (r["coll_pct"], -r["succ_rate"]))
    cols = ["name", "n", "succ", "coll%", "min_mm", "penalty", "margin", "beta"]
    widths = {c: max(len(c), 6) for c in cols}
    body = []
    for r in rows_sorted:
        cells = {
            "name": r["name"],
            "n": str(r["n"]),
            "succ": f"{r['success']}/{r['n']}",
            "coll%": f"{r['coll_pct']:.1f}",
            "min_mm": f"{r['min_clear_mm']:+.1f}",
            "penalty": r["penalty"] or "-",
            "margin": (f"{float(r['margin']):.1e}" if r["margin"] is not None else "-"),
            "beta": (f"{float(r['beta']):.0e}" if r["beta"] is not None else "-"),
        }
        for c in cols:
            widths[c] = max(widths[c], len(cells[c]))
        body.append(cells)
    # render
    fmt = "  ".join("{:<" + str(widths[c]) + "}" for c in cols)
    print(fmt.format(*cols))
    print("  ".join("-" * widths[c] for c in cols))
    for cells in body:
        print(fmt.format(*[cells[c] for c in cols]))
    if vanilla_succ is not None:
        print()
        print(f"vanilla baseline: success={vanilla_succ*100:.1f}%, collision={vanilla_coll:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
