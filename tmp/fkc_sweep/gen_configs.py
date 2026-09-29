"""Generate sweep YAMLs for the 5 margins x 4 penalty types matrix.

Beta picked per cell so that half_sigma_sq * beta * |grad_J| ~ |v_t| ~ 1
(half_sigma_sq peak ~0.008 with sigma_scale=0.25/zero_ends).
- linear/softplus: |grad_J| per active point ~1; small beta needed.
- squared_hinge/squared_distance_hinge: |grad_J| per active point ~ margin;
  larger beta needed (more so for smaller margins).
- safety_margin=0 + squared_distance_hinge is skipped (identically zero).
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent
TEMPLATE = """# Auto-generated sweep config: penalty={penalty} margin={margin} beta={beta:.0e}
mode: fkc

num_steps: null
num_particles: 4
particles_parallel: true
parallel_nn_and_cost: true

w_cost_value: 0.0
w_eq_value:   0.0
w_ineq_value: 10.0
w_cost_grad:  0.0
w_eq_grad:    0.0
w_ineq_grad:  1.0

beta_schedule: constant
beta_strength: {beta:.6g}

sigma_schedule: zero_ends
sigma_scale: 0.25

resample_interval: 1
resample_t_min: 0.05
resample_t_max: 0.95
center_log_weight_increment: true

fk:
  robot: franka_panda
  base_xyz: [0.0, 0.0, 0.0]
  base_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
  ee_offset_xyz: [0.0, 0.0, 0.01817]
  ee_offset_quat_xyzw: [0.0, 0.0, 0.0, 1.0]

collision:
  mode: ee_plus_arm
  penalty_type: {penalty}
  safety_margin: {margin}
  arm_sample_points: 8
  full_body_points: 30
  softplus_beta: 50.0

dynamics:
  enabled: true
  stiffness: 400.0
  damping: 80.0
  sim_dt: 0.008333333333333333
  decimation: 8

cost:
  target_xyz: [0.55, 0.0, 0.25]
"""

MARGINS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]

# Beta choice per (penalty, margin) cell. Suggested values: 5e2, 5e3, 7e3, 5e4, 5e5.
BETA = {
    "linear": {0.0: 5e3, 1e-4: 5e3, 1e-3: 5e2, 1e-2: 5e2, 1e-1: 5e2},
    "softplus": {0.0: 5e3, 1e-4: 5e3, 1e-3: 5e3, 1e-2: 7e3, 1e-1: 5e2},
    "squared_hinge": {0.0: 5e5, 1e-4: 5e5, 1e-3: 5e4, 1e-2: 7e3, 1e-1: 5e3},
    "squared_distance_hinge": {1e-4: 5e5, 1e-3: 5e4, 1e-2: 7e3, 1e-1: 5e2},
}


def margin_tag(m):
    if m == 0.0:
        return "m0"
    return "m" + ("%g" % m).replace(".", "p").replace("-", "n")


def main():
    sweep_dir = OUT / "sweep_configs"
    sweep_dir.mkdir(exist_ok=True)
    rows = []
    for penalty, beta_by_m in BETA.items():
        for m in MARGINS:
            if m not in beta_by_m:
                continue
            beta = beta_by_m[m]
            name = f"{penalty}_{margin_tag(m)}_b{int(beta):d}"
            path = sweep_dir / f"cfg_{name}.yaml"
            path.write_text(TEMPLATE.format(penalty=penalty, margin=m, beta=beta))
            rows.append((name, str(path)))
    print(f"wrote {len(rows)} configs to {sweep_dir}")
    # Emit a chain-driver bash file
    chain = OUT / "chain_full_sweep.sh"
    lines = ["#!/bin/bash", "set -u", "ROOT=/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion"]
    lines.append('echo "[chain] starting full sweep at $(date)"')
    # Vanilla baseline first (with new OOI list)
    lines.append('bash "$ROOT/tmp/fkc_sweep/run_one.sh" vanilla_oo2 "$ROOT/tmp/fkc_sweep/cfg_vanilla.yaml" 4 || true')
    for name, path in rows:
        lines.append(f'bash "$ROOT/tmp/fkc_sweep/run_one.sh" {name} "{path}" 4 || true')
    lines.append('echo "[chain] full sweep done at $(date)"')
    chain.write_text("\n".join(lines) + "\n")
    chain.chmod(0o755)
    print(f"wrote chain to {chain}")


if __name__ == "__main__":
    main()
