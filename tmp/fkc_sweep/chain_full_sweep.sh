#!/bin/bash
set -u
ROOT=/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion
echo "[chain] starting full sweep at $(date)"
bash "$ROOT/tmp/fkc_sweep/run_one.sh" vanilla_oo2 "$ROOT/tmp/fkc_sweep/cfg_vanilla.yaml" 4 || true
bash "$ROOT/tmp/fkc_sweep/run_one.sh" linear_m0_b5000 "/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion/tmp/fkc_sweep/sweep_configs/cfg_linear_m0_b5000.yaml" 4 || true
bash "$ROOT/tmp/fkc_sweep/run_one.sh" linear_m0p0001_b5000 "/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion/tmp/fkc_sweep/sweep_configs/cfg_linear_m0p0001_b5000.yaml" 4 || true
bash "$ROOT/tmp/fkc_sweep/run_one.sh" linear_m0p001_b500 "/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion/tmp/fkc_sweep/sweep_configs/cfg_linear_m0p001_b500.yaml" 4 || true
bash "$ROOT/tmp/fkc_sweep/run_one.sh" linear_m0p01_b500 "/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion/tmp/fkc_sweep/sweep_configs/cfg_linear_m0p01_b500.yaml" 4 || true
bash "$ROOT/tmp/fkc_sweep/run_one.sh" linear_m0p1_b500 "/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion/tmp/fkc_sweep/sweep_configs/cfg_linear_m0p1_b500.yaml" 4 || true
bash "$ROOT/tmp/fkc_sweep/run_one.sh" softplus_m0_b5000 "/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion/tmp/fkc_sweep/sweep_configs/cfg_softplus_m0_b5000.yaml" 4 || true
bash "$ROOT/tmp/fkc_sweep/run_one.sh" softplus_m0p0001_b5000 "/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion/tmp/fkc_sweep/sweep_configs/cfg_softplus_m0p0001_b5000.yaml" 4 || true
bash "$ROOT/tmp/fkc_sweep/run_one.sh" softplus_m0p001_b5000 "/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion/tmp/fkc_sweep/sweep_configs/cfg_softplus_m0p001_b5000.yaml" 4 || true
bash "$ROOT/tmp/fkc_sweep/run_one.sh" softplus_m0p01_b7000 "/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion/tmp/fkc_sweep/sweep_configs/cfg_softplus_m0p01_b7000.yaml" 4 || true
bash "$ROOT/tmp/fkc_sweep/run_one.sh" softplus_m0p1_b500 "/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion/tmp/fkc_sweep/sweep_configs/cfg_softplus_m0p1_b500.yaml" 4 || true
bash "$ROOT/tmp/fkc_sweep/run_one.sh" squared_hinge_m0_b500000 "/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion/tmp/fkc_sweep/sweep_configs/cfg_squared_hinge_m0_b500000.yaml" 4 || true
bash "$ROOT/tmp/fkc_sweep/run_one.sh" squared_hinge_m0p0001_b500000 "/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion/tmp/fkc_sweep/sweep_configs/cfg_squared_hinge_m0p0001_b500000.yaml" 4 || true
bash "$ROOT/tmp/fkc_sweep/run_one.sh" squared_hinge_m0p001_b50000 "/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion/tmp/fkc_sweep/sweep_configs/cfg_squared_hinge_m0p001_b50000.yaml" 4 || true
bash "$ROOT/tmp/fkc_sweep/run_one.sh" squared_hinge_m0p01_b7000 "/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion/tmp/fkc_sweep/sweep_configs/cfg_squared_hinge_m0p01_b7000.yaml" 4 || true
bash "$ROOT/tmp/fkc_sweep/run_one.sh" squared_hinge_m0p1_b5000 "/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion/tmp/fkc_sweep/sweep_configs/cfg_squared_hinge_m0p1_b5000.yaml" 4 || true
bash "$ROOT/tmp/fkc_sweep/run_one.sh" squared_distance_hinge_m0p0001_b500000 "/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion/tmp/fkc_sweep/sweep_configs/cfg_squared_distance_hinge_m0p0001_b500000.yaml" 4 || true
bash "$ROOT/tmp/fkc_sweep/run_one.sh" squared_distance_hinge_m0p001_b50000 "/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion/tmp/fkc_sweep/sweep_configs/cfg_squared_distance_hinge_m0p001_b50000.yaml" 4 || true
bash "$ROOT/tmp/fkc_sweep/run_one.sh" squared_distance_hinge_m0p01_b7000 "/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion/tmp/fkc_sweep/sweep_configs/cfg_squared_distance_hinge_m0p01_b7000.yaml" 4 || true
bash "$ROOT/tmp/fkc_sweep/run_one.sh" squared_distance_hinge_m0p1_b500 "/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion/tmp/fkc_sweep/sweep_configs/cfg_squared_distance_hinge_m0p1_b500.yaml" 4 || true
echo "[chain] full sweep done at $(date)"
