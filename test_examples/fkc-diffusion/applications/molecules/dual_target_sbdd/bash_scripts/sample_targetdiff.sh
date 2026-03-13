#!/bin/bash

idx1=$1
idx2=$2
numatoms=$3
bs=32
seed=0
EXP_DIR="/scratch/a/aspuru/mskrt/dualdiff"

python scripts/sample_for_pocket.py \
    configs/sampling_targetdiff.yml \
    --data_id $idx1 \
    --result_path ${EXP_DIR}/TEST_outputs/baseline_SDE_targetdiff_bs${bs}_fixednumatoms${numatoms}_seed${seed} \
    --num_samples $bs \
    --num_atoms_to_sample $numatoms \
    --experiment_dir ${EXP_DIR} \
    --seed $seed

python3 scripts/evaluate_targetdiff.py  \
    --result_path ${EXP_DIR}/TEST_outputs_prior/baseline_SDE_targetdiff_bs${bs}_fixednumatoms${numatoms}_seed${seed} \
    --experiment_dir ${EXP_DIR} \
    --sample_path ${EXP_DIR}/TEST_outputs/baseline_SDE_targetdiff_bs${bs}_fixednumatoms${numatoms}_seed${seed} \
    --idx1 $idx1 \
    --idx2 $idx1

python3 scripts/evaluate_targetdiff.py \
    --result_path ${EXP_DIR}/TEST_outputs_prior/baseline_SDE_targetdiff_bs${bs}_fixednumatoms${numatoms}_seed${seed} \
    --experiment_dir ${EXP_DIR} \
    --sample_path ${EXP_DIR}/TEST_outputs/baseline_SDE_targetdiff_bs${bs}_fixednumatoms${numatoms}_seed${seed} \
    --idx1 $idx1 \
    --idx2 $idx2

python3 scripts/align_protein_ligand_score.py  \
    --results_dir ${EXP_DIR}/TEST_outputs_prior/baseline_SDE_targetdiff_bs${bs}_fixednumatoms${numatoms}_seed${seed} \
    --save_dir ${EXP_DIR}/TEST_aligned_SDE_prot_pockets_bs${bs}_fixednumatoms${numatoms}_seed${seed} \
    --experiment_dir ${EXP_DIR} \
    --idx1 $idx1 \
    --idx2 $idx2
