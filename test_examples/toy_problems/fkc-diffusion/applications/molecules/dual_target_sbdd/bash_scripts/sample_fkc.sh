#!/bin/bash

idx1=$1
idx2=$2
numatoms=$3
inv_temp=$4
resample=1
num_steps=1000
seed=0
sample_num_atoms="fixed"
num_samples=32
exp_note=""
thresh=0.6

EXP_DIR="/scratch/a/aspuru/mskrt/dualdiff"

python scripts/compose_sample_score.py \
        configs/sampling_fkc.yml \
    --idx1 $idx1 \
    --idx2 $idx2 \
    --num_atoms_to_sample $numatoms \
    --inv_temp $inv_temp \
    --num_steps $num_steps \
    --seed $seed \
    --sample_num_atoms $sample_num_atoms \
    --num_samples $num_samples \
    --resample $resample \
    --resample_thresh $thresh \
    --result_path ${EXP_DIR}/TEST_outputs/dualtarget_SDE_invtemp${inv_temp}_resample${resample}_bs${num_samples}_ns${num_steps}_seed${seed}_fixednumatoms${numatoms}${exp_note} \
    --aligned_prots_path ${EXP_DIR}/TEST_aligned_SDE_prot_pockets_bs${num_samples}_fixednumatoms${numatoms}_seed${seed} \
    --experiment_dir ${EXP_DIR}

python3 scripts/evaluate_compose.py --sample_path ${EXP_DIR}/TEST_outputs/dualtarget_SDE_invtemp${inv_temp}_resample${resample}_bs${num_samples}_ns${num_steps}_seed${seed}_fixednumatoms${numatoms}${exp_note} \
    --idx1 $idx1 \
    --idx2 $idx2 \
    --experiment_dir ${EXP_DIR}

