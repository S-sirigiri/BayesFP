import argparse
import os, sys
import shutil
import pickle
import glob

import torch
from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms as trans
from datasets.pl_data import ProteinLigandData, torchify_dict
from models.molopt_score_model import ScorePosNet3D
from scripts.sample_diffusion import compose_sample_diffusion_ligand

from utils.data import PDBProtein
from utils import reconstruct

from rdkit import Chem
import wandb


def pdb_to_pocket_data(pdb_path):
    pocket_dict = PDBProtein(pdb_path).to_dict_atom()
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict={
            'element': torch.empty([0, ], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0, ], dtype=torch.long),
        }
    )

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--exp_note', type=str, default=None)
    parser.add_argument('--idx1', type=int)
    parser.add_argument('--idx2', type=int)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--experiment_dir', type=str, default='./', help="root directory where all experiments are stored")
    parser.add_argument('--aligned_prots_path', type=str, default='./', help="directory where aligned protein pockets are stored")
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--reverse_method', type=str, choices=["SDE", "baseline"], default="SDE")
    parser.add_argument('--inv_temp', type=float, default=1.)
    parser.add_argument('--tempered_score', action="store_true")
    parser.add_argument('--resample_thresh', type=float, default=0.)

    parser.add_argument('--resample', type=int, choices=[0,1])
    parser.add_argument('--num_samples', type=int, required=True)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--num_atoms_to_sample', type=int, default=-1)
    parser.add_argument('--sample_num_atoms', type=str, choices=["fixed"])
    args = parser.parse_args()

    logger = misc.get_logger('evaluate')

    # Load config
    config = misc.load_config(args.config)

    args.batch_size = args.num_samples
    config.sample.num_samples = args.num_samples
    config.sample.reverse_method = args.reverse_method
    config.sample.inv_temp = args.inv_temp
    config.sample.resample = args.resample
    config.sample.num_steps = args.num_steps
    config.sample.seed = args.seed
    config.sample.sample_num_atoms = args.sample_num_atoms
    config.sample.num_atoms_to_sample = args.num_atoms_to_sample
    logger.info(config)

    misc.seed_all(config.sample.seed)

    tags = ["inference"]
    exp_name = f"dualdiff_{config.sample.reverse_method}"
    if "inv_temp" in config['sample']:
        exp_name += f"_invtemp{config.sample.inv_temp}"
    if "resample" in config['sample']:
         exp_name += f"_resample{config.sample.resample}"

    exp_name += f"_bs{args.batch_size}_ns{config.sample.num_steps}_SEED{config.sample.seed}"

    if args.num_atoms_to_sample != -1 and config['sample']['sample_num_atoms'] == 'fixed':
        assert config['sample']['num_atoms_to_sample'] != -1
        exp_name += f"_fixednumatoms{config.sample.num_atoms_to_sample}"
    else:
        print("only operating in fixed atom mode now................."); exit()
    if args.exp_note:
        exp_name += f"{args.exp_note}"

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    logger.info(f"Training Config: {ckpt['config']}")



    wandb.init(project='feynman-kac-3dmols', name=exp_name, tags=tags)
    wandb.run.log_code(".")
    wandb.config.update(ckpt['config'])
    wandb.config.update(args, allow_val_change=True)
    wandb.config.update(config, allow_val_change=True)


    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
    ])

    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        inference_config=config,
        args=args,
    ).to(args.device)
    model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    # Load pocket
    with open(os.path.join(args.experiment_dir, 'data/processed/drug_synergy/synergy_idx_list.pkl'), 'rb') as f:
        synergy_idx_list = pickle.load(f)
    with open(os.path.join(args.experiment_dir, 'data/processed/dock/index_dict.pkl'), 'rb') as f:
        index = pickle.load(f)
    idx_to_smiles = index['idx_to_smiles']
    
    #idx1, idx2 = synergy_idx_list[args.synergy_idx]
    idx1 = args.idx1
    idx2 = args.idx2
    
    pdb_path_1 = os.path.join(args.aligned_prots_path, f"{idx1}/{idx2}/pocket_10A_1_transformed.pdb")
    pdb_path_2 = os.path.join(args.aligned_prots_path, f"{idx1}/{idx2}/pocket_10A_2.pdb")
    
    logger.info(f"pdb_path_1: {pdb_path_1}")
    logger.info(f"pdb_path_2: {pdb_path_2}")
    assert os.path.exists(pdb_path_1)
    assert os.path.exists(pdb_path_2)
    
    data_1 = pdb_to_pocket_data(pdb_path_1)
    data_2 = pdb_to_pocket_data(pdb_path_2)

    data_1 = transform(data_1)
    data_2 = transform(data_2)

    all_pred_pos, all_pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list, dw_traj, id_traj = compose_sample_diffusion_ligand(
        model, data_1, data_2, config.sample.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_steps=config.sample.num_steps,
        pos_only=config.sample.pos_only,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms,
        num_atoms_to_sample=config.sample.num_atoms_to_sample,
        inv_temp = config.sample.inv_temp,
        reverse_method=config.sample.reverse_method,

    )
    result = {
        'data_1': data_1,
        'data_2': data_2,
        'pred_ligand_pos': all_pred_pos,
        'pred_ligand_v': all_pred_v,
        "pred_pos_traj": pred_pos_traj,
        "pred_v_traj": pred_v_traj,
        "dw_traj": dw_traj,
        "id_traj": id_traj
    }
    logger.info('Sample done!')

    # reconstruction
    gen_mols = []
    n_recon_success, n_complete = 0, 0
    print("="*20)
    print("SMILES")
    print("="*20)
    smis = []
    for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_pos, all_pred_v)):
        pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode='add_aromatic')
        try:
            pred_aromatic = trans.is_aromatic_from_index(pred_v, mode='add_aromatic')
            mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
            smiles = Chem.MolToSmiles(mol)
            print(smiles)
        except reconstruct.MolReconsError:
            gen_mols.append(None)
            smis.append(None)
            continue
        n_recon_success += 1

        if '.' in smiles:
            gen_mols.append(None)
            smis.append(None)
            continue
        n_complete += 1
        gen_mols.append(mol)
        smis.append(smiles)
    result['mols'] = gen_mols
    result['smis'] = smis
    logger.info('Reconstruction done!')
    logger.info(f'n recon: {n_recon_success} n complete: {n_complete}')

    result_path = os.path.join(args.result_path, f"{idx1}/{idx2}")
    logger.info(f"result_path: {result_path}")
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    torch.save(result, os.path.join(result_path, f'sample.pt'))
    mols_save_path = os.path.join(result_path, f'sdf')
    os.makedirs(mols_save_path, exist_ok=True)
    for idx, mol in enumerate(gen_mols):
        if mol is not None:
            sdf_writer = Chem.SDWriter(os.path.join(mols_save_path, f'{idx:03d}.sdf'))
            sdf_writer.write(mol)
            sdf_writer.close()
    logger.info(f'Results are saved in {result_path}')
