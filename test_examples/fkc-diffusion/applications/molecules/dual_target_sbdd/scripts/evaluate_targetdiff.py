import argparse
import os, sys

import numpy as np
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from copy import deepcopy
from rdkit import Chem

from utils import misc
from utils.evaluation import scoring_func
from utils.evaluation.docking_vina import VinaDockingTask

from multiprocessing import Pool
from functools import partial
from glob import glob
import pickle
import re
from datasets.protein_ligand import parse_sdf_file_mol
import pandas as pd

pattern_bs = r"bs(\d+)"
pattern_numatoms = r"numatoms(\d+)"

def eval_single_datapoint(ligand_rdmol_list, protein_path, args, center):

    results = []
    
    n_eval_success = 0

    for rdmol in tqdm(ligand_rdmol_list):
        
        if rdmol is None:
            results.append({
                'mol': None,
                'smiles': None,
                'protein_path': protein_path,
            })
            continue

        try:
            Chem.SanitizeMol(rdmol)
        except Chem.rdchem.AtomValenceException as e:
            err = e
            N4_valence = re.compile(u"Explicit valence for atom # ([0-9]{1,}) N, 4, is greater than permitted")
            index = N4_valence.findall(err.args[0])
            if len(index) > 0:
                rdmol.GetAtomWithIdx(int(index[0])).SetFormalCharge(1)
                Chem.SanitizeMol(rdmol)
                
        smiles = Chem.MolToSmiles(rdmol)
        
        print(smiles)

        if '.' in smiles:
            results.append({
                'mol': rdmol,
                'smiles': smiles,
                'protein_path': protein_path,
            })
            continue
        
        mol = rdmol

        chem_results = scoring_func.get_chem(mol)

        vina_task = VinaDockingTask(
            protein_path=protein_path,
            ligand_rdmol=deepcopy(mol),
            size_factor=None,
            center=center.tolist(),
        )
            
        score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
        minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
        vina_results = {
            'score_only': score_only_results,
            'minimize': minimize_results
        }
        if args.docking_mode == 'vina_full':
            dock_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
            vina_results.update({
                'dock': dock_results,
            })
            print(dock_results[0]['affinity'])
        elif args.docking_mode == 'vina_score':
            pass
        else:
            raise NotImplementedError
        
        n_eval_success += 1
            
        results.append({
            'mol': mol,
            'smiles': smiles,
            'protein_path': protein_path,
            'chem_results': chem_results,
            'vina': vina_results,
            'docked_mol': Chem.MolFromPDBBlock(vina_results['dock'][0]['pose'])
        })
    logger.info(f'Evaluate No {id} done! {len(ligand_rdmol_list)} samples in total. {n_eval_success} eval success!')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx1', type=int)  
    parser.add_argument('--idx2', type=int)  
    parser.add_argument('--sample_path', type=str)
    
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--docking_mode', type=str, default='vina_full',
                        choices=['vina_full', 'vina_score'])
    parser.add_argument('--exhaustiveness', type=int, default=32)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--experiment_dir', type=str, default='./', help="root directory where all experiments are stored")
    
    args = parser.parse_args()
    
    with open(os.path.join(args.experiment_dir, 'data/processed/drug_synergy/synergy_idx_list.pkl'), 'rb') as f:
        synergy_idx_list = pickle.load(f)
    with open(os.path.join(args.experiment_dir, 'data/processed/dock/index_dict.pkl'), 'rb') as f:
        index_dict = pickle.load(f)

    idx_to_smiles = index_dict['idx_to_smiles']
    idx1, idx2 = args.idx1, args.idx2
    smiles1 = idx_to_smiles[idx1]
    smiles2 = idx_to_smiles[idx2]
    
    

    if args.result_path:
        os.makedirs(os.path.join(args.result_path, f'{idx1}/{idx2}'), exist_ok=True)
    logger = misc.get_logger('evaluate', args.result_path)
    logger.info(args)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    logger.info(f'synergy_idx: ({idx1}, {idx2})')

    
    ligand_meta_file = os.path.join(args.sample_path, f'{idx1}/sample.pt')
    assert os.path.exists(ligand_meta_file), f"{ligand_meta_file}"
    ligand_meta = torch.load(ligand_meta_file)
    ligand_rdmol_list = ligand_meta['mols']
    
    
    protein_path = glob(os.path.join(args.experiment_dir, f"data/processed/dock/ligand_protein_dataset_v2/{smiles2}/*/protein_clean.pdb"))[0]

    pid = str(protein_path).split("/")[-2]
    smi=smiles2
    print("smi, protein_path", smi, protein_path)
    anchor_ligand_dict = parse_sdf_file_mol(glob(os.path.join(args.experiment_dir, f"data/processed/dock/ligand_protein_dataset_v2/{smi}/*/ligand.sdf"))[0])
    ref_path = str(glob(os.path.join(args.experiment_dir, f"data/processed/dock/ligand_protein_dataset_v2/{smi}/*/ligand.sdf"))[0])
    ref_ligand_mol = Chem.SDMolSupplier(ref_path)

    print("docking reference molecule..................")
    ref_ligand_results = eval_single_datapoint(ref_ligand_mol, protein_path, args, center=anchor_ligand_dict['center_of_mass'])
    vina_dock_ref = ref_ligand_results[0]['vina']['dock'][0]['affinity']

    
    print("docking generated molecules..................")
    testset_results = eval_single_datapoint(ligand_rdmol_list, protein_path, args, center=anchor_ligand_dict['center_of_mass'])

    if args.result_path:
        with open(os.path.join(args.result_path, f'{idx1}/{idx2}/eval_all.pkl'), 'wb') as f:
            pickle.dump(testset_results, f)
    
    print("Finished")
    bs_value, numatoms_value = None, None
    bs_match = re.search(pattern_bs, ligand_meta_file)
    if bs_match:
        bs_value = int(bs_match.group(1))
    numatoms_match = re.search(pattern_numatoms, ligand_meta_file)
    if numatoms_match:
        numatoms_value = int(numatoms_match.group(1)) 

    qed = [x['chem_results']['qed'] for x in testset_results if x['mol'] is not None]
    sa = [x['chem_results']['sa'] for x in testset_results if x['mol'] is not None]
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))
    if args.docking_mode in ['vina', 'qvina']:
        vina = [x['vina'][0]['affinity'] for x in testset_results if x['mol'] is not None]
        logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
    elif args.docking_mode in ['vina_full', 'vina_score']:
        vina_score_only = [x['vina']['score_only'][0]['affinity'] for x in testset_results if x['mol'] is not None]
        vina_min = [x['vina']['minimize'][0]['affinity'] for x in testset_results if x['mol'] is not None]
        logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
        if args.docking_mode == 'vina_full':
            vina_dock = [x['vina']['dock'][0]['affinity'] for x in testset_results if x['mol'] is not None]
            logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))

    i=1
    data={}
    for x in testset_results:
        if x['mol'] is None: continue
        s = x['smiles']
        if s not in data:
            data[s] = {}

        data[s]['idx1'] = idx1
        data[s]['idx2'] = idx2
        data[s][f"pid{i+1}_dock"] = x['vina']['dock'][0]['affinity']
        data[s][f"pid{i+1}_refsmi"] = smi
        data[s][f"pid{i+1}_refsmi_dock"] = vina_dock_ref
        data[s][f'pid{i+1}'] = pid
        data[s]['qed'] = x['chem_results']['qed']
        data[s]['sa'] = x['chem_results']['sa']
        data[s]['bs'] = bs_value
        data[s]['numatoms'] = numatoms_value
        data[s][f'path{i+1}'] = protein_path

    df = pd.DataFrame.from_dict(data, orient='index')

    # Reset the index so that the keys become a column
    df = df.reset_index().rename(columns={'index': 'smi'})
    print("saving to...........", os.path.join(args.result_path, f'{idx1}/{idx2}/results.csv'))
    df.to_csv(os.path.join(args.result_path, f'{idx1}/{idx2}/results.csv'))
