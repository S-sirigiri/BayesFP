import pickle 
import os, sys
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import PDBQTMolecule
from tqdm.auto import tqdm
import argparse
from rdkit.Chem import rdMolTransforms
from copy import deepcopy

from utils.data import PDBProtein, parse_sdf_file

from glob import glob
import numpy as np
import shutil

import tempfile

def split_range(n, k, i):
    # Validate the part requested
    if not (0 <= i < k):
        raise ValueError("i must be in the range [0, k)")
        
    # Compute size of each chunk
    chunk_size = n // k
    # Compute the number of chunks that will have an extra element
    leftovers = n % k
    
    # Determine the starting point of the ith part
    start = i * chunk_size + min(i, leftovers)
    # If i is less than leftovers, this part will be one element longer
    end = start + chunk_size + (1 if i < leftovers else 0)
    
    return range(start, end)


def load_pickle(fp):
    with open(fp, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(obj, fp):
    with open(fp, 'wb') as f:
        pickle.dump(obj, f)

def transform_pdb(trans_matrix, pdb_path, target_pdb_path):
    with open(pdb_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    new_lines.extend(lines[:2])
    for line in lines[2:-1]:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        coords = np.array([x, y, z, 1.0])
        new_coords = np.dot(trans_matrix, coords)
        new_lines.append(
            line[:30] + 
            f"{new_coords[0]:<8.3f}" + 
            f"{new_coords[1]:<8.3f}" + 
            f"{new_coords[2]:<8.3f}" + 
            line[54:])

    new_lines.append(lines[-1])
        
    with open(target_pdb_path, 'w') as f:
        for line in new_lines:
            f.write(line)

parser = argparse.ArgumentParser()
parser.add_argument('--idx1', type=int)  
parser.add_argument('--idx2', type=int)  
parser.add_argument('--results_dir', type=str)  
parser.add_argument('--save_dir', type=str)  
parser.add_argument('--experiment_dir', type=str, default='./', help="root directory where all experiments are stored")
args = parser.parse_args()



def print_coordinates(molecule):
    for i, atom in enumerate(molecule.GetAtoms()):
        positions = molecule.GetConformer().GetAtomPosition(i)
        print(atom.GetSymbol(), positions.x, positions.y, positions.z)
        
def get_coordinates(molecule):
    coordinates = []
    for i, atom in enumerate(molecule.GetAtoms()):
        positions = molecule.GetConformer().GetAtomPosition(i)
        coordinates.append([positions.x, positions.y, positions.z])
    

with open(os.path.join(args.experiment_dir, 'data/processed/drug_synergy/synergy_idx_list.pkl'), 'rb') as f:
    synergy_idx_list = pickle.load(f)
with open(os.path.join(args.experiment_dir, 'data/processed/dock/index_dict.pkl'), 'rb') as f:
    index_dict = pickle.load(f)
idx_to_smiles = index_dict['idx_to_smiles']


idx1 = args.idx1
idx2 = args.idx2
smiles1 = idx_to_smiles[idx1]
smiles2 = idx_to_smiles[idx2]

save_dir = f"{args.save_dir}/{idx1}/{idx2}"
os.makedirs(save_dir, exist_ok=True)

res_list_1 = load_pickle(f'{args.results_dir}/{idx1}/{idx1}/eval_all.pkl')
res_list_2 = load_pickle(f'{args.results_dir}/{idx1}/{idx2}/eval_all.pkl')



best_align = None
best_score = 999.

for res1, res2 in zip(res_list_1, res_list_2):

    if res1['mol'] is None:
        continue
    
    mol1 = res1['docked_mol']
    mol2 = res2['docked_mol']
    if mol1 is None:
        continue
    
    score1 = res1['vina']['dock'][0]['affinity']
    score2 = res2['vina']['dock'][0]['affinity']

    score = max(score1, score2)
    
    
    rmsd, trans_matrix, atom_map = AllChem.GetBestAlignmentTransform(mol1, mol2)
    
    
    if score < best_score:
        best_score = score 
        best_align = {
            'res1': res1,
            'res2': res2, 
            'rmsd': rmsd, 
            'score': score, 
            'trans_matrix': trans_matrix,
            'atom_map': atom_map,
        }
        

trans_matrix = best_align['trans_matrix']

anchor_mol1 = deepcopy(best_align['res1']['docked_mol'])
anchor_mol2 = deepcopy(best_align['res2']['docked_mol'])


anchor_mol1_transformed = deepcopy(anchor_mol1)
rdMolTransforms.TransformConformer(anchor_mol1_transformed.GetConformer(0), trans_matrix)


protein_path_1 = glob(os.path.join(args.experiment_dir, f"data/processed/dock/ligand_protein_dataset_v2/{smiles1}/*/protein_clean.pdb"))[0]
protein_path_2 = glob(os.path.join(args.experiment_dir, f"data/processed/dock/ligand_protein_dataset_v2/{smiles2}/*/protein_clean.pdb"))[0]
pocket_path_1 = glob(os.path.join(args.experiment_dir, f"data/processed/dock/ligand_protein_dataset_v2/{smiles1}/*/pocket_10A.pdb"))[0]
pocket_path_2 = glob(os.path.join(args.experiment_dir, f"data/processed/dock/ligand_protein_dataset_v2/{smiles2}/*/pocket_10A.pdb"))[0]
ligand_path_1 = glob(os.path.join(args.experiment_dir, f"data/processed/dock/ligand_protein_dataset_v2/{smiles1}/*/ligand.sdf"))[0]
ligand_path_2 = glob(os.path.join(args.experiment_dir, f"data/processed/dock/ligand_protein_dataset_v2/{smiles2}/*/ligand.sdf"))[0]


with open(os.path.join(save_dir, 'best_align.pkl'), 'wb') as f:
    pickle.dump(best_align, f)
    
shutil.copy2(protein_path_1, os.path.join(save_dir, 'protein_1.pdb'))
shutil.copy2(pocket_path_1, os.path.join(save_dir, 'pocket_10A_1.pdb'))
shutil.copy2(ligand_path_1, os.path.join(save_dir, 'gt_ligand_1.sdf'))
shutil.copy2(protein_path_2, os.path.join(save_dir, 'protein_2.pdb'))
shutil.copy2(pocket_path_2, os.path.join(save_dir, 'pocket_10A_2.pdb'))
shutil.copy2(ligand_path_2, os.path.join(save_dir, 'gt_ligand_2.sdf'))

transform_pdb(trans_matrix, protein_path_1, os.path.join(save_dir, 'protein_1_transformed.pdb'))
transform_pdb(trans_matrix, pocket_path_1, os.path.join(save_dir, 'pocket_10A_1_transformed.pdb'))

with Chem.SDWriter(os.path.join(save_dir, 'anchor_mol_1.sdf')) as w:
    w.write(anchor_mol1)
with Chem.SDWriter(os.path.join(save_dir, 'anchor_mol_1_transformed.sdf')) as w:
    w.write(anchor_mol1_transformed)
with Chem.SDWriter(os.path.join(save_dir, 'anchor_mol_2.sdf')) as w:
    w.write(anchor_mol2)

print(f"{idx1}/{idx2}")





