import argparse
import os
import shutil
import time

import math
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH, FOLLOW_BATCH_INPAINT
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from utils.evaluation import atom_num


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior', fixed_atom_num=False, num_atoms_to_sample=-1, reverse_method="baseline", inv_temp=1,):
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)

        t1 = time.time()
        with torch.no_grad():
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size, fixed_atom_num).astype(int) for _ in range(n_data)]
                print("ligand_num_atoms", ligand_num_atoms)
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            elif sample_num_atoms == "fixed":
                assert num_atoms_to_sample != -1
                ligand_num_atoms = [int(num_atoms_to_sample) for _ in range(n_data)]
                print("ligand_num_atoms", ligand_num_atoms)
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            else:
                raise ValueError

            # init ligand pos
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

            # init ligand v
            if pos_only:
                init_ligand_v = batch.ligand_atom_feature_full
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                init_ligand_v = log_sample_categorical(uniform_logits)

            r = model.sample_diffusion(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,
                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode,
                reverse_method=reverse_method, inv_temp=inv_temp
            )

            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            # unbatch pos
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]  # num_samples * [num_atoms_i, 3]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]

            if not pos_only:
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list


def compose_sample_diffusion_ligand(model, data_1, data_2, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior', reverse_method="baseline", inv_temp=1, fixed_atom_num=False, num_atoms_to_sample=-1):
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch_1 = Batch.from_data_list([data_1.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)
        batch_2 = Batch.from_data_list([data_2.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)

        t1 = time.time()
        with torch.no_grad():
            batch_protein_1 = batch_1.protein_element_batch
            batch_protein_2 = batch_2.protein_element_batch

            if sample_num_atoms == 'prior': ## this is default
                pocket_size_1 = atom_num.get_space_size(data_1.protein_pos.detach().cpu().numpy())
                pocket_size_2 = atom_num.get_space_size(data_2.protein_pos.detach().cpu().numpy())

                ligand_num_atoms_1 = [atom_num.sample_atom_num(pocket_size_1, fixed_atom_num).astype(int) for _ in range(n_data)]
                ligand_num_atoms_2 = [atom_num.sample_atom_num(pocket_size_2, fixed_atom_num).astype(int) for _ in range(n_data)]

                ligand_num_atoms = [math.ceil((num_1 + num_2) / 2) for num_1, num_2 in zip(ligand_num_atoms_1, ligand_num_atoms_2)] ## list of len BS that seems to sample from distr of atoms
                print("ligand_num_atoms", ligand_num_atoms)
                if fixed_atom_num:
                    assert [l == ligand_num_atoms[0] for l in ligand_num_atoms]

                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device) ## batch index spread over number of atoms

            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                raise ValueError
                # batch_ligand = batch.ligand_element_batch
                # ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            elif sample_num_atoms == "fixed":
                assert num_atoms_to_sample != -1
                ligand_num_atoms = [int(num_atoms_to_sample) for _ in range(n_data)]
                print("ligand_num_atoms", ligand_num_atoms)
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            else:
                raise ValueError

            # init ligand pos
            center_pos_1 = scatter_mean(batch_1.protein_pos, batch_protein_1, dim=0) # computes mean of protein atom positions
            center_pos_2 = scatter_mean(batch_2.protein_pos, batch_protein_2, dim=0)
            center_pos = (center_pos_1 + center_pos_2) / 2.
            

            batch_center_pos = center_pos[batch_ligand] # should be same values 
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos) ## center of protein pocket with noise added to it?

            # init ligand v
            if pos_only:
                raise ValueError
                # init_ligand_v = batch.ligand_atom_feature_full
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                init_ligand_v = log_sample_categorical(uniform_logits)

            assert isinstance(model, ScorePosNet3D)
            r = model.compose_sample_diffusion(
                protein_pos_1=batch_1.protein_pos,
                protein_v_1=batch_1.protein_atom_feature.float(),
                batch_protein_1=batch_protein_1,
                
                protein_pos_2=batch_2.protein_pos, 
                protein_v_2=batch_2.protein_atom_feature.float(), 
                batch_protein_2=batch_protein_2,

                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode,
                reverse_method=reverse_method, inv_temp=inv_temp,
            )
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            dw_traj = r['dw_traj']
            id_traj = r['id_traj']
            # unbatch pos
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]  # num_samples * [num_atoms_i, 3]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]

            if not pos_only:
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list, dw_traj, id_traj


def deep_compose_sample_diffusion_ligand(model, data_1, data_2, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior'):
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch_1 = Batch.from_data_list([data_1.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)
        batch_2 = Batch.from_data_list([data_2.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)

        t1 = time.time()
        with torch.no_grad():
            batch_protein_1 = batch_1.protein_element_batch
            batch_protein_2 = batch_2.protein_element_batch

            if sample_num_atoms == 'prior':
                pocket_size_1 = atom_num.get_space_size(data_1.protein_pos.detach().cpu().numpy())
                pocket_size_2 = atom_num.get_space_size(data_2.protein_pos.detach().cpu().numpy())

                ligand_num_atoms_1 = [atom_num.sample_atom_num(pocket_size_1).astype(int) for _ in range(n_data)]
                ligand_num_atoms_2 = [atom_num.sample_atom_num(pocket_size_2).astype(int) for _ in range(n_data)]

                ligand_num_atoms = [math.ceil((num_1 + num_2) / 2) for num_1, num_2 in zip(ligand_num_atoms_1, ligand_num_atoms_2)]

                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)

            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                raise ValueError
                # batch_ligand = batch.ligand_element_batch
                # ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            else:
                raise ValueError

            # init ligand pos
            center_pos_1 = scatter_mean(batch_1.protein_pos, batch_protein_1, dim=0)
            center_pos_2 = scatter_mean(batch_2.protein_pos, batch_protein_2, dim=0)
            center_pos = (center_pos_1 + center_pos_2) / 2.

            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

            # init ligand v
            if pos_only:
                raise ValueError
                # init_ligand_v = batch.ligand_atom_feature_full
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                init_ligand_v = log_sample_categorical(uniform_logits)

            assert isinstance(model, ScorePosNet3D)
            r = model.deep_compose_sample_diffusion(
                protein_pos_1=batch_1.protein_pos,
                protein_v_1=batch_1.protein_atom_feature.float(),
                batch_protein_1=batch_protein_1,
                
                protein_pos_2=batch_2.protein_pos, 
                protein_v_2=batch_2.protein_atom_feature.float(), 
                batch_protein_2=batch_protein_2,

                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode
            )
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            # unbatch pos
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]  # num_samples * [num_atoms_i, 3]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]

            if not pos_only:
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list



def sample_diffusion_ligand_inpaint(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior'):
    
    # print(data)
    # ProteinLigandData(
    #     protein_element=[379],
    #     protein_molecule_name='pocket',
    #     protein_pos=[379, 3],
    #     protein_is_backbone=[379],
    #     protein_atom_name=[379],
    #     protein_atom_to_aa_type=[379],
    #     frag_smiles='Fc1ccccc1',
    #     frag_element=[7],
    #     frag_pos=[7, 3],
    #     frag_bond_index=[2, 14],
    #     frag_bond_type=[14],
    #     frag_center_of_mass=[3],
    #     frag_atom_feature=[7, 8],
    #     frag_hybridization=[7],
    #     frag_nbh_list={
    #         0=[1],
    #         1=[3],
    #         2=[2],
    #         3=[2],
    #         4=[2],
    #         5=[2],
    #         6=[2]
    #     },
    #     ligand_element=[0],
    #     ligand_pos=[0, 3],
    #     ligand_atom_feature=[0, 8],
    #     ligand_bond_index=[2, 0],
    #     ligand_bond_type=[0],
    #     ligand_nbh_list={},
    #     protein_atom_feature=[379, 27],
    #     frag_atom_feature_full=[7]
    # )
    # 7, 2, 2, 2, 2, 2, 2
    
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH_INPAINT).to(device)

        t1 = time.time()
        with torch.no_grad():
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                
                assert hasattr(data, 'frag_element')
                frag_num_atom = data.frag_element.shape[0]
                print('[Before] ligand_num_atoms:', ligand_num_atoms)
                print('frag_num_atom:', frag_num_atom)
                ligand_num_atoms = [max(num_a, frag_num_atom+1) for num_a in ligand_num_atoms]
                print('[After] ligand_num_atoms:', ligand_num_atoms)
                in_batch_indices = torch.cat([torch.arange(num_a) for num_a in ligand_num_atoms])
    
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            else:
                raise ValueError
            
            batch_frag = batch.frag_element_batch
            
            # print(batch_ligand.shape)
            # print(batch_ligand)

            # init ligand pos
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)
            
            
            ref_mask = (in_batch_indices < frag_num_atom) # True = using ref_frag
            


            # init ligand v
            if pos_only:
                init_ligand_v = batch.ligand_atom_feature_full
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                init_ligand_v = log_sample_categorical(uniform_logits)

            # print(batch)
            # ProteinLigandDataBatch(
            # protein_element=[3790],
            # protein_element_batch=[3790],
            # protein_element_ptr=[11],
            # protein_molecule_name=[10],
            # protein_pos=[3790, 3],
            # protein_is_backbone=[3790],
            # protein_atom_name=[10],
            # protein_atom_to_aa_type=[3790],
            # frag_smiles=[10],
            # frag_element=[70],
            # frag_pos=[70, 3],
            # frag_bond_index=[2, 140],
            # frag_bond_type=[140],
            # frag_center_of_mass=[30],
            # frag_atom_feature=[70, 8],
            # frag_hybridization=[10],
            # frag_nbh_list={
            #     0=[10],
            #     1=[10],
            #     2=[10],
            #     3=[10],
            #     4=[10],
            #     5=[10],
            #     6=[10]
            # },
            # ligand_element=[0],
            # ligand_element_batch=[0],
            # ligand_element_ptr=[11],
            # ligand_pos=[0, 3],
            # ligand_atom_feature=[0, 8],
            # ligand_bond_index=[2, 0],
            # ligand_bond_type=[0],
            # ligand_bond_type_batch=[0],
            # ligand_bond_type_ptr=[11],
            # ligand_nbh_list={},
            # protein_atom_feature=[3790, 27],
            # frag_atom_feature_full=[70]
            # )
            

            assert isinstance(model, ScorePosNet3D)
            r = model.sample_diffusion_inpaint(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,

                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                
                frag_pos=batch.frag_pos, 
                frag_v=batch.frag_atom_feature_full,
                batch_frag=batch_frag,
                
                ref_mask=ref_mask,
                
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode
            )
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            # unbatch pos
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]  # num_samples * [num_atoms_i, 3]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]

            if not pos_only:
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('-i', '--data_id', type=int)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--result_path', type=str, default='./outputs')
    args = parser.parse_args()

    logger = misc.get_logger('sampling')

    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    # Load dataset
    dataset, subsets = get_dataset(
        config=ckpt['config'].data,
        transform=transform
    )
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    data = test_set[args.data_id]
    pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
        model, data, config.sample.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_steps=config.sample.num_steps,
        pos_only=config.sample.pos_only,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms
    )
    result = {
        'data': data,
        'pred_ligand_pos': pred_pos,
        'pred_ligand_v': pred_v,
        # 'pred_ligand_pos_traj': pred_pos_traj,
        # 'pred_ligand_v_traj': pred_v_traj,
        'time': time_list
    }
    logger.info('Sample done!')

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    torch.save(result, os.path.join(result_path, f'result_{args.data_id}.pt'))
