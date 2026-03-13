import pandas as pd
import torch
pd.options.display.max_colwidth = 1000
from glob import glob
from rdkit import Chem
from tdc import Evaluator, Oracle
import numpy as np
from scipy import stats

NAME2PATH = {
        "invtemp2.0_noresample_bs32": "/scratch/a/aspuru/mskrt/dualdiff/outputs/dualtarget_SDE_invtemp2.0_resample0_bs32_ns1000_seed0_fixednumatomsMOLLEN_SAVEDTRAJ/IDX1/IDX2/results.csv",
         }

IDXS = [(29,371)]
idx1, idx2 = 29, 371


MOLLENS = [19]


def mean(df, col):
    return df[col].mean()

def median(df, col):
    return df[col].median()

def max_(df, col):
    return df[col].max()

def min_(df, col):
    return df[col].min()

def better_than_ref(df, col1, col2, ref_mol1, ref_mol2):

    count = ((df[col1] < ref_mol1) & (df[col2] < ref_mol2)).sum()
    return count

def diversity(smis):
    evaluator = Evaluator(name = 'Diversity')
    return evaluator(smis)


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return None



for model in NAME2PATH:
    print("="*200)

    mean_pid1_dock_arr, mean_pid2_dock_arr = [], []
    ranks = []

    path_template = NAME2PATH[model]
    smis = []
    # loop over protein pairs
    paths = path_template.replace("IDX1", str(idx1)).replace("IDX2", str(idx2))
    temp_top1_pid1_dock_arr, temp_top1_pid2_dock_arr = [], []
    temp_worst = []
    top1_prod = []
    for mollen in MOLLENS:
        path_ = paths.replace("MOLLEN", str(mollen))
        print("loading path_............", path_)
        path_ = glob(path_)
        assert len(path_) == 1
        path_ = path_[0]

        df = pd.read_csv(path_)
        df['prod'] = df['pid1_dock'] * df['pid2_dock']
        fitness = np.array(df['prod'].tolist())
        #sorted_ids = np.argsort(fitness)
        #top_ids = np.stack((sorted_ids[:5], sorted_ids[-5:]))
        #top_ids = np.stack((sorted_ids[:16], sorted_ids[-16:]))
        
        saved_weights_path =  str(path_).replace("results.csv", "sample.pt")
        weights = torch.load(saved_weights_path)['dw_traj'] # 1000, 32, 1
        for ts in range(int(len(weights) * 1.0)):
            curr_w = weights[ts].squeeze().detach().cpu().numpy()
            sorted_ids = np.argsort(fitness)
            top_ids = np.stack((sorted_ids[:5], sorted_ids[-5:]))
            res_1 = stats.spearmanr(fitness[top_ids][0], curr_w[top_ids][0]).statistic
            #res_1 = stats.spearmanr(fitness, curr_w).statistic
            ranks.append(res_1)
print(ranks)
