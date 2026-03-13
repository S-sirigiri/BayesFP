import pandas as pd
pd.options.display.max_colwidth = 1000
from glob import glob
from rdkit import Chem
from tdc import Evaluator, Oracle
import numpy as np

############## TO MODIFY ###############
# PATH TEMPLATE DICT
NAME2PATH = {
        # name_of_model : path_to_model_results --> note that path must contain MOLLEN which will replace with mol length and IDX1, IDX2 which will replace with protein IDs; see below
        "baseline": "./outputs_prior/baseline_SDE_targetdiff_bs32_fixednumatomsMOLLEN_seed0/IDX1/IDX2/results.csv",
        "invtemp0.5_noresample_bs32": "./outputs/dualtarget_SDE_invtemp0.5_resample0_bs32_ns1000_seed0_fixednumatomsMOLLEN/IDX1/IDX2/results.csv",
        "invtemp0.5_resample_bs32": "./outputs/dualtarget_SDE_invtemp0.5_resample1_bs32_ns1000_seed0_fixednumatomsMOLLEN_resampletault0.6/IDX1/IDX2/results.csv",
        "invtemp1.0_noresample_bs32": "./outputs/dualtarget_SDE_invtemp1.0_resample0_bs32_ns1000_seed0_fixednumatomsMOLLEN/IDX1/IDX2/results.csv",
        "invtemp1.0_resample_bs32": "./outputs/dualtarget_SDE_invtemp1.0_resample1_bs32_ns1000_seed0_fixednumatomsMOLLEN_resampletault0.6/IDX1/IDX2/results.csv",
        "invtemp2.0_noresample_bs32": "./outputs/dualtarget_SDE_invtemp2.0_resample0_bs32_ns1000_seed0_fixednumatomsMOLLEN/IDX1/IDX2/results.csv",
        "invtemp2.0_resample_bs32": "./outputs/dualtarget_SDE_invtemp2.0_resample1_bs32_ns1000_seed0_fixednumatomsMOLLEN_resampletault0.6/IDX1/IDX2/results.csv",
         }

# list of protein pair IDs which will be used to fill in path template above
IDXS = [(8, 226),  (29, 371), (200, 416), (164, 287), (208, 209), (31, 313), (373, 398),
        (226, 8),  (371, 29), (416, 200), (287, 164), (209, 208), (313, 31), (398, 373)]

# list of molecule sizes which will be used to fill in path template above
MOLLENS = [15, 19, 23, 27, 35]
#######################################

def process_baseline(file_path_idx2, df, id1, id2):
    file_path_idx1 = file_path_idx2.replace(f"/{id1}/{id2}/", f"/{id1}/{id1}/")
    df_idx1 = pd.read_csv(file_path_idx1)
    df_idx1.columns = df_idx1.columns.str.replace("pid2", "pid1")
    df_idx1.columns = df_idx1.columns.str.replace("path2", "path1")
    df = df_idx1.merge(df, on="smi", how="inner", suffixes=("", "_drop"))

    df = df[[col for col in df.columns if not col.endswith("_drop")]]
    return df

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

def single_molecule_validity(smiles):
    if smiles.strip() == "":
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return False
    return True


def validity(list_of_smiles, bs):
    valid_list_smiles = list(filter(single_molecule_validity, list_of_smiles))
    return valid_list_smiles, len(valid_list_smiles) / bs

def unique_lst_of_smiles(list_of_smiles):
    canonical_smiles_lst = list(map(canonicalize, list_of_smiles))
    canonical_smiles_lst = list(
        filter(lambda x: x is not None, canonical_smiles_lst))
    canonical_smiles_lst = list(set(canonical_smiles_lst))
    return canonical_smiles_lst


def uniqueness(valid_smi):
    canonical_smiles_lst = unique_lst_of_smiles(valid_smi)
    return canonical_smiles_lst, len(canonical_smiles_lst) / len(valid_smi) 

def get_qual(unique_smi):
    qed_oracle = Oracle(name = 'QED')
    sa_oracle = Oracle(name = 'SA')

    passed_smi = []
    for i in range(len(unique_smi)):
        smi = unique_smi[i]
        qed_score = qed_oracle(smi)
        sa_score = sa_oracle(smi)
        if qed_score >= 0.6 and sa_score <= 4.0:
            passed_smi.append(smi)
    return passed_smi

def format_single_result(arr):
    arr = np.array(arr)
    mean_ = np.round(np.mean(arr), 3)
    std_ = np.round(np.std(arr), 3)
    return "${:.3f}_{{\pm {:.3f}}}$".format(mean_, std_) 

def format_results(col_names, row_names, arrs):
    cols = "c" * len(col_names)
    preamble = f"\\begin{{table}}[t]\n\\centering\n\\vspace{{-0.9em}}\n\\resizebox{{\linewidth}}{{!}}{{\n\\begin{{tabular}}{{l{'c' * len(col_names)}}}\n\\toprule\n\n"
    conc = "\\bottomrule\n\\end{tabular}\n}\n\\end{table}"
    out_str = preamble + "&"
    for i in range(len(col_names)):
        out_str += f" {col_names[i]} &"
    out_str = out_str[:-1] + '\\\\ \\midrule \n\n'

    for i in range(len(row_names)):
        row_name = row_names[i].replace("_", "\\_")
        curr_row_str = f"{row_name} &"
        for j in range(len(arrs[i])):
            result_arr = arrs[i][j]
            single_val = format_single_result(result_arr)
            curr_row_str += f" {single_val} &"

        curr_row_str = curr_row_str[:-1] + '\\\\ \n\n'
        out_str += curr_row_str
    out_str += conc
    print(out_str)

row_names = []
all_data = []
col_names = [
             "(\\texttt{P$_1$} * \\texttt{P$_2$}) ($\\uparrow$)",
             "min(\\texttt{P$_1$}, \\texttt{P$_2$}) ($\\uparrow$)",
             "\\texttt{P$_1$} ($\\downarrow$)",
             "\\texttt{P$_2$} ($\\downarrow$)",
             "\\texttt{P$_1$} top-1 ($\\downarrow$)",
             "\\texttt{P$_2$} top-1 ($\\downarrow$)",
             "Better than ref. ($\\uparrow$)",
             "Div. ($\\uparrow$)",
             "Val. \\& Uniq. ($\\uparrow$)",
             "Qual. ($\\uparrow$)",
             ]

for model in NAME2PATH:
    print("="*200)

    row_names.append(model)
    val_uniq_arr, div_arr, qed_arr, sa_arr, qual_arr = [], [], [], [], []
    mean_pid1_dock_arr, mean_pid2_dock_arr = [], []
    median_pid1_dock_arr, median_pid2_dock_arr = [], []
    mean_prod_arr, max_prod_arr = [], []
    better_than_ref_arr = []
    top1_pid1_dock_arr, top1_pid2_dock_arr = [], []
    worst_binding_score = []

    path_template = NAME2PATH[model]
    smis = []
    total_mols = 0
    # loop over protein pairs
    for idx1, idx2 in IDXS:
        paths = path_template.replace("IDX1", str(idx1)).replace("IDX2", str(idx2))
        temp_top1_pid1_dock_arr, temp_top1_pid2_dock_arr = [], []
        temp_worst = []
        top1_prod = []

        # loop over number of atoms
        for mollen in MOLLENS:
            path_ = paths.replace("MOLLEN", str(mollen))
            print("loading path_............", path_)
            path_ = glob(path_)
            assert len(path_) == 1
            path_ = path_[0]

            df = pd.read_csv(path_)
            if model == "baseline":
                df = process_baseline(path_, df, idx1, idx2)
            if model == "reference":
                df['pid1_dock'] = df['pid1_refsmi_dock']
                df['pid2_dock'] = df['pid2_refsmi_dock']
                df['smi'] = df['pid1_refsmi']

            smi = df['smi'].tolist()
            bs = int(df.iloc[0]['bs'])
            if model == "reference":
                smis.append(smi[0])
                total_mols+= 1
            else:
                smis.extend(smi)
                total_mols+= bs

            numatoms = int(df.iloc[0]['numatoms'])
            idx1 = int(df.iloc[0]['idx1'])
            idx2 = int(df.iloc[0]['idx2'])
            
            ref_mol1 = float(df.iloc[0]['pid1_refsmi_dock'])
            ref_mol2 = float(df.iloc[0]['pid2_refsmi_dock'])

            valid_smi, val = validity(smi, bs)
            unique_smi, uniq = uniqueness(valid_smi)
            div = diversity(unique_smi)

            val_uniq_arr.append(val * uniq)
            div_arr.append(div)

            df_unique = df.drop_duplicates(subset='smi', keep='first')
            assert len(df_unique) == len(unique_smi)

            df_unique = df_unique[(df_unique['pid1_dock'] < 0) | (df_unique['pid2_dock'] < 0)] # remove cases where both docking scores are positive 
            df_unique['prod'] = df_unique['pid1_dock'] * df_unique['pid2_dock']
            top_1_idx = sorted(df_unique.nlargest(1, 'prod').index)
            top_10_idx = sorted(df_unique.nlargest(10, 'prod').index)
            top_1_df = df_unique.loc[top_1_idx]
            print(df_unique[['pid1_dock', 'pid2_dock',]])
            print(top_1_df[['smi', 'pid1_dock', 'pid2_dock', 'qed', 'sa']])
            print(top_1_df[['path1', 'path2']])
            print("*"*50)

            min_pid1_dock = min_(top_1_df, "pid1_dock")
            min_pid2_dock = min_(top_1_df, "pid2_dock")
            max_score = max(min_pid1_dock, min_pid2_dock)
            
            if min_pid1_dock < 0 or min_pid2_dock < 0: 
                temp_top1_pid1_dock_arr.append(min_pid1_dock)
                temp_top1_pid2_dock_arr.append(min_pid2_dock)
                top1_prod.append(min_pid1_dock*min_pid2_dock)

            max_prod_dock = max_(df_unique, "prod")
            assert max_prod_dock == min_pid1_dock * min_pid2_dock
            max_prod_arr.append(max_prod_dock)
            
            # worst binder
            max_pid1_dock = max_(top_1_df, "pid1_dock")
            max_pid2_dock = max_(top_1_df, "pid2_dock")

            df_unique['max_prop'] = df_unique[['pid1_dock', 'pid2_dock']].max(axis=1)
            worst_binding_score.append(mean(df_unique, 'max_prop'))

            mean_prod_dock = mean(df_unique, "prod")
            mean_prod_arr.append(mean_prod_dock)

            mean_pid1_dock = mean(df_unique, "pid1_dock")
            median_pid1_dock = median(df_unique, "pid1_dock")
            mean_pid1_dock_arr.append(mean_pid1_dock)
            median_pid1_dock_arr.append(median_pid1_dock)

            mean_pid2_dock = mean(df_unique, "pid2_dock")
            median_pid2_dock = median(df_unique, "pid2_dock")
            mean_pid2_dock_arr.append(mean_pid2_dock)
            median_pid2_dock_arr.append(median_pid2_dock)

            mean_qed = mean(df_unique, "qed")
            mean_sa = mean(df_unique, "sa")
            qed_arr.append(mean_qed)
            sa_arr.append(mean_sa)
            
            per_better_than_ref = None
            count_better_than_ref = better_than_ref(df_unique, "pid1_dock", "pid2_dock", ref_mol1, ref_mol2)
            per_better_than_ref = count_better_than_ref / len(unique_smi)
            better_than_ref_arr.append(per_better_than_ref)

            qual_smi = get_qual(unique_smi)
            per_qual_smi = len(qual_smi) / bs
            qual_arr.append(per_qual_smi)
        top1_pid1_dock_arr.append(temp_top1_pid1_dock_arr[np.argmax(top1_prod)])
        top1_pid2_dock_arr.append(temp_top1_pid2_dock_arr[np.argmax(top1_prod)])

    row_data = [mean_prod_arr, worst_binding_score, mean_pid1_dock_arr, mean_pid2_dock_arr, better_than_ref_arr, div_arr, val_uniq_arr, qual_arr]
    all_data.append(row_data)

format_results(col_names, row_names, all_data) 
