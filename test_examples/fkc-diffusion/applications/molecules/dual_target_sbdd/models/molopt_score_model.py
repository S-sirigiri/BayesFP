import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

from models.common import compose_context, ShiftedSoftplus, compose_context_deep_compose
from models.egnn import EGNN
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral
from torch_scatter import scatter
import wandb


def get_refine_net(refine_net_type, config):
    if refine_net_type == 'uni_o2':
        refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup
        )
    elif refine_net_type == 'egnn':
        refine_net = EGNN(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=1,
            k=config.knn,
            cutoff_mode=config.cutoff_mode
        )
    else:
        raise ValueError(refine_net_type)
    return refine_net


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x


def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    if mode == 'none':
        offset = 0.
        pass
    elif mode == 'protein':
        # 25 atoms per molecule
        # protein_pos --> [25*20, 3]
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        # offset --> [20, 3]
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset

def center_pos_compose(protein_pos_1, protein_pos_2, 
                       ligand_pos, 
                       batch_protein_1, batch_protein_2, 
                       batch_ligand, mode='protein'):
    if mode == 'none':
        offset = 0.
        pass
    elif mode == 'protein':
        offset_1 = scatter_mean(protein_pos_1, batch_protein_1, dim=0)
        offset_2 = scatter_mean(protein_pos_2, batch_protein_2, dim=0)
        offset = (offset_1 + offset_2) / 2.

        protein_pos_1 = protein_pos_1 - offset[batch_protein_1]
        protein_pos_2 = protein_pos_2 - offset[batch_protein_2]

        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos_1, protein_pos_2, ligand_pos, offset

def center_pos_inpaint(protein_pos, ligand_pos, frag_pos, batch_protein, batch_ligand, batch_frag, mode='protein'):
    if mode == 'none':
        offset = 0.
        pass
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
        frag_pos = frag_pos - offset[batch_frag]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, frag_pos, offset


# %% categorical diffusion related
def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    # permute_order = (0, -1) + tuple(range(1, len(x.size())))
    # x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return sample_index


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


# %%


# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Model
class ScorePosNet3D(nn.Module):

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, inference_config, args):
        super().__init__()
        self.config = config
        self.inference_config = inference_config
        self.args = args
        self.resample=False
        if "resample" in self.inference_config['sample']:
            self.resample = self.inference_config['sample']['resample']

        # variance schedule
        self.model_mean_type = config.model_mean_type  # ['noise', 'C0']
        self.loss_v_weight = config.loss_v_weight
        # self.v_mode = config.v_mode
        # assert self.v_mode == 'categorical'
        # self.v_net_type = getattr(config, 'v_net_type', 'mlp')
        # self.bond_loss = getattr(config, 'bond_loss', False)
        # self.bond_net_type = getattr(config, 'bond_net_type', 'pre_att')
        # self.loss_bond_weight = getattr(config, 'loss_bond_weight', 0.)
        # self.loss_non_bond_weight = getattr(config, 'loss_non_bond_weight', 0.)

        self.sample_time_method = config.sample_time_method  # ['importance', 'symmetric']
        # self.loss_pos_type = config.loss_pos_type  # ['mse', 'kl']
        # print(f'Loss pos mode {self.loss_pos_type} applied!')
        # print(f'Loss bond net type: {self.bond_net_type} '
        #       f'bond weight: {self.loss_bond_weight} non bond weight: {self.loss_non_bond_weight}')

        if config.beta_schedule == 'cosine':
            alphas = cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s) ** 2
            # print('cosine pos alpha schedule applied!')
            betas = 1. - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
       # print("alphas_cumprod_prev", list((1. - alphas_cumprod_prev) / (1. - alphas_cumprod))); exit()
       # print("posterior_variance", list(posterior_variance)); exit()
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_var = to_torch_const(posterior_variance)
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))

        # atom type diffusion schedule in log space
        if config.v_beta_schedule == 'cosine':
            alphas_v = cosine_beta_schedule(self.num_timesteps, config.v_beta_s)
            # print('cosine v alpha schedule applied!')
        else:
            raise NotImplementedError
        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        # model definition
        self.hidden_dim = config.hidden_dim
        self.num_classes = ligand_atom_feature_dim
        if self.config.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)

        # center pos
        self.center_pos_mode = config.center_pos_mode  # ['none', 'protein']

        # time embedding
        self.time_emb_dim = config.time_emb_dim
        self.time_emb_mode = config.time_emb_mode  # ['simple', 'sin']
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)
            elif self.time_emb_mode == 'sin':
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(self.time_emb_dim),
                    nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
                )
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim, emb_dim)
            else:
                raise NotImplementedError
        else:
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        self.refine_net_type = config.model_type
        self.refine_net = get_refine_net(self.refine_net_type, config)
        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )

    def sigmoid(self, x):
        return 1 / (np.exp(-x) + 1)

    def beta_old(self, t):
        t = 1 - t
        s = 6
        e = -6
        v_s =  self.sigmoid(6)
        v_e =  self.sigmoid(-6)
        x = (t * (e - s) + s)
        b_t = -self.sigmoid(x) + v_e
        b_t /= (v_e - v_s)
        b_t *= 2
        return b_t
        #return np.clip(b_t, 1e-4, 2)

    def beta(self, t):
        beta_s = 10e-4
        beta_e = 2
        x = (12*t - 6)
        b_t = beta_s + (beta_e-beta_s)*self.sigmoid(x)
        return b_t

    def marginal_b_t_old(self, t):
        v_e = self.sigmoid(-6)
        v_s =  self.sigmoid(6)

        #int_b_t = 12 * (v_e - 1)*t + np.log(np.exp(12*t) + np.exp(6))
        int_b_t = 2*((v_e - 1)*t + (1/12) * np.log(np.exp(12*t) + np.exp(6)))
        c = v_e - v_s
        #return int_b_t / (6*c)
        return int_b_t / c

    def marginal_b_t(self, t):
        beta_s = 10e-4
        beta_e = 2
        x = (12*t - 6)
        x = np.log((1+np.exp(x))/(1 + np.exp(-6)))
        return t * beta_s + (beta_e-beta_s)*x*(1/12)

    def mu(self, t):
        return np.sqrt(np.exp(-self.marginal_b_t(t)))

    def sigma(self, t):
        return np.sqrt(1 - np.exp(-self.marginal_b_t(t)))

    def score(self, t, x0, xt):
        mu = self.mu(t)
        sigma = self.sigma(t)
        score = (x0 * mu / sigma**2) - (xt / sigma**2) 
        return score
    #def sigma(self, t):
    #    int_b_t = self.marginal_b_t(t) - self.marginal_b_t(0)
        #return int_b_t
    #    return np.sqrt(int_b_t)
        


    def forward(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand,
                time_step=None, return_all=False, fix_x=False):

        batch_size = batch_protein.max().item() + 1
        init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
        # time embedding
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v

        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)

        h_all, pos_all, batch_all, mask_ligand = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=init_ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x)
        final_pos, final_h = outputs['x'], outputs['h']
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        final_ligand_v = self.v_inference(final_ligand_h)

        preds = {
            'pred_ligand_pos': final_ligand_pos,
            'pred_ligand_v': final_ligand_v,
            'final_h': final_h,
            'final_ligand_h': final_ligand_h
        }
        if return_all:
            final_all_pos, final_all_h = outputs['all_x'], outputs['all_h']
            final_all_ligand_pos = [pos[mask_ligand] for pos in final_all_pos]
            final_all_ligand_v = [self.v_inference(h[mask_ligand]) for h in final_all_h]
            preds.update({
                'layer_pred_ligand_pos': final_all_ligand_pos,
                'layer_pred_ligand_v': final_all_ligand_v
            })
        return preds

    def forward_deep_compose(self, 
                             protein_pos_1, protein_v_1, batch_protein_1, 
                             protein_pos_2, protein_v_2, batch_protein_2, 
                             init_ligand_pos, init_ligand_v, batch_ligand,
                time_step=None, return_all=False, fix_x=False):

        batch_size = batch_protein_1.max().item() + 1

        init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
        # time embedding
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v

        h_protein_1 = self.protein_atom_emb(protein_v_1)
        h_protein_2 = self.protein_atom_emb(protein_v_2)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        if self.config.node_indicator:
            h_protein_1 = torch.cat([h_protein_1, torch.zeros(len(h_protein_1), 1).to(h_protein_1)], -1)
            h_protein_2 = torch.cat([h_protein_2, torch.zeros(len(h_protein_2), 1).to(h_protein_2)], -1)
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein_1)], -1)

        # h_all, pos_all, batch_all, mask_ligand 
        h_all_1, h_all_2, pos_all_1, pos_all_2, batch_all_1, batch_all_2, mask_ligand_1, mask_ligand_2 = \
            compose_context_deep_compose(
                h_protein_1=h_protein_1,
                h_protein_2=h_protein_2,
                h_ligand=init_ligand_h,
                pos_protein_1=protein_pos_1,
                pos_protein_2=protein_pos_2,
                pos_ligand=init_ligand_pos,
                batch_protein_1=batch_protein_1,
                batch_protein_2=batch_protein_2,
                batch_ligand=batch_ligand,
                # h_protein=h_protein,
                # h_ligand=init_ligand_h,
                # pos_protein=protein_pos,
                # pos_ligand=init_ligand_pos,
                # batch_protein=batch_protein,
                # batch_ligand=batch_ligand,
            )
            
        # print(h_all_1.shape)
        # print(h_all_2.shape)
        # [5186, 128]                                                                                                                                                                                                                                | 0/1000 [00:00<?, ?it/s]
        # [4386, 128]

        assert isinstance(self.refine_net, UniTransformerO2TwoUpdateGeneral)
        outputs = self.refine_net.forward_deep_compose(
            h_all_1, h_all_2, 
            pos_all_1, pos_all_2, 
            mask_ligand_1, mask_ligand_2,
            batch_all_1, batch_all_2,
            return_all=return_all, fix_x=fix_x)

        
        # final_pos, final_h = outputs['x'], outputs['h']
        final_pos_1, final_h_1 = outputs['x_1'], outputs['h_1']
        final_pos_2, final_h_2 = outputs['x_2'], outputs['h_2']
        
        # final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        final_ligand_pos, final_ligand_h = final_pos_1[mask_ligand_1], final_h_1[mask_ligand_1]
        

        final_ligand_v = self.v_inference(final_ligand_h)

        preds = {
            'pred_ligand_pos': final_ligand_pos,
            'pred_ligand_v': final_ligand_v,
            # 'final_h': final_h,
            'final_h_1': final_h_1,
            'final_h_2': final_h_2,
            'final_ligand_h': final_ligand_h
        }
        
        # print('return_all:', return_all) False
        # if return_all:
        #     final_all_pos, final_all_h = outputs['all_x'], outputs['all_h']
        #     final_all_ligand_pos = [pos[mask_ligand] for pos in final_all_pos]
        #     final_all_ligand_v = [self.v_inference(h[mask_ligand]) for h in final_all_h]
        #     preds.update({
        #         'layer_pred_ligand_pos': final_all_ligand_pos,
        #         'layer_pred_ligand_v': final_all_ligand_v
        #     })
        return preds


    # atom type diffusion process
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        # q(vt | vt-1)
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        # alpha_t * vt + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_v_pred(self, log_v0, t, batch):
        # compute q(vt | v0)
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)

        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs

    def q_v_sample(self, log_v0, t, batch):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_index = log_sample_categorical(log_qvt_v0)
        log_sample = index_to_log_onehot(sample_index, self.num_classes)
        return sample_index, log_sample

    # atom type generative process
    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(vt-1 | vt, v0) = q(vt | vt-1, x0) * q(vt-1 | x0) / q(vt | x0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0

    def kl_v_prior(self, log_x_start, batch):
        num_graphs = batch.max().item() + 1
        log_qxT_prob = self.q_v_pred(log_x_start, t=[self.num_timesteps - 1] * num_graphs, batch=batch)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))
        kl_prior = categorical_kl(log_qxT_prob, log_half_prob)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt - \
                      extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps
        return pos0_from_e

    def q_pos_posterior(self, x0, xt, t, batch):
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                         extract(self.posterior_mean_ct_coef, t, batch) * xt
        return pos_model_mean

    def kl_pos_prior(self, pos0, batch):
        num_graphs = batch.max().item() + 1
        a_pos = extract(self.alphas_cumprod, [self.num_timesteps - 1] * num_graphs, batch)  # (num_ligand_atoms, 1)
        pos_model_mean = a_pos.sqrt() * pos0
        pos_log_variance = torch.log((1.0 - a_pos).sqrt())
        kl_prior = normal_kl(torch.zeros_like(pos_model_mean), torch.zeros_like(pos_log_variance),
                             pos_model_mean, pos_log_variance)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def sample_time(self, num_graphs, device, method):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(num_graphs, device, method='symmetric')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            time_step = torch.multinomial(pt_all, num_samples=num_graphs, replacement=True)
            pt = pt_all.gather(dim=0, index=time_step)
            return time_step, pt

        elif method == 'symmetric':
            time_step = torch.randint(
                0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
            time_step = torch.cat(
                [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt

        else:
            raise ValueError

    def compute_pos_Lt(self, pos_model_mean, x0, xt, t, batch):
        # fixed pos variance
        pos_log_variance = extract(self.posterior_logvar, t, batch)
        pos_true_mean = self.q_pos_posterior(x0=x0, xt=xt, t=t, batch=batch)
        kl_pos = normal_kl(pos_true_mean, pos_log_variance, pos_model_mean, pos_log_variance)
        kl_pos = kl_pos / np.log(2.)

        decoder_nll_pos = -log_normal(x0, means=pos_model_mean, log_scales=0.5 * pos_log_variance)
        assert kl_pos.shape == decoder_nll_pos.shape
        mask = (t == 0).float()[batch]
        loss_pos = scatter_mean(mask * decoder_nll_pos + (1. - mask) * kl_pos, batch, dim=0)
        return loss_pos

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
        return loss_v

    def get_diffusion_loss(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step=None
    ):
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)

        # 1. sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )

        # 2. perturb pos and v
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std
        # Vt = a * V0 + (1-a) / K
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)

        # 3. forward-pass NN, feed perturbed pos and v, output noise
        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,

            init_ligand_pos=ligand_pos_perturbed,
            init_ligand_v=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            time_step=time_step
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed
        # atom position
        if self.model_mean_type == 'noise':
            pos0_from_e = self._predict_x0_from_eps(
                xt=ligand_pos_perturbed, eps=pred_pos_noise, t=time_step, batch=batch_ligand)
            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        elif self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        else:
            raise ValueError

        # atom pos loss
        if self.model_mean_type == 'C0':
            target, pred = ligand_pos, pred_ligand_pos
        elif self.model_mean_type == 'noise':
            target, pred = pos_noise, pred_pos_noise
        else:
            raise ValueError
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0)
        loss_pos = torch.mean(loss_pos)

        # atom type loss
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        loss_v = torch.mean(kl_v)
        loss = loss_pos + loss_v * self.loss_v_weight

        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss': loss,
            'x0': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1)
        }

    @torch.no_grad()
    def likelihood_estimation(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step
    ):
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein')
        assert (time_step == self.num_timesteps).all() or (time_step < self.num_timesteps).all()
        if (time_step == self.num_timesteps).all():
            kl_pos_prior = self.kl_pos_prior(ligand_pos, batch_ligand)
            log_ligand_v0 = index_to_log_onehot(batch_ligand, self.num_classes)
            kl_v_prior = self.kl_v_prior(log_ligand_v0, batch_ligand)
            return kl_pos_prior, kl_v_prior

        # perturb pos and v
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std
        # Vt = a * V0 + (1-a) / K
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)

        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,

            init_ligand_pos=ligand_pos_perturbed,
            init_ligand_v=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            time_step=time_step
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']
        if self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        else:
            raise ValueError

        # atom type
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)

        # t = [T-1, ... , 0]
        kl_pos = self.compute_pos_Lt(pos_model_mean=pos_model_mean, x0=ligand_pos,
                                     xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        return kl_pos, kl_v

    @torch.no_grad()
    def fetch_embedding(self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand):
        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,

            init_ligand_pos=ligand_pos,
            init_ligand_v=ligand_v,
            batch_ligand=batch_ligand,
            fix_x=True
        )
        return preds

    @torch.no_grad()
    def sample_diffusion(self, protein_pos, protein_v, batch_protein,
                         init_ligand_pos, init_ligand_v, batch_ligand,
                         num_steps=None, center_pos_mode=None, pos_only=False, 
                        reverse_method="SDE", inv_temp=1):
        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)

        pos_traj, v_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        # time sequence
        t_ = 1
        dt = 1/self.num_timesteps
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        for j, i in tqdm(enumerate(time_seq), desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)
            preds = self(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,

                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v,
                batch_ligand=batch_ligand,
                time_step=t
            )
            # Compute posterior mean and variance
            if self.model_mean_type == 'noise':
                pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                v0_from_e = preds['pred_ligand_v']
            elif self.model_mean_type == 'C0':
                pos0_from_e = preds['pred_ligand_pos']
                v0_from_e = preds['pred_ligand_v']
            else:
                raise ValueError

            if reverse_method == "baseline":
                pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
                pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
                # no noise when t == 0
                nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                    ligand_pos)
                ligand_pos = ligand_pos_next
            elif reverse_method == "SDE":
                sigma = torch.tensor(self.sigma(t_))
                score = self.score(t_, pos0_from_e, ligand_pos)
                beta_t = torch.tensor(self.beta(t_))
                dlog_alphadt = -(1/2) * beta_t
                g = torch.sqrt(beta_t)
                f_x = dlog_alphadt*ligand_pos

                nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
                eps = torch.randn_like(score)

                dx = dt*(-f_x + inv_temp*g**2*score) + g*np.sqrt(dt)*eps*nonzero_mask
                ligand_pos += dx
                t_ -= dt
            if not pos_only:
                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
                log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next = log_sample_categorical(log_model_prob)

                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
                vt_pred_traj.append(log_model_prob.clone().cpu())
                ligand_v = ligand_v_next

            ori_ligand_pos = ligand_pos + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())

        ligand_pos = ligand_pos + offset[batch_ligand]
        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj
        }

    @torch.no_grad()
    def sample_cat_sys(self, bs, logits, device, T=1):
        bs = int(bs.item())
        u = torch.rand((1,))
        u = (u + 1/bs*torch.arange(bs)) % 1
        u = u.to(device)
        s = F.softmax(logits/T)
        bins = torch.cumsum(s, dim=0)
        ids = torch.bucketize(u, bins)
        return ids

    @torch.no_grad()
    def compose_sample_diffusion(self, 
                                 protein_pos_1, protein_v_1, batch_protein_1,
                                 protein_pos_2, protein_v_2, batch_protein_2,
                         init_ligand_pos, init_ligand_v, batch_ligand,
                         num_steps=None, center_pos_mode=None, pos_only=False,
                         reverse_method="baseline", inv_temp=1):

        if num_steps is None:
            num_steps = self.num_timesteps

        num_graphs_1 = batch_protein_1.max().item() + 1
        num_graphs_2 = batch_protein_2.max().item() + 1
        assert num_graphs_1 == num_graphs_2
        num_graphs = num_graphs_1

        protein_pos_1, protein_pos_2, init_ligand_pos, offset = center_pos_compose(
            protein_pos_1, protein_pos_2, 
            init_ligand_pos, 
            batch_protein_1, batch_protein_2, 
            batch_ligand, mode=center_pos_mode)
        id_traj, dw_traj = [], []
        pos_traj, v_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        if reverse_method == "SDE":
            ligand_pos /= np.sqrt(2 * inv_temp)

        
        # time sequence
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        t_ = 1
        dt = 1/self.num_timesteps
        bs = torch.max(batch_ligand) +1
        device=protein_pos_1.device
        w_gen = torch.zeros((bs,2,1), device=device)
        for j, i in tqdm(enumerate(time_seq), desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos_1.device)
            preds_1 = self(
                protein_pos=protein_pos_1,
                protein_v=protein_v_1,
                batch_protein=batch_protein_1,

                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v,
                batch_ligand=batch_ligand,
                time_step=t
            )
            preds_2 = self(
                protein_pos=protein_pos_2,
                protein_v=protein_v_2,
                batch_protein=batch_protein_2,

                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v,
                batch_ligand=batch_ligand,
                time_step=t
            )
            # Compute posterior mean and variance
            if self.model_mean_type == 'noise':
                raise ValueError
            elif self.model_mean_type == 'C0':
                pos0_from_e_1 = preds_1['pred_ligand_pos']
                v0_from_e_1 = preds_1['pred_ligand_v']
                pos0_from_e_2 = preds_2['pred_ligand_pos']
                v0_from_e_2 = preds_2['pred_ligand_v']
            else:
                raise ValueError

            if reverse_method == "baseline":
                pos0_from_e = (pos0_from_e_1 + pos0_from_e_2) / 2.
                pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)

                pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
                # no noise when t == 0
                nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                    ligand_pos)
                ligand_pos = ligand_pos_next
            elif reverse_method == "SDE":
                sigma = torch.tensor(self.sigma(t_))

                score_1 = self.score(t_, pos0_from_e_1, ligand_pos)
                score_2 = self.score(t_, pos0_from_e_2, ligand_pos)
                score = score_1 + score_2

                beta_t = torch.tensor(self.beta(t_))
                dlog_alphadt = -(1/2) * beta_t
                g = torch.sqrt(beta_t)
                f_x = dlog_alphadt*ligand_pos

                wandb.log({"dt": dt, "t": t_, "beta_t": beta_t, "sigma": sigma, "g": g, "dlog_alphadt": dlog_alphadt}, step=j)
                nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)

                eps = torch.randn_like(score)

                if self.args.tempered_score:
                    dx = dt*(-f_x + 0.5*(inv_temp+1)*g**2*score) + (g/np.sqrt(inv_temp))*np.sqrt(dt)*eps*nonzero_mask
                else:
                    dx = dt*(-f_x + inv_temp*g**2*score) + g*np.sqrt(dt)*eps*nonzero_mask
                ligand_pos += dx
                t_ -= dt
                ndim = ligand_pos.shape[1] 

                a = ndim*dlog_alphadt
                a *= (2*inv_temp - 1)

                b = (inv_temp*g**2*(score_1*score_2)).sum((1), keepdims=True) + (inv_temp*(inv_temp-1)*0.5*g**2*(score_1 + score_2)**2).sum((1), keepdims=True)

                dw = a+b
                dw *= dt

                dw_over_mols = scatter(dw.squeeze(), batch_ligand).unsqueeze(1)
                dw_std = torch.std(dw_over_mols).item()
                wandb.log({"dw_std": dw_std, "dw_mean": torch.mean(dw_over_mols).item()}, step=j)
                w_gen[:,1,:] = w_gen[:, 0, :].detach().clone() + dw_over_mols

                resample = self.resample
                if t_  < (1 - self.args.resample_thresh):
                    resample=False
                new_ids = np.arange(bs.item())
                if resample:

                    new_ids = self.sample_cat_sys(bs, w_gen[:,1,0], device, T=1).detach().cpu().numpy()
                    new_ids = sorted(new_ids)
                    print("ids_to_sample", new_ids)
                    unique_ids = set(new_ids)

                    wandb.log({"num_unique_ids": len(unique_ids)}, step=j)
                    num_atoms_per_mol = int((ligand_pos.shape[0] // bs).item())
                    id_ranges = [(int(i) * num_atoms_per_mol, (int(i) + 1) * num_atoms_per_mol) for i in new_ids]
                    new_ligand_pos = []

                    for id_range in id_ranges:
                        start, end = id_range
                        new_ligand_pos.append(ligand_pos[start:end].detach().clone())
                    ligand_pos = torch.cat(new_ligand_pos, dim=0)
                    w_gen[:,1,:] = 0
                w_gen[:,0,:] = w_gen[:,1,:].detach().clone()
                wandb.log({"w_std": torch.std(w_gen[:,0,:]).item(), "w_mean": torch.mean(w_gen[:,0,:]).item()}, step=j)
                id_traj.append(new_ids)
                dw_traj.append(dw_over_mols.clone().cpu())


            if not pos_only:
                # log_ligand_v_recon = F.log_softmax(v0_from_e_1, dim=-1) + F.log_softmax(v0_from_e_2, dim=-1) - torch.log(2.0 * torch.ones_like(v0_from_e_1))
                logp1 = F.log_softmax(v0_from_e_1, dim=-1)
                logp2 = F.log_softmax(v0_from_e_2, dim=-1)
                logZ = torch.logsumexp(logp1 + logp2, dim=-1, keepdim=True)
                log_ligand_v_recon = logp1 + logp2 - logZ
                # log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)

                log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next = log_sample_categorical(log_model_prob)

                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
                vt_pred_traj.append(log_model_prob.clone().cpu())
                ligand_v = ligand_v_next

            ori_ligand_pos = ligand_pos + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())

        ligand_pos = ligand_pos + offset[batch_ligand]
        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj,
            'dw_traj': dw_traj,
            'id_traj': id_traj,
        }

    @torch.no_grad()
    def deep_compose_sample_diffusion(self, 
                                 protein_pos_1, protein_v_1, batch_protein_1,
                                 protein_pos_2, protein_v_2, batch_protein_2,
                         init_ligand_pos, init_ligand_v, batch_ligand,
                         num_steps=None, center_pos_mode=None, pos_only=False):

        if num_steps is None:
            num_steps = self.num_timesteps

        num_graphs_1 = batch_protein_1.max().item() + 1
        num_graphs_2 = batch_protein_2.max().item() + 1
        assert num_graphs_1 == num_graphs_2
        num_graphs = num_graphs_1

        # protein_pos, init_ligand_pos, offset = center_pos(
        #     protein_pos, init_ligand_pos, batch_protein_1, batch_ligand, mode=center_pos_mode)
        protein_pos_1, protein_pos_2, init_ligand_pos, offset = center_pos_compose(
            protein_pos_1, protein_pos_2, 
            init_ligand_pos, 
            batch_protein_1, batch_protein_2, 
            batch_ligand, mode=center_pos_mode)
        # protein_pos_1, init_ligand_pos_1, offset_1 = center_pos(
        #     protein_pos_1, init_ligand_pos, batch_protein_1, batch_ligand, mode=center_pos_mode)
        # protein_pos_2, init_ligand_pos_2, offset_2 = center_pos(
        #     protein_pos_2, init_ligand_pos, batch_protein_2, batch_ligand, mode=center_pos_mode)

        pos_traj, v_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        # ligand_pos_1 = init_ligand_pos_1
        # ligand_pos_2 = init_ligand_pos_2
        # ligand_v = init_ligand_v
        
        # time sequence
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos_1.device)
            assert isinstance(self, ScorePosNet3D)
            preds = self.forward_deep_compose(
                protein_pos_1=protein_pos_1,
                protein_v_1=protein_v_1,
                batch_protein_1=batch_protein_1,

                protein_pos_2=protein_pos_2,
                protein_v_2=protein_v_2,
                batch_protein_2=batch_protein_2,

                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v,
                batch_ligand=batch_ligand,
                time_step=t
            )
            
            # Compute posterior mean and variance
            if self.model_mean_type == 'noise':
                raise ValueError
                # pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                # pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                # v0_from_e = preds['pred_ligand_v']
            elif self.model_mean_type == 'C0':
                pos0_from_e = preds['pred_ligand_pos']
                v0_from_e = preds['pred_ligand_v']
            else:
                raise ValueError
            
            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)

            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
            # no noise when t == 0
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                ligand_pos)
            ligand_pos = ligand_pos_next

            if not pos_only:
                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)

                log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next = log_sample_categorical(log_model_prob)

                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
                vt_pred_traj.append(log_model_prob.clone().cpu())
                ligand_v = ligand_v_next

            ori_ligand_pos = ligand_pos + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())

        ligand_pos = ligand_pos + offset[batch_ligand]
        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj
        }


    @torch.no_grad()
    def sample_diffusion_inpaint(self, protein_pos, protein_v, batch_protein,
                                init_ligand_pos, init_ligand_v, batch_ligand, 
                                frag_pos, frag_v, batch_frag, ref_mask,
                                num_steps=None, center_pos_mode=None, pos_only=False):

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        protein_pos, init_ligand_pos, frag_pos, offset = center_pos_inpaint(
            protein_pos, init_ligand_pos, frag_pos, batch_protein, batch_ligand, batch_frag, mode=center_pos_mode)

        pos_traj, v_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        # time sequence
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)
            preds = self(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,

                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v,
                batch_ligand=batch_ligand,
                time_step=t
            )
            # Compute posterior mean and variance
            if self.model_mean_type == 'noise':
                pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                v0_from_e = preds['pred_ligand_v']
            elif self.model_mean_type == 'C0':
                pos0_from_e = preds['pred_ligand_pos']
                v0_from_e = preds['pred_ligand_v']
            else:
                raise ValueError
            
            pos0_from_e[ref_mask] = frag_pos

            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
            # no noise when t == 0
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                ligand_pos)
            ligand_pos = ligand_pos_next

            

            if not pos_only:
                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
                log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)
                
                log_frag_v = index_to_log_onehot(frag_v, self.num_classes)
                
                
                log_ligand_v_recon[ref_mask] = log_frag_v

                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next = log_sample_categorical(log_model_prob)

                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
                vt_pred_traj.append(log_model_prob.clone().cpu())
                ligand_v = ligand_v_next

            ori_ligand_pos = ligand_pos + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())

        ligand_pos = ligand_pos + offset[batch_ligand]
        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj
        }


def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)
