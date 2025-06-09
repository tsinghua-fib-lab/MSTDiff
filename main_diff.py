import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_models import TF_diff, Conv1d_with_init
import os
from scipy.signal import stft, istft
from matplotlib import pyplot as plt
import pywt
from pytorch_wavelets import DWT1D, IDWT1D
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import coo_matrix
import math
from tqdm import tqdm
import h5py
import random

os.environ['CUDA_LAUNCH_BLOCKING'] = '5,6'


class Diff_base(nn.Module):
    def __init__(self, config, device, Qt_all, Qtbar_all, model_revin=None, model_vqvae=None):
        super().__init__()
        self.device = device
        self.is_unconditional = True
        config_diff = config["diffusion"]
        if "vqvae_config" in config:
            config_vqvae = config["vqvae_config"]
        else:
            config_vqvae = None

        self.NUM_TrajID = Qt_all.shape[1]
        self.Qt_all, self.Qtbar_all, = Qt_all, Qtbar_all

        input_dim = 1 if self.is_unconditional == True else 2
        self.level = 3
        self.wavelet = "db1"
        self.split_list = compute_wtlen(siglen=336, level=self.level, wavelet=self.wavelet)
        self.diffmodel_tr = TF_diff(config_diff, config_vqvae, device, self.level, self.split_list, model_revin,
                                    model_vqvae, mode="traf", num_traj=self.NUM_TrajID, inputdim=input_dim)
        self.diffmodel_tj = TF_diff(config_diff, config_vqvae, device, self.level, self.split_list, model_revin,
                                    model_vqvae, mode="traj", num_traj=self.NUM_TrajID, inputdim=input_dim)

        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta_tr = np.linspace(
                config_diff["beta_tr_start"] ** 0.5, config_diff["beta_tr_end"] ** 0.5, self.num_steps
            ) ** 2

            self.alpha_hat_tj = np.exp(np.linspace(np.log(config_diff["alpha_tj_start"]),
                                                   np.log(config_diff["alpha_tj_end"]),
                                                   self.num_steps
                                                   ))

        elif config_diff["schedule"] == "linear":
            self.beta_tr = np.linspace(
                config_diff["beta_tr_start"], config_diff["beta_tr_end"], self.num_steps
            )
            self.alpha_hat_tj = np.linspace(
                config_diff["alpha_tj_start"], config_diff["alpha_tj_end"], self.num_steps
            )

        self.alpha_hat_tr = 1 - self.beta_tr
        self.alpha_tr = np.cumprod(self.alpha_hat_tr)
        self.alpha_torch_tr = torch.tensor(self.alpha_tr).float().to(self.device).unsqueeze(1)  # (B,1)

        self.alpha_tj = np.cumsum(self.alpha_hat_tj)  # s<=t 求和
        self.alpha_torch_tj = torch.tensor(self.alpha_tj).float().to(self.device).unsqueeze(1).unsqueeze(1)
        self.alpha_hat_torch_tj = torch.tensor(self.alpha_hat_tj).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def trans_t_prob_batch(self, x0, t):
        B, L = x0.shape
        x0_flat = x0.view(-1)
        if t.ndim == 0 or t.numel() == 1:
            t_flat = torch.full_like(x0_flat, t.item())
        else:
            t_flat = t.unsqueeze(1).expand(-1, L).reshape(-1)
        N = self.Qtbar_all.shape[1]
        probs_flat = self.Qtbar_all[t_flat, x0_flat, :]
        return probs_flat.view(B, L, N)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def calc_loss_valid(
            self, target_traf, target_traj, is_train, fast_valid=True
    ):
        loss_sum = 0
        loss_tf_sum = 0
        loss_tj_sum = 0
        if fast_valid:
            t = random.randint(0, self.num_steps - 1)
            loss, loss_tf, loss_tj = self.calc_loss(
                target_traf, target_traj, is_train, set_t=t
            )
            loss_sum += loss.detach()
            loss_tf_sum += loss_tf.detach()
            loss_tj_sum += loss_tj.detach()
            return loss_sum, loss_tf_sum, loss_tj_sum
        else:
            for t in range(self.num_steps):
                loss, loss_tf, loss_tj = self.calc_loss(
                    target_traf, target_traj, is_train, set_t=t
                )
                loss_sum += loss.detach()
                loss_tf_sum += loss_tf.detach()
                loss_tj_sum += loss_tj.detach()
            return loss_sum / self.num_steps, loss_tf_sum / self.num_steps, loss_tj_sum / self.num_steps

    def calc_loss(
            self, target_traf, target_traj, is_train, set_t=-1
    ):
        B, L = target_traf.shape
        target_wv = DWT_torch(target_traf, self.split_list, self.level, self.wavelet)
        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        # Diffusion of traffic
        current_alpha_tr = self.alpha_torch_tr[t]
        noise = torch.randn(B, L, device=self.device)
        noisy_tr = (current_alpha_tr ** 0.5) * target_wv + (1.0 - current_alpha_tr) ** 0.5 * noise
        # Diffusion of traffic
        noisy_tj = self.trans_t_prob_batch(target_traj, t)
        # Input to model
        noisy_data = torch.cat([noisy_tr.unsqueeze(-1), noisy_tj], dim=2)
        total_input = self.get_input_to_diffmodel(noisy_data)
        predicted_tr = self.diffmodel_tr(total_input, t)  # (B,L)
        predicted_tj = self.diffmodel_tj(total_input, t)  # (B,L,N)

        residual_tr = (noise - predicted_tr) ** 2
        num_pixel = B * L
        loss_tr = residual_tr.sum() / (num_pixel if num_pixel > 0 else 1)
        loss_tj_kl = self.loss_kl(x0=target_traj, logits_tj=predicted_tj, t=t)
        a1, a2 = 1, 50
        loss_tj = a2 * loss_tj_kl
        loss = a1 * loss_tr + loss_tj
        return loss, loss_tr, loss_tj

    def q_posterior_from_x0(self, x0, t):
        """
            Qt_all: (T, N, N) transition matrix Qt
            Qtbar_all: (T, N, N) cumulative transition Qbart
        """
        B, L = x0.shape
        N = self.NUM_TrajID
        eps = 1e-10

        if isinstance(t, int):  # int when test
            t = torch.full((B,), t, dtype=torch.long, device=x0.device)

        x0_flat = x0.view(-1)
        t_flat = t.unsqueeze(1).expand(-1, L).reshape(-1)
        t_clamp = torch.clamp(t_flat - 1, min=0)

        q_xt = self.Qtbar_all[t_flat, x0_flat].view(B, L, N)
        q_xtm1 = self.Qtbar_all[t_clamp, x0_flat].view(B, L, N)
        Qt = self.Qt_all[t_flat].view(B, L, N, N)

        numerator = torch.einsum('blij,blj->bli', Qt.transpose(2, 3), q_xtm1)
        q_posterior = numerator / (numerator.sum(dim=-1, keepdim=True) + eps)
        return q_posterior

    def loss_kl(self, x0, logits_tj, t):
        eps = 1e-10
        q_post = self.q_posterior_from_x0(x0=x0, t=t)
        p_x0 = F.softmax(logits_tj, dim=-1)
        p_xtm1 = (q_post * p_x0) / (q_post * p_x0).sum(-1, keepdim=True)
        kl = q_post * (torch.log(q_post + eps) - torch.log(p_xtm1 + eps))
        kl_loss = kl.sum(dim=-1).mean()
        return kl_loss

    def get_input_to_diffmodel(self, noisy_data):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)
        else:
            pass
        return total_input

    def sample_point_from_prob(self, probs):
        B, L, N = probs.shape
        probs_flat = probs.view(-1, N)
        sampled_indices = torch.multinomial(probs_flat, num_samples=1).view(B, L).to(self.device)
        return sampled_indices  # (B, L)

    def impute(self, target_traf, target_traj, n_samples):  # utils中定义，n_samples = 1
        # (B, L,), (B, L, T)
        B, L = target_traf.shape
        dim_target = 2
        N = self.NUM_TrajID
        imputed_samples = torch.zeros(B, n_samples, L, dim_target).to(self.device)

        for i in range(n_samples):
            max_list = []
            min_list = []
            current_sample_tr = torch.randn(B, L, device=self.device)  # (B,L) 本身在小波域

            probs = torch.ones(N, device=self.device) / N  # (N,)
            probs = probs.view(1, 1, N).expand(B, L, N)  # (B, L, N)
            # (B,L) 初始的位置采样
            current_sample_tj = self.sample_point_from_prob(probs)
            current_sample_tj_prob = probs
            for t in tqdm(range(self.num_steps - 1, -1, -1), total=self.num_steps):
                if self.is_unconditional == True:
                    diff_input = torch.cat([current_sample_tr.unsqueeze(-1), current_sample_tj_prob], dim=2)
                    diff_input = diff_input.unsqueeze(1)  # (1,1,336,33)
                    # print("diff input", diff_input.shape)

                max_list.append(torch.max(diff_input).cpu().numpy())
                min_list.append(torch.min(diff_input).cpu().numpy())

                predicted_tr = self.diffmodel_tr(diff_input, torch.tensor([t])).to(self.device)  # (1, 336, 33)
                predicted_tj = self.diffmodel_tj(diff_input, torch.tensor([t])).to(self.device)  # (1, 336, 33)

                # ---------------1. 流量部分---------------
                coeff1 = 1 / self.alpha_hat_tr[t] ** 0.5
                coeff2 = (1 - self.alpha_hat_tr[t]) / (1 - self.alpha_tr[t]) ** 0.5
                current_sample_tr = coeff1 * (current_sample_tr - coeff2 * predicted_tr)  # mean
                if t > 0:
                    noise_tr = torch.randn_like(current_sample_tr).to(self.device)  # (B,H,W)
                    sigma = (
                                    (1.0 - self.alpha_tr[t - 1]) / (1.0 - self.alpha_tr[t]) * self.beta_tr[t]
                            ) ** 0.5  # 标准差

                    current_sample_tr += sigma * noise_tr

                # ---------------2. 轨迹部分---------------
                # q(x_{t-1} | x_t) = sum_x0 p(x0|xt) * q(x_{t-1}|xt,x0)
                logits_x0 = predicted_tj  # (B, L, N)
                probs = self.prob_xtm1_from_samplesX0(logits_x0, current_sample_tj, t, SAMPLE_X0_NUM=10)  # (B,L,N)
                current_sample_tj = self.sample_point_from_prob(probs)  # (B, L)
                current_sample_tj_prob = probs

            sample_tr = iDWT_torch(current_sample_tr, self.split_list, self.level, self.wavelet)  # (B,336)
            # sample_tr = current_sample_tr
            imputed_samples[:, i, :, :] = torch.cat([sample_tr.unsqueeze(-1), current_sample_tj.unsqueeze(-1)], dim=2)
            plt.figure(figsize=(10, 6))
            plt.plot(max_list, color='red')
            plt.plot(min_list, color='orange')
            plt.savefig("./GenData/max_min.png")
            plt.close()
        # print("imputed samples", imputed_samples.shape)  # [1, 1, 336, 33]
        return imputed_samples  # # (1,1,336,33)

    def prob_xtm1_from_samplesX0(self, logits_x0, current_xt, t, SAMPLE_X0_NUM=1):
        '''
        Returns: p(xt-1|xt), (B,L,N)
        '''
        B, L, N = logits_x0.shape
        probs_x0 = F.softmax(logits_x0, dim=-1)
        samples = torch.multinomial(probs_x0.view(-1, N), num_samples=SAMPLE_X0_NUM, replacement=True)
        x0_ids_k = samples.view(B, L, SAMPLE_X0_NUM).permute(2, 0, 1)  # (K, B, L)

        sampled_probs = torch.gather(probs_x0.unsqueeze(0).expand(SAMPLE_X0_NUM, -1, -1, -1),
                                     dim=3, index=x0_ids_k.unsqueeze(-1))  # (K,B,L,1)
        sampled_probs = sampled_probs.squeeze(-1)  # (K,B,L)

        post_xtm1 = torch.zeros(B, L, N, device=self.device)

        for k in range(SAMPLE_X0_NUM):
            x0_k = x0_ids_k[k]  # (B,L)
            q_post = self.q_posterior_from_x0(x0=x0_k, t=t)
            post_xtm1 += q_post * sampled_probs[k].unsqueeze(-1)

        post_xtm1 = post_xtm1 / (post_xtm1.sum(dim=-1, keepdim=True) + 1e-8)
        return post_xtm1  # (B,L,N)

    def forward(self, batch, is_train=1):
        (
            target_traf,
            target_traj,
            observed_tp
        ) = self.process_data(batch)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(target_traf, target_traj, is_train)

    def evaluate(self, batch, n_samples):
        (
            target_traf,
            target_traj,
            observed_tp
        ) = self.process_data(batch)

        with torch.no_grad():
            samples = self.impute(target_traf, target_traj, n_samples)
        return samples, target_traf, target_traj

    def DWT_0221(self, signal, sampling_period=0.5, wavelet='morl'):
        B, L = signal.shape
        signal = signal  # (1,336)
        signal = signal.cpu().numpy()
        scales = [window / sampling_period for window in self.time_windows]
        coeffs = []
        for i in range(B):
            cof_per_user = []
            for idx, scale in enumerate(scales):
                sig = signal[i]
                coeff, _ = pywt.cwt(sig, [scale], wavelet, sampling_period)
                cof_per_user.append(np.abs(coeff[0]))
            coeffs.append(cof_per_user)
        coeffs = np.array(coeffs)
        coeffs = torch.tensor(coeffs, device=self.device).permute(0, 2, 1)  # (1,336,5)
        return coeffs


class main_CSDI(Diff_base):
    def __init__(self, config, device, Qt_all, Qtbar_all, model_revin, model_vqvae):
        super(main_CSDI, self).__init__(config, device, Qt_all, Qtbar_all, model_revin, model_vqvae)

    def process_data(self, batch):
        target_traf = batch["utf_norm"].to(self.device).float()  # (B, 336)
        target_traj = batch["traj_cid"].to(self.device).int()  # (B, 336)，以位置id作为扩散项
        observed_tp = batch["timepoints"].to(self.device).float()
        return (
            target_traf,
            target_traj,
            observed_tp
        )


def DWT(signal, level, wavelet, device, sampling_period=0.5):
    B, L = signal.shape
    signal = signal.cpu().numpy()
    coeffs_list = []
    for i in range(B):
        sig = signal[i]
        c = pywt.wavedec(sig, wavelet, level=level, mode="constant")
        c_tensors = [torch.tensor(ci, dtype=torch.float32, device=device) for ci in c]
        coeffs_list.append(torch.cat(c_tensors, dim=-1))
    coeffs = torch.stack(coeffs_list, dim=0)
    return coeffs


def DWT_torch(signal, split_list, level=3, wavelet='db1'):
    B, L = signal.shape
    x = signal.unsqueeze(1)  # (B, 1, L)
    dwt = DWT1D(wave=wavelet, J=level).to(signal.device)
    cA, cDs = dwt(x)
    coeff_list = [cA] + [cDs[2]] + [cDs[1]] + [cDs[0]]  # 42， 84, 168
    coeff_flat = [c.squeeze(1) for c in coeff_list]
    coeffs_fin = torch.cat(coeff_flat, dim=-1)  # (B, 336)
    return coeffs_fin  # (B,L)


def iDWT_torch(coeffs_cat, split_list, level=3, wavelet='db1'):
    B, L = coeffs_cat.shape
    coeffs_ori = coeffs_cat
    cA3, cD3, cD2, cD1 = torch.split(coeffs_ori, split_list, dim=-1)
    cA3 = cA3.unsqueeze(1)
    cD3 = cD3.unsqueeze(1)
    cD2 = cD2.unsqueeze(1)
    cD1 = cD1.unsqueeze(1)
    idwt = IDWT1D(wave=wavelet, mode='zero').to(coeffs_cat.device)
    return idwt((cA3, [cD1, cD2, cD3])).squeeze(1)  # (B, L)


def compute_wtlen(siglen, level, wavelet):
    wvlen = pywt.Wavelet(wavelet).dec_len
    signal_length = siglen
    len_list = []
    for i in range(0, level):
        signal_length = pywt.dwt_coeff_len(signal_length, wvlen, "constant")
        len_list.append(signal_length)
        if i == level - 1:
            len_list.append(signal_length)
    len_list.reverse()
    print("Length of dwt list: ", len_list)
    return len_list
