import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '4,5,6'
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer
from matplotlib import pyplot as plt
import numpy as np
from torch.autograd import Variable


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def get_linear_trans(heads=8, layers=1, channels=64, localheads=0, localwindow=0):
    return LinearAttentionTransformer(
        dim=channels,
        depth=layers,
        heads=heads,
        max_seq_len=256,
        n_local_attn_heads=0,
        local_attn_window_size=0,
    )


def Conv3d_with_init(inputdim, out_channels, kernel_size_H, kernel_size_W):
    layer = nn.Conv3d(in_channels=inputdim, out_channels=out_channels, kernel_size=(3, kernel_size_H, kernel_size_W),
                      padding=(0, 1, 1))
    nn.init.kaiming_normal_(layer.weight)
    return layer


def Conv2d_with_init(in_channels, out_channels, kernel_size_H, kernel_size_W, stride=1, padding=(1, 1)):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size_H, kernel_size_W), stride=stride,
                      padding=padding)
    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    padding = (kernel_size - 1) // 2
    layer = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=1)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)  # 128

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class TF_diff(nn.Module):
    def __init__(self, config_diff, config_vqvae, device, level, split_list,model_revin, model_vqvae, mode, num_traj, inputdim=1, hidden_traj=32, emb_dim=32):
        super().__init__()
        self.device = device
        self.channels = config_diff["channels"]
        self.level = level
        self.split_list = split_list
        self.mode = mode
        self.config_vqvae = config_vqvae
        if self.config_vqvae:
            self.revIN = model_revin
            self.vqvae = model_vqvae
            self.CF = config_vqvae["compression_factor"]
        self.num_traj = num_traj
        self.hidden_traj = hidden_traj
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config_diff["num_steps"],
            embedding_dim=config_diff["diffusion_embedding_dim"],
        )

        self.input_mlp_tj = nn.Sequential(
            nn.Linear(self.num_traj, 64),
            nn.GELU(),
            nn.Linear(64, self.hidden_traj)
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 3)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 3)
        self.output_projection2 = Conv1d_with_init(self.channels, inputdim, 3)
        self.wv_projection1 = Conv1d_with_init(self.channels, self.channels, 3)
        self.wv_projection2 = Conv1d_with_init(self.channels, inputdim, 3)

        self.pos_embedding = PositionEmbedding(max(self.split_list), device)
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    channels=self.channels,
                    diffusion_embedding_dim=config_diff["diffusion_embedding_dim"],
                    att_emb_dim=32,
                    split_len=self.split_list[i],
                    nheads=config_diff["nheads"],
                    device=self.device,
                    is_linear=config_diff["is_linear"],
                )
                for i in range(config_diff["layers"])
            ]
        )
        self.skip_tr_projection = nn.Linear(2 * emb_dim, emb_dim)
        self.skip_tj_projection = nn.Linear(2 * emb_dim, emb_dim)
        self.output_mlp_tr = nn.Sequential(
            Conv1d_with_init(emb_dim, 2 * emb_dim, kernel_size=3),
            nn.ReLU(),
            Conv1d_with_init(2 * emb_dim, 1, kernel_size=3)
        )
        self.output_mlp_tj = nn.Sequential(
            nn.Linear(self.hidden_traj, 64),
            nn.GELU(),
            nn.Linear(64, self.num_traj)
        )


    def forward(self, x, diffusion_step):
        B, inputdim, L, _ = x.shape
        x = x.squeeze(1)
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        y_tr_list = []
        y_tj_list = []
        x_tr_split = torch.split(x[:, :, 0], self.split_list, dim=1)
        # Aggregating of trajectory
        x_tj_agg = []
        x_tj_ori = x[:, :, 1:]
        for i in range(len(self.residual_layers)-1, -1, -1):  # 3, 2, 1, 0
            if i == len(self.residual_layers)-1:
                x_reshaped = x_tj_ori.view(B, L // 2 ** i, 2 ** i, self.num_traj)
            else:
                x_reshaped = x_tj_ori.view(B, L // 2 ** (i+1), 2 ** (i+1), self.num_traj)
            x_merged = x_reshaped.mean(dim=2)
            x_tj_agg.append(x_merged)

        for i in range(len(self.residual_layers)):
            layer = self.residual_layers[i]
            x_tr = x_tr_split[i]
            x_tj = self.input_mlp_tj(x_tj_agg[i])

            if i == 0:
                model_vq = self.vqvae.vq
                model_encoder = self.vqvae.encoder
                latent = model_encoder(x_tr, self.CF)
                vq_loss, quantized, _, embedding_weight, encoding_indices, encodings = model_vq(latent)
                compressed_time = int(self.split_list[i]/ self.CF)
                code_dim = quantized.shape[-2]
                codes = quantized.reshape(B, code_dim, compressed_time)
                x_tr = codes.permute(0, 2, 1)
            else:
                x_tr = x_tr.unsqueeze(-1)

            emb_tr, emb_tj = self.pos_embedding(x_tr, x_tj, i)

            if i > 0:
                emb_tr = self.skip_tr_projection(torch.cat([emb_tr, skip_tr], dim=2))
                emb_tj = self.skip_tj_projection(torch.cat([emb_tj, skip_tj], dim=2))
            y_tr, y_tj = layer(emb_tr, emb_tj, diffusion_emb)

            if i > 0:
                skip_tr = y_tr.repeat_interleave(2, dim=1)[:, 1:, :]
                skip_tj = y_tj.repeat_interleave(2, dim=1)[:, 1:, :]
                y_tr_list.append(y_tr[:, 1:, :])
                y_tj_list.append(y_tj[:, 1:, :])
            else:
                skip_tr = y_tr
                skip_tj = y_tj
                y_tr_list.append(y_tr[:, 1:, :])
                y_tj_list.append(y_tj[:, 1:, :])

        sum_y_tr = torch.cat(y_tr_list, dim=1)
        sum_y_tj = torch.cat(y_tj_list, dim=1)

        if self.mode == "traf":
            y_out = self.output_mlp_tr(sum_y_tr.permute(0, 2, 1)).permute(0, 2, 1)
            y_out = y_out.squeeze(-1)  # (B,L)
        elif self.mode == "traj":
            y_out = self.output_mlp_tj(sum_y_tj)  # (B,L,N)
        else:
            print("wrong mode:", self.mode)
            exit(0)
        return y_out


class ResidualBlock(nn.Module):
    def __init__(self, channels, diffusion_embedding_dim, att_emb_dim, split_len, nheads, device, is_linear=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.mid_projection = Conv2d_with_init(channels, 2 * channels, 3, 3)
        self.output_projection = Conv2d_with_init(channels, 2 * channels, 3, 3)
        self.is_linear = is_linear
        if is_linear:
            self.feature_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
        else:
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.split_len = split_len
        self.TSAtt = TempSpatioCrossAtt(self.split_len, att_emb_dim, device)

    def forward(self, x_tr, x_tj, diffusion_emb):
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(1)
        x_tr += diffusion_emb
        x_tj += diffusion_emb
        y_tr, y_tj = self.TSAtt(x_tr, x_tj)
        return (x_tr + y_tr) / math.sqrt(2.0), (x_tj + y_tj) / math.sqrt(2.0)


def draw_single(sample, step, mode):
    B, channels, H, W = sample.shape
    samples = sample.cpu().detach().numpy()
    step_numpy = step[0].cpu().detach().numpy()
    t = step_numpy
    for i in range(B):
        stft_sample = samples[i]
        plt.figure(figsize=(5, 5))
        plt.pcolormesh(np.linspace(0, 336, 336), np.linspace(0, 32, 32), stft_sample[0, :, :], shading='gouraud')
        plt.colorbar()
        plt.title('Single')
        plt.savefig("save_model/img/img_{}_{}".format(t, mode))
        break


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class TempSpatioCrossAtt(nn.Module):
    def __init__(self, split_len, emb_dim, device, heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn_1 = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=heads, dropout=dropout)
        self.cross_attn_2 = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=heads, dropout=dropout)
        self.norm = nn.ModuleList([
            nn.LayerNorm(emb_dim, elementwise_affine=False, eps=1e-6)   # 沿着特征维度（最后一个维度）归一化
            for _ in range(4)])
        self.mlp = nn.ModuleList([
            self.create_mlp(emb_dim)
            for _ in range(4)])

    def create_mlp(self, in_dim):
        return nn.Sequential(
            Conv1d_with_init(in_dim, 2 * in_dim, 3),  # 对dim=1
            nn.ReLU(),
            Conv1d_with_init(2 * in_dim, in_dim, 3)
        )

    def forward(self, x_tr, x_tj):
        x_tr = self.norm[0](x_tr)
        x_tj = self.norm[1](x_tj)
        x_tr = x_tr.permute(1, 0, 2)
        x_tj = x_tj.permute(1, 0, 2)
        attn_out_1, _ = self.cross_attn_1(query=x_tr, key=x_tj, value=x_tj)
        attn_out_2, _ = self.cross_attn_2(query=x_tj, key=x_tr, value=x_tr)
        x_tr = x_tr + attn_out_1
        x_tj = x_tj + attn_out_2

        y_tr = self.norm[2](x_tr.permute(1, 0, 2)).permute(0, 2, 1)
        y_tj = self.norm[3](x_tj.permute(1, 0, 2)).permute(0, 2, 1)
        y_tr = self.mlp[0](y_tr).permute(0, 2, 1)
        y_tj = self.mlp[1](y_tj).permute(0, 2, 1)

        y_tr = y_tr + x_tr.permute(1, 0, 2)
        y_tj = y_tj + x_tj.permute(1, 0, 2)
        return y_tr, y_tj


class PositionEmbedding(nn.Module):
    def __init__(self, max_seqLen, device, dim_tr=1, emb_size=32):
        super().__init__()
        self.max_seqLen = max_seqLen
        self.mlp = nn.Sequential(
            nn.Linear(dim_tr, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )
        self.device = device
        self.cls_tr = nn.Parameter(torch.randn(1, 1, emb_size)).to(self.device)
        self.cls_tj = nn.Parameter(torch.randn(1, 1, emb_size)).to(self.device)
        self.posEmb = PositionalEncoding(emb_size, 0).to(self.device)
        self.typeEmb = nn.Embedding(2, emb_size).to(self.device)

    def forward(self, x1, x2, i):
        B, seqLen_tr, _ = x1.shape
        B, seqLen_tj, _ = x2.shape
        if i > 0:
            x1 = self.mlp(x1)
        cls_tr = self.cls_tr.repeat(B, 1, 1)
        cls_tj = self.cls_tj.repeat(B, 1, 1)
        x_tr = torch.cat([cls_tr, x1], dim=1)
        x_tj = torch.cat([cls_tj, x2], dim=1)
        x_tr = self.posEmb(x_tr)
        x_tj = self.posEmb(x_tj)
        emb_tr, emb_tj = (
            x_tr + self.typeEmb(torch.zeros((B, seqLen_tr+1), dtype=torch.long).to(self.device)),
            x_tj + self.typeEmb(torch.ones((B, seqLen_tr+1), dtype=torch.long).to(self.device))
        )
        return emb_tr, emb_tj

