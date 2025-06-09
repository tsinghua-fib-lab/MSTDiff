import os

import torch
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm
from models.revin import RevIN
from models.vqvae import vqvae
from get_vqvae_dataset import *
from collections import defaultdict


def get_colormap(cm_name, N):
    base = plt.get_cmap(cm_name)
    return [base(i % base.N) for i in range(N)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=str, required=True)
    parser.add_argument('--coff', type=str, required=True)
    parser.add_argument('--K', type=int, required=True)
    parser.add_argument('--dim', type=int, required=True)
    parser.add_argument('--it', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epoch', default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    fig_dir = f"./result_fig_K{args.K}dim{args.dim}"
    os.makedirs(fig_dir, exist_ok=True)

    # dim_name = "Dim64_Num256_CF1_B100_ITR100000"
    dim_name = f"Dim{args.dim}_Num{args.K}_CF1_B10_ITR{args.it}"
    save_folder = f"./save_model/{dim_name}"
    with open(f"{save_folder}/config.json", 'r') as jsonfile:
        config = json.load(jsonfile)

    vqvae_config = config["vqvae_config"]
    model = vqvae(vqvae_config)
    if args.epoch == 0:
        checkpoint = torch.load(f"{save_folder}/checkpoints/{args.time}/{args.coff}/model_final_{args.coff}.pth", map_location=torch.device(args.device))
        ep = "final"
    else:
        checkpoint = torch.load(f"{save_folder}/checkpoints/{args.time}/{args.coff}/model_{args.coff}_epoch{args.epoch}.pth",map_location=torch.device(args.device))
        ep = args.epoch
    model.load_state_dict(checkpoint)
    model.to(args.device)
    model.eval()

    model_RevIN = RevIN(num_features=1).to(args.device)
    train_loader = get_dataloader(
        seed=args.seed,
        batch_size=vqvae_config["batch"],
    )
    coff_all = []
    # normalize and flatten
    with torch.no_grad():
        for batch in tqdm(train_loader):
            batch_coff = batch[args.coff].to(args.device)
            batch_coff_norm = model_RevIN(batch_coff, "norm")
            coff_all.append(batch_coff_norm)
    coff_all = torch.cat(coff_all, dim=0)  # shape: [N, T]
    print("coff_all", coff_all.shape)  # [5375, 42]
    NUM_USR = coff_all.shape[0]
    T = coff_all.shape[1]

    model_vq = model.vq
    model_encoder = model.encoder

    latent = model_encoder(coff_all, vqvae_config["compression_factor"])
    vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = model_vq(latent)

    compressed_time = int(T/vqvae_config["compression_factor"])  # if CF=1, then it is T
    code_dim = quantized.shape[-2]
    codebook = embedding_weight  # [128, 32]
    codes = quantized.reshape(NUM_USR, code_dim, compressed_time)  # [5375, 32, 42]
    code_ids = encoding_indices.view(NUM_USR, compressed_time)  # [5375, 42]
    print("codebook, codes, code_ids", codebook.shape, codes.shape, code_ids.shape)

    N_CLU = 10  # CLUSTER Num
    colors = get_colormap("tab10", N_CLU)
    kmeans = KMeans(n_clusters=N_CLU, random_state=42)
    cluster_labels = kmeans.fit_predict(coff_all.cpu().numpy())
    print("CLUSTER num: ", np.bincount(cluster_labels))
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    coff_2d = tsne.fit_transform(coff_all.cpu().numpy())  # shape: [N, 2]
    plt.figure(figsize=(16, 16))
    for i in range(N_CLU):
        idx = (cluster_labels == i)
        plt.scatter(coff_2d[idx, 0], coff_2d[idx, 1], color=colors[i], label=f"Cluster {i}", alpha=0.7)
    plt.title(f"{args.coff} clustered by KMeans, cluster={N_CLU}", fontsize=20)
    # plt.legend(loc="upper right", fontsize=10)
    plt.grid(True)
    plt.savefig(f"{fig_dir}/t-SNE_{args.coff}_clu{N_CLU}.png")
    plt.close()

    # Get t-sne of codebook
    quantized_flat = quantized.permute(0, 2, 1).reshape(NUM_USR, -1).detach().cpu().numpy()  # shape: [NUM, 42×32]
    quantized_2d = TSNE(n_components=2, perplexity=100, random_state=42).fit_transform(quantized_flat)
    plt.figure(figsize=(16, 16))
    for i in np.unique(cluster_labels):
        idx = (cluster_labels == i)
        plt.scatter(quantized_2d[idx, 0], quantized_2d[idx, 1], color=colors[i], label=f"Cluster {i}", alpha=0.7)
    plt.title(f"{args.coff} clustered by KMeans, map to codebook, cluster={N_CLU}", fontsize=20)
    plt.grid(True)
    plt.savefig(f"{fig_dir}/t-SNE_{args.coff}_clu{N_CLU}_cb.png")
    plt.close()


    # -------- plot heatmap -------------
    # codebook = codebook.detach().cpu().numpy()

    # plt.figure(figsize=(24, 6))
    # plt.imshow(codebook.T, cmap='YlGnBu', interpolation='nearest')
    # plt.colorbar()
    # plt.title(f"Codebook Embedding ({args.coff})", fontsize=20)
    # plt.ylabel("Embedding Dim", fontsize=20)
    # plt.xlabel("Index", fontsize=20)
    # plt.xticks([])
    # plt.tight_layout()
    # plt.savefig(f"{fig_dir}/Codebook_{args.coff}_ep_{ep}.png")
    # plt.close()

