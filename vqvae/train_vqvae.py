import argparse
import json
import numpy as np
import os
import pdb
import random
import time
import datetime
import torch
import yaml
from tqdm import tqdm
from models import get_model_class
from time import gmtime, strftime
import matplotlib.pyplot as plt
from models.vqvae import vqvae
from models.revin import RevIN
from get_vqvae_dataset import *


def run_main(device, vqvae_config, save_dir, args):
    if 'general_seed' not in vqvae_config:
        vqvae_config['seed'] = random.randint(0, 9999)
    general_seed = vqvae_config['general_seed']
    torch.manual_seed(general_seed)
    random.seed(general_seed)
    np.random.seed(general_seed)
    torch.backends.cudnn.deterministic = True

    model = vqvae(vqvae_config)
    model_RevIN = RevIN(num_features=1)
    print('Total # trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    train_loader = get_dataloader(
        seed=args.seed,
        batch_size=vqvae_config["batch"],
    )
    # Start training
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    coff_names = ["CA3", "CD3", "CD2", "CD1"]
    for coff_name in coff_names:
        model_save_dir = f'{save_dir}/checkpoints/{start_time}/{coff_name}'
        os.makedirs(model_save_dir)
        print(f"\nTraining model for: {model_save_dir}")
        model = train_model(model, model_RevIN, train_loader, device, vqvae_config, model_save_dir, coff_name)
        torch.save(model.state_dict(), f'{model_save_dir}/model_final_{coff_name}.pth')
    return vqvae_config


def train_model(model, model_RevIN, train_loader, device, vqvae_config, model_save_dir, coff_name):
    optimizer = model.configure_optimizers(lr=vqvae_config['lr'])
    model.to(device)
    model_RevIN.to(device)
    print(f"Batch size: {vqvae_config['batch']}")

    loss_list = []
    vq_loss_list = []
    recon_error_list = []
    for epoch in tqdm(range(int((vqvae_config['total_iter']/len(train_loader)) + 0.5))):
        model.train()
        for i, (batch_x) in enumerate(train_loader):
            batch_coff = batch_x[coff_name]  # 42，42，84，168
            batch_coff_tensor = torch.tensor(batch_coff, dtype=torch.float, device=device)
            batch_coff_tensor_norm = model_RevIN(batch_coff_tensor, "norm")

            loss, vq_loss, recon_error, x_recon, perplexity, embedding_weight, encoding_indices, encodings = model.shared_eval(batch_coff_tensor_norm, optimizer, 'train')

            loss_list.append(loss.detach().cpu().item())
            vq_loss_list.append(vq_loss.detach().cpu().item())
            recon_error_list.append(recon_error.detach().cpu().item())
        if (epoch+1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(model_save_dir, f'model_{coff_name}_epoch{epoch+1}.pth'))
            print('Saved model from epoch ', epoch+1)

    fig_dir = "./result_fig"
    os.makedirs(fig_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    plt.plot(loss_list, 'r', label="Total loss")
    plt.plot(vq_loss_list, 'b', label="VQ loss")
    plt.plot(recon_error_list, 'purple', label="Recon loss")
    plt.legend(loc="upper right", fontsize=20)
    plt.xlabel("Iteration", fontsize=25)
    plt.ylabel("Loss", fontsize=25)
    plt.title("Loss plot", fontsize=25)
    plt.savefig(f"{fig_dir}/loss_{epoch+1}_{coff_name}.png")
    plt.close()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:4')
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--compression_factor', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nfold', type=int, default=0)
    parser.add_argument('--wvLevel', type=int, default=3)
    args = parser.parse_args()

    config_file = "./config/config_vqvae.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    config['vqvae_config']['batch'] = args.batch
    config['vqvae_config']['compression_factor'] = args.compression_factor

    folder_name = 'Dim' + str(config['vqvae_config']['embedding_dim']) + '_Num' + str(
        config['vqvae_config']['num_embeddings']) + '_CF' + str(
        config['vqvae_config']['compression_factor']) + '_B' + str(
        config['vqvae_config']['batch']) + '_ITR' + str(
        config['vqvae_config']['total_iter'])

    save_dir = f"./save_model/{folder_name}"
    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = args.device
    else:
        device = 'cpu'

    vqvae_config = run_main(device=device, vqvae_config=config['vqvae_config'], save_dir=save_dir, args=args)

    config['vqvae_config'] = vqvae_config
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)





