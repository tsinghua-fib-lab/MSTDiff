import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '4,5,6'
import argparse
import json
import yaml
import os
from main_diff import main_CSDI
from dataset import get_dataloader
from utils import train, evaluate
import datetime
import torch
import sys
sys.path.append('xx')
from vqvae.models.vqvae import vqvae
from vqvae.models.revin import RevIN

parser = argparse.ArgumentParser(description="MSDIT")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:5', help='Device for Attack')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--testmissingratio", type=float, default=0)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", default=True)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1)
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--trainp", type=float, default=0.7)
# VQconfig
parser.add_argument('--do_vq', action='store_true')
parser.add_argument('--time', type=str)
parser.add_argument('--coff', type=str, default="CA3")
parser.add_argument('--K', type=int, default=128)
parser.add_argument('--dim', type=int, default=32)
parser.add_argument('--it', type=int, default=500000)
args = parser.parse_args()


path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["test_missing_ratio"] = 0

config["model"]["is_unconditional"] = True
config["model"]["featureemb"] = 16

config["train"]["epoch"] = args.epoch
config["train"]["itr_per_epoch"] = 100000
config["train"]["batch_size"] = 20
config["train"]["lr"] = 5e-5

config["diffusion"]["num_steps"] = 50
config["diffusion"]["diffusion_embedding_dim"] = 16
config["diffusion"]["channels"] = 32
config["diffusion"]["layers"] = 4

# parameters for traffic
config["diffusion"]["beta_tr_start"] = 0.0001
config["diffusion"]["beta_tr_end"] = 0.1
# parameters for trajectory
config["diffusion"]["alpha_tj_start"] = 0.1
config["diffusion"]["alpha_tj_end"] = 1

config["diffusion"]["schedule"] = "quad"


train_loader, valid_loader, test_loader, gen_loader = get_dataloader(
    trainp=args.trainp,
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)

total_batch = len(train_loader)
config["train"]["total_batch"] = total_batch

print("Preparing models · · ·")

model_vqvae = None
model_RevIN = None
if args.do_vq:
    print("Preparing Vqvae · · ·")
    dim_name = f"Dim{args.dim}_Num{args.K}_CF1_B10_ITR{args.it}"
    save_folder_vq = f"./vqvae/save_model/{dim_name}"
    with open(f"{save_folder_vq}/config.json", 'r') as jsonfile:
        config_vq = json.load(jsonfile)
    vqvae_config = config_vq["vqvae_config"]
    model_vqvae = vqvae(vqvae_config)
    checkpoint = torch.load(f"{save_folder_vq}/checkpoints/{args.time}/{args.coff}/model_final_{args.coff}.pth", map_location=torch.device(args.device))
    model_vqvae.load_state_dict(checkpoint)
    model_vqvae.to(args.device)
    model_vqvae.eval()
    model_RevIN = RevIN(num_features=1)
    model_RevIN.eval()
    config["vqvae_config"] = vqvae_config

Qt_all = train_loader.dataset.Qt_all.to(args.device)
Qtbar_all = train_loader.dataset.Qtbar_all.to(args.device)
model = main_CSDI(config, args.device, Qt_all, Qtbar_all, model_RevIN, model_vqvae).to(args.device)

print("Successfully load models")

if args.modelfolder == "":
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = "save_model/Model_" + str(args.nfold) + "_" + current_time + "/"
    print('【model folder】', foldername)

    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

    train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
            device=args.device
        )
else:
    foldername = args.modelfolder
    print("\n****************** load model ***********************")
    checkpoint = torch.load("save_model/" + args.modelfolder + "/model_ep{}.pth".format(args.epoch),
                            map_location=torch.device(args.device), weights_only=True)
    model.load_state_dict(checkpoint)

    print("\n****************** testing model **************")
    evaluate(model, test_loader, device=args.device, nsample=args.nsample, scaler=1, foldername=foldername)

