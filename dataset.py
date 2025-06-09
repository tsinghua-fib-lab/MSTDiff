import os
import pickle
import torch
import re
import numpy as np
import math
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from scipy.signal import stft, istft
from matplotlib import pyplot as plt
import json
import pywt

def normalize_max(data):
    N = data.shape[0]
    data_new = data
    for n in range(N):
        max = np.max(data[n])
        if max == 0:
            max = 1
        data_new[n] = data[n]/max
    return data_new


def normalize_energy(data):
    print("normalize energy", data.shape)
    N = data.shape[0]
    data_new = data.copy()
    for n in range(N):
        energy = np.sqrt(np.sum(data[n] ** 2))
        if energy == 0:
            energy = 1
        data_new[n] = data[n] / energy
    return data_new


def normalize_z(data):
    return (data-np.mean(data))/np.std(data)


def load_Qt(path):
    data = torch.load(path, weights_only=True)
    Qt = data['Qt'].requires_grad_(False)
    Qtbar = data['Qt_bar'].requires_grad_(False)
    return Qt, Qtbar


class Traffic_Dataset(Dataset):
    def __init__(self, eval_length=336, use_index_list=None, missing_ratio=0, seed=0):
        np.random.seed(seed)
        self.eval_length = eval_length
        idx = 5375  # 1000  5375
        DIM = 32
        wv_is_norm = False
        path = "../data/Train" + "_sd" + str(seed) + "_len" + str(idx) + "_dim" + str(DIM) + "_wvn"+ str(wv_is_norm) +".pk"

        data_path = "../data_src/30min/Traf_Traj_5375_emb32_20250604_dense_30min_CID_final.npz"
        qtall_path = "../data/Qt_all.pt"

        if os.path.isfile(path) == False:
            data = np.load(data_path, allow_pickle=True)
            print(data.files)
            self.user_id = data["userID"][:idx].astype(np.float32)
            self.ts = data["ts"][:idx].astype(np.float32)
            self.utf_norm = normalize_max(data["utf"][:idx]).astype(np.float32)
            self.traj_cid = data["traj_cid"][:idx].astype(np.float32)
            self.traj_emb = data["bs_emb"][:idx].astype(np.float32)
            self.utf_wv, self.wv_maxmin = DWT(self.utf_norm, level=3, wavelet='db1', is_norm=wv_is_norm)  # 不考虑归一化，因为发现归一化后数据相对关系有变化
            self.Qt_all, self.Qtbar_all = load_Qt(path=qtall_path)
            with open(path, "wb") as f:
                pickle.dump(
                    [self.user_id, self.ts,
                     self.traj_emb, self.utf_norm, self.traj_cid, self.utf_wv, self.wv_maxmin, self.Qt_all, self.Qtbar_all
                     ], f
                )
        else:
            with open(path, "rb") as f:
                (self.user_id, self.ts,
                     self.traj_emb, self.utf_norm, self.traj_cid, self.utf_wv, self.wv_maxmin, self.Qt_all, self.Qtbar_all) = pickle.load(f)

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.user_id))
            print("None: use_index_list length: ", len(self.use_index_list))
        else:
            self.use_index_list = use_index_list
            print("Use_index_list length: ", len(self.use_index_list))

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "user_id": self.user_id[index],
            "ts": self.ts[index],
            "utf_norm": self.utf_norm[index],
            "timepoints": np.arange(self.eval_length),
            "traj_cid": self.traj_cid[index],
            "traj_emb": self.traj_emb[index],
            "utf_wv": self.utf_wv[index]
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(trainp=0.7, seed=0, nfold=0, batch_size=1, missing_ratio=0, finetune=False):
    dataset_tmp = Traffic_Dataset(missing_ratio=missing_ratio, seed=seed)
    print("\n****************** load dataset **************************")

    ind_list = np.arange(len(dataset_tmp))
    np.random.seed(seed)
    np.random.shuffle(ind_list)
    start = (int)(nfold * 0.2 * len(dataset_tmp))
    end = (int)((nfold + 1) * 0.2 * len(dataset_tmp))
    test_index = ind_list[start:end]

    remain_list = np.delete(ind_list, np.arange(start, end))
    num_train = (int)(len(remain_list) * trainp)
    train_index = remain_list[:num_train]  # 70% train
    valid_index = remain_list[num_train:]  # 30% valid

    print("len of train: ",  len(train_index))
    print("len of valid: ", len(valid_index))
    print("len of test: ", len(test_index))

    train_dataset = Traffic_Dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = Traffic_Dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = Traffic_Dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total_loader = DataLoader(dataset_tmp, batch_size=batch_size, shuffle=False)

    print("Successfully load dataset!")

    return train_loader, valid_loader, test_loader, total_loader


def norm_coeff(c):
    c_arr = np.stack(c, axis=0)
    cmin = float(c_arr.min())
    cmax = float(c_arr.max())
    c_norm = (c_arr - cmin) / max(cmax - cmin, 1e-6)
    return c_norm, cmax, cmin


def DWT(signal, level, wavelet, is_norm=True):
    B, L = signal.shape
    CA3s, CD3s, CD2s, CD1s = [], [], [], []
    for i in range(B):
        sig = signal[i]
        coeffs = pywt.wavedec(sig, wavelet, level=level, mode="constant")
        CA3s.append(coeffs[0])
        CD3s.append(coeffs[1])
        CD2s.append(coeffs[2])
        CD1s.append(coeffs[3])

    CA3s = np.stack(CA3s, axis=0)
    CD3s = np.stack(CD3s, axis=0)
    CD2s = np.stack(CD2s, axis=0)
    CD1s = np.stack(CD1s, axis=0)

    dict_maxmin = {}
    if is_norm:
        CA3s, maxA, minA = norm_coeff(CA3s)
        CD3s, max3, min3 = norm_coeff(CD3s)
        CD2s, max2, min2 = norm_coeff(CD2s)
        CD1s, max1, min1 = norm_coeff(CD1s)
        dict_maxmin = {"CA3": [maxA, minA], "CD3": [max3, min3], "CD2": [max2, min2], "CD1": [max1, min1]}
    coeffs_cat = np.concatenate([CA3s, CD3s, CD2s, CD1s], axis=1)  # (B, L)
    print("coeff.shape", coeffs_cat.shape)
    return coeffs_cat, dict_maxmin


def iDWT(coeffs, split_list, level, wavelet, dict_maxmin, is_norm=True):
    B, L = coeffs.shape
    rec_signals = []
    for i in range(B):
        c = coeffs[i]
        split_vals = np.split(c, np.cumsum(split_list)[:-1])  # (42, 84, 168)
        cA3, cD3, cD2, cD1 = split_vals
        if is_norm:
            cA3 = cA3 * (dict_maxmin["CA3"][0] - dict_maxmin["CA3"][1]) + dict_maxmin["CA3"][1]
            cD3 = cD3 * (dict_maxmin["CD3"][0] - dict_maxmin["CD3"][1]) + dict_maxmin["CD3"][1]
            cD2 = cD2 * (dict_maxmin["CD2"][0] - dict_maxmin["CD2"][1]) + dict_maxmin["CD2"][1]
            cD1 = cD1 * (dict_maxmin["CD1"][0] - dict_maxmin["CD1"][1]) + dict_maxmin["CD1"][1]
        rec = pywt.waverec([cA3, cD3, cD2, cD1], wavelet=wavelet, mode='constant')
        rec_signals.append(rec)
    rec_signals = np.stack([r for r in rec_signals], axis=0)
    return rec_signals


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


def total_traj_embs(path):
    with open(path, 'r') as jf:
        emb_dict = json.load(jf)
    N = len(list(emb_dict.keys()))
    itm = list(emb_dict.values())
    print(f"Num of bss:{N}")
    embs = []
    for i in range(N):
        embs.append(itm[i]["emb"])
    print("Len of embs", len(embs), len(embs[0]))
    return np.array(embs, dtype=np.float32)

def load_npy(path):
    data=np.load(path, allow_pickle=True)
    print(type(data))
    return data


if __name__ == "__main__":
    train_loader, valid_loader, test_loader, total_loader = get_dataloader(
        trainp=0.7,
        seed=0,
        nfold=0,
        batch_size=1,
        missing_ratio=0
    )

    for data in train_loader:
        print(data["ts"][0])
        break