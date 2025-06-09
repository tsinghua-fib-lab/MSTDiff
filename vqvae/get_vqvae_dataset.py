import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import pywt
from torch.utils.data import DataLoader, Dataset


def DWT_list(signal, level, wavelet):
    CA3 = []
    CD3 = []
    CD2 = []
    CD1 = []
    for i in range(len(signal)):
        sig = signal[i]
        c = pywt.wavedec(sig, wavelet, level=level, mode="constant")
        CA3.append(c[0])
        CD3.append(c[1])
        CD2.append(c[2])
        CD1.append(c[3])
    return CA3, CD3, CD2, CD1


def normalize_max(data):
    N = data.shape[0]
    data_new = data
    max_list = []
    for n in range(N):
        max = np.max(data[n])
        if max == 0:
            max = 1
        data_new[n] = data[n]/max
        max_list.append(max)
    return data_new, max_list


class VQDataset(Dataset):
    def __init__(self, level=3, wavelet="db1", eval_length=336, use_index_list=None, seed=0):
        np.random.seed(seed)
        self.eval_length = eval_length
        idx = 5375
        data_path = "./data_vq"
        os.makedirs(data_path, exist_ok=True)
        data_file = f"{data_path}/VQ_Train_Dwt{level}_{wavelet}_S{seed}_Num{idx}.pk"
        src_path = f"./data_src/30min/Traf_Traj_5375_emb32_20250116_dense_30min_idx_final.npz"

        if os.path.isfile(data_file) == False:
            print(f"-------- Creating {data_file} --------")
            data = np.load(src_path, allow_pickle=True)
            print(data.files)
            self.user_id = data["userID"][:idx].astype(np.float32)
            self.ts = data["ts"][:idx].astype(np.float32)
            self.utf_norm, max_list = normalize_max(data["utf"][:idx])
            self.utf_norm = self.utf_norm.astype(np.float32)
            # perform DWT
            self.CA3, self.CD3, self.CD2, self.CD1 = DWT_list(self.utf_norm, level=level, wavelet=wavelet)
            with open(data_file, "wb") as f:
                pickle.dump(
                    [self.user_id, self.ts, self.utf_norm,
                     self.CA3, self.CD3, self.CD2, self.CD1], f
                )
            print("-------- Finish creating --------")
        else:
            with open(data_file, "rb") as f:
                (self.user_id, self.ts, self.utf_norm,
                 self.CA3, self.CD3, self.CD2, self.CD1) = pickle.load(f)

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.user_id))
        else:
            self.use_index_list = use_index_list


    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "user_id": self.user_id[index],
            "ts": self.ts[index],
            "utf_norm": self.utf_norm[index],
            "CA3": self.CA3[index],
            "CD3": self.CD3[index],
            "CD2": self.CD2[index],
            "CD1": self.CD1[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=0, batch_size=100):
    dataset_tmp = VQDataset(seed=seed)

    ind_list = np.arange(len(dataset_tmp))
    np.random.seed(seed)
    np.random.shuffle(ind_list)

    train_dataset = VQDataset(
        use_index_list=ind_list, seed=seed
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("\n-------- Successfully load dataloader! --------")
    print("Train length: ",  len(ind_list))
    print("Train_loader length: ", len(train_loader))
    print("\n")

    return train_loader


if __name__ == "__main__":
    src_path = f"./data_src/30min/Traf_Traj_5375_emb32_20250116_dense_30min_idx_final.npz"

    data = np.load(src_path, allow_pickle=True)
    print(data.files)
    utf,_ = normalize_max(data["utf"])

    utf0 = utf[10]
    utf0_norm = (utf0-np.mean(utf0))/np.std(utf0)
    plt.figure()
    plt.plot(utf0,'b')
    plt.show()

    plt.figure()
    plt.plot(utf0_norm, 'r')
    plt.show()
