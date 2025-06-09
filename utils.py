import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.spatial import distance
import os


def train(
        model,
        config,
        train_loader,
        device,
        valid_loader=None,
        valid_epoch_interval=20,
        foldername="",
        finetune=False,
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)

    epoch = config["epoch"]

    p1 = int(0.75 * epoch)
    p2 = int(0.9 * epoch)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10

    epoch_train_losses = []
    epoch_valid_losses = []
    epoch_train_losses_tf = []
    epoch_train_losses_tj = []

    for epoch_no in range(epoch):
        if foldername != "":
            output_path = foldername + "/model_ep{}.pth".format(epoch_no + 1)

        avg_loss = 0
        avg_loss_tf = 0
        avg_loss_tj = 0

        model.train()

        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss, loss_tf, loss_tj = model(train_batch)
                loss.backward()

                avg_loss += loss.item()
                avg_loss_tf += loss_tf.item()
                avg_loss_tj += loss_tj.item()

                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

            epoch_train_losses.append(avg_loss / batch_no)
            epoch_train_losses_tf.append(avg_loss_tf / batch_no)
            epoch_train_losses_tj.append(avg_loss_tj / batch_no)
            lr_scheduler.step()

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:  # 进度条
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss,_,_ = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )

            avg_valid_loss = avg_loss_valid / batch_no
            epoch_valid_losses.append(avg_valid_loss)

            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

        if foldername != "":
            if (epoch_no + 1) % 10 == 0:
                torch.save(model.state_dict(), output_path)

    plot_loss(epoch_train_losses, epoch, "Loss")
    plot_loss(epoch_train_losses_tf, epoch, "Loss_tf")
    plot_loss(epoch_train_losses_tj, epoch, "Loss_tj")


def plot_loss(losses, ep, figName):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, ep + 1), losses, linewidth=3)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.title("Training Loss", fontsize=20)
    plt.savefig(f"result_metrics/{figName}_m1_{ep}.png")


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def evaluate(model, test_loader, device, nsample=1, scaler=1, mean_scaler=0, foldername=""):
    IMG_COUNTER = 0

    with torch.no_grad():
        model.eval()
        generated_tr_list = []
        generated_tj_list = []
        real_traf_list = []
        real_traj_list = []

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, batch_data in enumerate(it, start=1):
                samples, target_traf, target_traj = model.evaluate(batch_data, nsample)
                IMG_COUNTER += 1
                # generated data
                generated_tr_list.append(samples[:, 0, :, 0])
                generated_tj_list.append(samples[:, 0, :, 1])
                # real data
                real_traf_list.append(target_traf.cpu().numpy())
                real_traj_list.append(target_traj.cpu().numpy())
                it.set_postfix(
                    ordered_dict={
                        "batch_no": batch_no,
                    },
                    refresh=True
                )
        generated_tr_tensor = [arr.cpu().numpy() if isinstance(arr, torch.Tensor) else arr for arr in
                               generated_tr_list]
        generated_tr = np.concatenate(generated_tr_tensor, axis=0)  # (Num, 336)

        generated_tj_tensor = [arr.cpu().numpy() if isinstance(arr, torch.Tensor) else arr for arr in
                               generated_tj_list]
        generated_tj = np.concatenate(generated_tj_tensor, axis=0)  # (Num, 336, 32)

        real_traf = np.concatenate(real_traf_list, axis=0)
        real_traj = np.concatenate(real_traj_list, axis=0)

        np.savez(f"GenData/generated_data_{generated_tr.shape[0]}.npz",
                 generated_traf=generated_tr,
                 generated_traj=generated_tj,
                 real_traf=real_traf,
                 real_traj=real_traj)

        metric_results = np.zeros((nsample, 3))
        for n in range(nsample):
            jsd, jsd_diff = evaluate_time(generated_tr, real_traf, save_dir="GenData")
            rmse = evaluate_freq(generated_tr, real_traf)
            metric_results[n, :] = [jsd, jsd_diff, rmse]
        metric_result_mean = metric_results.mean(0)
        print("results of traffic: ", metric_result_mean)
        save_metrics_to_file(metric_result_mean, 'result_metrics/metrix.txt')


def draw_images(samples_in, target, IMG_COUNTER):
    B, n_samples, dim, H, W = samples_in.shape
    samples = samples_in.cpu().numpy()

    target_data = target.cpu().numpy()
    nfold = 0

    print("[target is] ", target_data.shape)
    for i in range(B):
        for j in range(n_samples):
            stft_sample = samples[i][j]

            plt.figure(figsize=(10, 6))
            plt.pcolormesh(np.linspace(0, 336, 336), np.linspace(0, 32, 32), stft_sample[1], shading='gouraud')
            plt.colorbar()
            plt.title('Generated Magnitude of Traffic Data')
            plt.savefig(f"../img_gen/img_{IMG_COUNTER * B + i + 1}_mag.png")
            plt.close()
            plt.show()

            plt.figure(figsize=(10, 6))
            plt.pcolormesh(np.linspace(0, 336, 336), np.linspace(0, 32, 32), stft_sample[2], shading='gouraud')
            plt.colorbar()
            plt.title('Generated Phase of Traffic Data')
            plt.savefig(f"../img_gen/img_{IMG_COUNTER * B + i + 1}_ph.png")
            plt.close()
            plt.show()


def draw_series(samples, target_traf, IMG_COUNTER):
    B, n_samples, L = samples.shape
    target_traf = target_traf.cpu().numpy()
    print("[target traf]", target_traf.shape)
    for i in range(B):
        for j in range(n_samples):
            istft_sample = samples[i][j]
            plt.figure(figsize=(10, 6))
            plt.plot(np.linspace(0, L, L), istft_sample)
            plt.title('ISTFT of sample')
            plt.savefig(f"../img_series/img_{IMG_COUNTER}_istft.png")
            plt.close()
            # plt.show()
        # draw target
        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, L, L), target_traf[i])
        plt.title('Target traf')
        plt.savefig(f"../img_series/img_{IMG_COUNTER}_target.png")
        plt.close()
        # plt.show()


def draw_series_gen_real(samples, target_traf, IMG_COUNTER):
    B, n_samples, L = samples.shape
    print("B is: ", B)
    target_traf = target_traf.cpu().numpy()
    nfold = 0
    print("[target traf]", target_traf.shape)
    for i in range(B):
        for j in range(n_samples):
            istft_sample = samples[i][j]
            # normalize
            istft_sample_norm = normalize_max_sample(istft_sample)
            plt.figure(figsize=(6, 3.5))
            plt.plot(np.linspace(0, L, L), istft_sample_norm, color='steelblue', linewidth=1, label="Generated Traffic")
            plt.ylabel("Normalized Traffic", fontsize=16, labelpad=0.15)
            plt.ylim(0, 1.2)
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            xticks_positions = np.arange(0, 336, 48)
            plt.xticks(xticks_positions, [''] * len(xticks_positions))
            for n, day in enumerate(days):
                plt.text(n * 48 + 24, -0.065, day, ha='center', transform=plt.gca().get_xaxis_transform(), fontsize=14)
            plt.tight_layout()
            plt.savefig(f"../img_gen/img_{IMG_COUNTER * B + i + 1}_istft.png")
            plt.close()


def evaluate_time(samples, target_traf, save_dir='', n_bins=100):
    # do not use this to evaluate
    samples[samples < 0] = 0
    fig, ax = plt.subplots(figsize=(24, 16))
    line_w = 2
    use_cumulative = False
    use_log = True
    n_gene, bins, patches = ax.hist(samples.flatten(), n_bins, density=True, histtype='step',
                                    cumulative=use_cumulative, label='Gen data', facecolor='blue',
                                    linewidth=line_w, log=use_log)
    n_real, bins, patches = ax.hist(target_traf.flatten(), n_bins, density=True, histtype='step',
                                    cumulative=use_cumulative, label='Real data', facecolor='orange',
                                    linewidth=line_w, log=use_log)
    JSD = distance.jensenshannon(n_real.flatten(), n_gene.flatten(), 2.0)
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=28)
    ax.set_title('Cumulative step histograms (CDF)', fontsize=45)
    ax.set_xlabel('Log traffic value', fontsize=45)
    ax.set_ylabel('Likelihood of occurrence', fontsize=45)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'fig_hist.jpg'))
    plt.close()

    fig, ax = plt.subplots(figsize=(24, 16))
    real_diff = target_traf[1:] - target_traf[:-1]
    generated_diff = samples[1:] - samples[:-1]
    n_gene, bins, patches = ax.hist(generated_diff.flatten(), n_bins, density=True, histtype='step',
                                    cumulative=use_cumulative, label='Gen data', facecolor='blue',
                                    linewidth=line_w, log=use_log)
    n_real, bins, patches = ax.hist(real_diff.flatten(), n_bins, density=True, histtype='step',
                                    cumulative=use_cumulative, label='Real data', facecolor='orange',
                                    linewidth=line_w, log=use_log)
    ax.grid(True)
    ax.legend(loc='right')
    ax.set_title('Cumulative step histograms')
    ax.set_xlabel('Value')
    ax.set_ylabel('Likelihood of occurrence')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'fig_diff_hist.jpg'))
    plt.close()
    JSD_diff = distance.jensenshannon(n_real.flatten(), n_gene.flatten(), 2.0)
    return JSD, JSD_diff


def evaluate_freq(gen_traf, real_traf):
    data_f_real = np.abs(np.fft.rfft(real_traf, axis=1))
    daily_real = data_f_real[:, 7] / np.sum(data_f_real, axis=1)
    daily_real = np.nan_to_num(daily_real)
    data_f = np.abs(np.fft.rfft(gen_traf, axis=1))
    daily_gen = data_f[:, 7] / np.sum(data_f, axis=1) if data_f.sum(1).all() > 0 else data_f[:, 7]
    daily_gen = np.nan_to_num(daily_gen)
    rmse_daily = np.sqrt(np.mean((daily_real - daily_gen) ** 2))
    return rmse_daily


def save_metrics_to_file(metrics, file_path):
    with open(file_path, 'a') as file:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_metrics = ', '.join(map(str, metrics))
        file.write("********* Current Time: " + current_time + " *********\n")
        file.write(formatted_metrics + '\n\n')


def normalize_max_sample(data):
    max = np.max(data)
    if max == 0:
        max = 1
    return data / max


def normalize_z(data, mean, std):
    epsilon = 1e-10
    std = max(std, epsilon)
    return (data - mean) / std


def normalize_region_sample(data):
    max = np.max(data)
    min = np.min(data)
    if max == 0:
        max = 1
    return (data - min) / (max - min)
