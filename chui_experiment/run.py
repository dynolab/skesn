import argparse
import os, sys
import numpy as np
from tqdm import tqdm
from math import isclose
import datetime
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from matplotlib.gridspec import GridSpec

sys.path.append(os.getcwd()+"/../")
from skesn.esn import EsnForecaster, update_modes
from skesn.weight_generators import optimal_weights_generator
from skesn.esn_controllers import *
from skesn.data_preprocess import ToNormalConverter

def get_maximas(v):
    left_difs = v[1:-1] - v[:-2]
    right_difs = v[1:-1] - v[2:]
    dif_prods = left_difs * right_difs
    return np.where((left_difs > 0) * (right_difs > 0))[0]+1

def _chui_moffatt(x_0, dt, t_final, xi = 1.):
    alpha_ = 1.5
    omega_ = 1.
    eta_ = 4.
    kappa_ = 4.
    def rhs(x):
        f_ = np.zeros(5)
        f_[0] = alpha_ * (-eta_ * x[0] + omega_ * x[1] * x[2])
        f_[1] = -eta_ * x[1] + omega_ * x[0] * x[2]
        f_[2] = kappa_ * (x[3] - x[2] - x[0] * x[1])
        f_[3] = -x[3] + xi * x[2] - x[4] * x[2]
        f_[4] = -x[4] + x[3] * x[2]
        return f_

    times = np.arange(0, t_final, dt)
    ts = np.zeros((len(times), 5))
    ts[0, :] = x_0
    cur_x = x_0
    dt_integr = 10**(-3)
    n_timesteps = int(np.ceil(dt / dt_integr))
    dt_integr = dt / n_timesteps
    for i in range(1, n_timesteps*len(times)):
        cur_x = cur_x + dt_integr * rhs(cur_x)
        saved_time_i = i*dt_integr / dt
        if isclose(saved_time_i, np.round(saved_time_i)):
            saved_time_i = int(np.round(i*dt_integr / dt))
            ts[saved_time_i, :] = cur_x
    return ts, times

def forecasting(model, data, controls = None):
    samples = data.shape[0]
    T = samples//2
    steps = (T//10-10)
    
    error = 0
    if(controls is not None): controls = controls[:10]
    for j in range(steps):
        model.update(data[T+j*10:T+10+j*10], controls, mode=update_modes.synchronization)
        output = model.predict(10, controls, inspect=False)
        error += np.mean((output - data[T+10+j*10:T+20+j*10])**2)
    error = error / steps

    return error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments with chui model")
    parser.add_argument("--type", choices=["preproc_orig", "preproc_proc", "preproc_log", "train_orig", "train_proc", 
            "train_log", "infer_orig", "infer_proc", "infer_log"], default="infer_orig", help="Type of the experiment")
    parser.add_argument("--controller", choices=["inject", "transfer", "homotopy_simple", "homotopy_transfer"], 
                        default="inject", help="Type of controller")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--xi", type=int, default=32, help="Value of the analyzed xi")

    args = parser.parse_args()

    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.size"] = 12

    xi_idx = args.xi / 2 - 1
    xi_idx_i = int(xi_idx)

    if(not os.path.exists("figures")):
        os.mkdir("figures")

    figfolder = f"figures/{'_'.join(map(str, datetime.datetime.now().utctimetuple()[:6]))}_{args.type}_{args.controller}_{args.xi}/"
    os.mkdir(figfolder)


    if(not os.path.exists("data_chui.npy")):
        np.random.seed(args.seed)
        print("Data generation")
        data = np.zeros((20, 20, 5000, 5))
        pbar = tqdm(total=400, position=0)
        for j in range(20):
            for rho in range(20):
                y0 = np.random.rand(5, )
                ts, time = _chui_moffatt(y0, 5e-4, 220.5, rho*2+2)
                data[j, rho] = ts[1000::80][500:]
                pbar.update(1)
        time = time[:-1000:80]
        np.save("data_chui.npy", data)

    time = np.arange(0, 200, 0.04)
    data_orig = np.load("data_chui.npy")
    data = data_orig.copy()

    #### DRAW ORIGINAL DATA ####
    if("preproc" in args.type):
        plt.figure(figsize=(12,5))
        for i in range(5):
            plt.subplot(5, 1, i+1)
            plt.plot(time[:2500], data[1, xi_idx_i, :2500, i])
            plt.ylabel(f"${'xyzuv'[i]}$")
            if(i == 4): plt.xlabel("$t$")
            else: plt.xticks([])
        plt.tight_layout()
        plt.savefig(figfolder + "original_timeseries.png", dpi=200)

        plt.figure(figsize=(20,3))
        for i in range(5):
            plt.subplot(1, 5, i+1)
            plt.hist(data[:4, xi_idx_i, :, i].flatten(), 100, density=True)
            plt.xlabel(f"${'xyzuv'[i]}$")
        plt.tight_layout()
        plt.savefig(figfolder + "original_distribution.png", dpi=200)
    ############################


    MED, STD = np.mean(data[0, 10, 500:], axis=0), np.std(data[0, -1, 500:], axis=0)

    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(5):
                if(not (k == 1 and ("_proc" in args.type or "_log" in args.type))): 
                    data[i, j, :, k] = (data[i, j, :, k] - MED[k]) / STD[k]

    if("_proc" in args.type): 
        scaler = ToNormalConverter().fit(data[:4, 15:, :, 1].reshape(-1))
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i, j, :, 1] = scaler.transform(data[i, j, :, 1])

    if("_log" in args.type): 
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i, j, :, 1] = np.log(data[i, j, :, 1])
        MED1, STD1 = np.mean(data[0, 15:, 500:, 1]), np.std(data[0, 15:, 500:, 1])
        data[:, :, :, 1] = (data[:, :, :, 1] - MED1) / STD1

    if("preproc_proc" in args.type): 
        # scaler = ToNormalConverter().fit(data[:4, 15:, :, 1].reshape(-1))

        plt.figure(figsize=(12,5))
        t = np.linspace(-5, 5, 100)
        plt.plot(t, scaler.data_to_uni_(t))
        plt.xlabel("$x$")
        plt.ylabel("$F_{\\xi_1}(x)$")
        plt.tight_layout()
        plt.savefig(figfolder + "preprocessed_data2unif.png", dpi=200)

        plt.figure(figsize=(12,5))
        t = np.linspace(-5, 5, 100)
        plt.plot(t, scaler.norm_to_uni_(t))
        plt.xlabel("$x$")
        plt.ylabel("$F_{\\xi_2}(x)$")
        plt.tight_layout()
        plt.savefig(figfolder + "preprocessed_unif2norm.png", dpi=200)

    #### DRAW PREPROCESSED DATA ####
    if("preproc" in args.type):
        plt.figure(figsize=(12,5))
        for i in range(5):
            plt.subplot(5, 1, i+1)
            plt.plot(time[:2500], data[0, xi_idx_i, :2500, i])
            plt.ylabel(f"${'xyzuv'[i]}$")
            if(i == 4): plt.xlabel("$t$")
            else: plt.xticks([])
        plt.tight_layout()
        plt.savefig(figfolder + "preprocessed_timeseries.png", dpi=200)

        plt.figure(figsize=(20,3))
        for i in range(5):
            plt.subplot(1, 5, i+1)
            plt.hist(data[:4, xi_idx_i, :, i].flatten(), 100, density=True)
            plt.xlabel(f"${'xyzuv'[i]}$")
        plt.tight_layout()
        plt.savefig(figfolder + "preprocessed_distribution.png", dpi=200)

        exit(0)
    ############################

    controls = (range(0,20) * np.ones((1,5000,1))).T/9.5 - 1
    if(args.controller == "transfer"): ANC = [14]
    else: ANC = [5,10,14,18]

    np.random.seed(args.seed)
    N = 20
    w_errors = np.zeros((N, 20, ))
    f_errors = np.zeros((N, 20, ))
    print("Network training...")
    if("train" in args.type): pbar = tqdm(total=N, position=0)

    m_kwargs = {}
    
    if(args.controller == "inject"):
        m_kwargs["controller"] = InjectedController()
    elif(args.controller == "transfer"):
        m_kwargs["controller"] = None
    elif(args.controller == "homotopy_simple"):
        m_kwargs["controller"] = HomotopyController(False, eps=1e-2)
    elif(args.controller == "homotopy_transfer"):   
        m_kwargs["controller"] = HomotopyController(True, 0.5, 1e-2) 

    for ep in range(N):
        model = EsnForecaster(
            n_reservoir=500,
            spectral_radius=0.9,
            sparsity=0.1,
            regularization='l2|noise',
            lambda_r=5e-2,
            noise_theta=1e-4,
            in_activation='tanh',
            random_state=args.seed,
            **m_kwargs
        )

        model.fit(data[ep, ANC, :5000], controls[ANC] if (args.controller != "transfer") else None, inspect = False, initialization_strategy = optimal_weights_generator(
            verbose = 0,
            range_generator=np.linspace,
            steps = 400,
            hidden_std = 0.5,
            find_optimal_input = False,
            thinning_step = 50,
        ))

        if("infer" in args.type): break

        if(args.controller == "transfer"): W_out_ = model.W_out_.copy()
        for i in range(20):
            if(args.controller == "transfer"): 
                model._update_via_transfer_learning(data[0, i, :400], mu=1e-2)
                p = model.predict(5000)
                model.W_out_ = W_out_.copy()
            else:
                model.update(data[ep, i, :500], controls[i, :500] if (args.controller != "transfer") else None, mode=update_modes.synchronization)
                p = model.predict(5000, np.ones((5000,1)) * controls[i, 0] if (args.controller != "transfer") else None)
            for j in range(5):
                w_errors[ep, i] += wasserstein_distance(data[ep, i,:,j], p[:,j])
            w_errors[ep, i] /= 5

        for i in range(20):
            if(args.controller == "transfer"): 
                model._update_via_transfer_learning(data[0, i, :400], mu=1e-2)
                f_errors[ep, i] = forecasting(model, data[ep, i, :2000])
                model.W_out_ = W_out_.copy()
            else:
                f_errors[ep, i] = forecasting(model, data[ep, i, :2000], controls[i, :2000] if (args.controller != "transfer") else None)

        pbar.update(1)

    postfix = f"{args.type.split('_')[-1]}_{args.controller}"
    if("train" in args.type):
        np.save(f"w_errors_{postfix}.npy", w_errors)
        np.save(f"f_errors_{postfix}.npy", f_errors)
        exit(0)
    else:
        w_errors = np.load(f"w_errors_{postfix}.npy")
        f_errors = np.load(f"f_errors_{postfix}.npy")

    
    # DRAW PREDICTED TIMESERIES
    w_error = np.median(w_errors, 0)
    f_error = np.median(f_errors, 0)
    w_std = np.std(w_errors, 0)
    f_std = np.std(f_errors, 0)
    w_maxs = np.sort(w_errors, axis=0)[-4]
    f_maxs = np.sort(f_errors, axis=0)[-4]

    plt.figure(figsize=(12,5))

    gs = GridSpec(10, 2, figure=plt.gcf())
    axs = [plt.gcf().add_subplot(gs[i*2:i*2+2, 0]) for i in range(5)]
    ax2 = plt.gcf().add_subplot(gs[:5, 1])
    ax3 = plt.gcf().add_subplot(gs[5:, 1])
    ax2.set_xticks([])

    if(args.controller == "transfer"): 
        W_out_ = model.W_out_.copy()
        model._update_via_transfer_learning(data[0, xi_idx_i, :400], mu=1e-2)
    model.update(data[0, xi_idx_i, :400], controls[xi_idx_i] if (args.controller != "transfer") else None, mode=update_modes.synchronization)
    output = model.predict(600, controls[xi_idx_i] if (args.controller != "transfer") else None)
    if(args.controller == "transfer"): model.W_out_ = W_out_.copy()

    D = data_orig[0, xi_idx_i]
    O = output.copy()

    for k in range(5):
        if(not (k == 1 and ("preproc_proc" in args.type or "preproc_log" in args.type))): 
            O[:, k] = O[:, k] * STD[k] + MED[k]
        if(k == 1 and "_proc" in args.type): O[:, k] = scaler.inverse_transform(O[:, k])
        if(k == 1 and "_log" in args.type): O[:, k] = np.exp(O[:, k] * STD1 + MED1)
            
    for i in range(5):
        if(i < 4): axs[i].set_xticks([])
        axs[i].plot(time[:401], D[:401,i], label="Synchronization")
        axs[i].plot(time[400:1000], O[:,i], label="Prediction")
        axs[i].plot(time[400:1000], D[400:1000,i], "--", label="Target")
        axs[i].set_ylabel("$%s$" % ("xyzuv"[i], ), rotation=0)

    axs[0].set_ylim(-4, 4)
    axs[1].set_ylim(-1, 4)
    axs[0].legend(loc = (0.02, 1.1), ncol=3)

    axs[4].set_xlabel("$t$")
    ax3.set_xlabel("$\\xi$")
    ax2.set_ylabel("Wasserstein m.")
    ax3.set_ylabel("Forecasting m.")

    rhos = np.linspace(1,40,20)
    ax3.set_xticks(list(range(0, 44, 4)))
    for i in range(2):
        ax = [ax2, ax3][i]
        E = [w_error, f_error][i]
        M = [w_maxs, f_maxs][i]
        ax.semilogy(rhos, E)
        ax.semilogy(rhos[ANC], E[ANC],"o",color="red")
        ax.semilogy(rhos[[xi_idx_i]], E[[xi_idx_i]],"o",color="green")
        ax.fill_between(rhos, 0, M, alpha = 0.25)

    plt.tight_layout()
    plt.savefig(figfolder + "predicted_timeseries.png", dpi=200)

    # DRAW LAMBDA FIGURE
    plt.figure(figsize=(5,4))
    if(args.controller == "transfer"): 
        W_out_ = model.W_out_.copy()
        model._update_via_transfer_learning(data[0, xi_idx_i, :400], mu=1e-2)
    model.update(data[0, xi_idx_i, :400], controls[xi_idx_i] if (args.controller != "transfer") else None, mode=update_modes.synchronization)
    predict = model.predict(5000, np.array([controls[xi_idx_i, 0, 0]] * 5000) if (args.controller != "transfer") else None)
    if(args.controller == "transfer"): model.W_out_ = W_out_.copy()

    D = data_orig[0, xi_idx_i]
    O = predict.copy()

    for k in range(5):
        if(not (k == 1 and ("preproc_proc" in args.type or "preproc_log" in args.type))): 
            O[:, k] = O[:, k] * STD[k] + MED[k]
        if(k == 1 and "_proc" in args.type): O[:, k] = scaler.inverse_transform(O[:, k])
        if(k == 1 and "_log" in args.type): O[:, k] = np.exp(O[:, k] * STD1 + MED1)

    tv = D[:, -1]
    pv = O[:, -1]

    idx = get_maximas(tv)
    plt.scatter(tv[idx[:-1]], tv[idx[1:]], alpha=0.5, label="true")

    idx = get_maximas(pv)
    plt.scatter(pv[idx[:-1]], pv[idx[1:]], alpha=0.5, label="predict")
    plt.xlabel("$v_{i}$")
    plt.ylabel("$v_{i+1}$")
    plt.tight_layout()
    plt.savefig(figfolder + "predicted_lambda.png", dpi=200)

    # DRAW PREDICTED DATA DISTRIBUTION
    plt.figure(figsize=(20,3))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.hist(D[:, i], 100, fc=(0, 0, 1, 0.5), density=True, label="true")
        plt.hist(O[:1000, i], 100, fc=(1, 0, 0, 0.5), density=True, label="predict")
        if(i == 0): plt.legend()
        plt.xlabel("xyzuv"[i])
    plt.tight_layout()
    plt.savefig(figfolder + "predicted_distribution.png", dpi=200)

    # time = np.arange(0, 200, 0.04)
    # data = np.load("data_chui.npy")