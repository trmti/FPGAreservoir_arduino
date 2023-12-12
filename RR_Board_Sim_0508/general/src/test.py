import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

import reservoir as rsv

####波形CSVファイル読み込み###########
wave_path = "./hakei2.CSV"
wave_file = open(wave_path,'r')
skip_row_count = 0

#ヘッダ行等のスキップ
while skip_row_count !=0:
    row_data = wave_file.readline()
    skip_row_count-=1

row_data = wave_file.readline()
split_data = row_data.split(",")
dt_cnt=0
t = np.array([])
CH1 = np.array([])

while row_data:
    index_dt = float(split_data[3])
    #ch1_dt = (float(split_data[4])+0.5)/1.5
    ch1_dt = float(split_data[4])/1.5+0.5
    t = np.append(t, index_dt)
    CH1 = np.append(CH1, ch1_dt)
    #ch2_dt = float(split_data[2])
    ###########この間に解析処理を入れる##########

    #######################################
    ##次の行を読み込む。ここで最終行なら空白が返ってくるためループ終了
    dt_cnt+=1
    row_data = wave_file.readline()
    split_data = row_data.split(",")
wave_file.close()

####################################################################

def task_narma10(path_results, time_dict, model, training_method):
    print("========NARMA10========")
    os.makedirs(f"{path_results}/NARMA10", exist_ok=True)
    init = time_dict["init"]
    train = time_dict["train"]
    eval = time_dict["eval"]
    total = time_dict["total"]

    input = np.random.uniform(0, 0.5, total)
    supervisor = rsv.gen_NARMA10(input=input)
    #supervisor = CH1
    matrix_x = model.run(input=input, init=init, train=train, eval=eval)

    def batch():
        model.fit_batch(supervisor=supervisor)
        readout = model.predict()
        NRMSE = rsv.get_NRMSE(readout=readout, supervisor=supervisor[init + train :], name="NRMSE")
        np.savez(f"{path_results}/NARMA10/data", input, supervisor, matrix_x, readout)
        save_data(f"{path_results}/NARMA10", model, {"NRMSE": NRMSE})
        ax1.plot(readout, color="blue", label="output")
        ax1.plot(supervisor[init + train :], color="red", alpha=0.5, label="supervisor")

    def force():
        fit_log = model.fit_force(supervisor=supervisor)
        readout = model.predict()
        NRMSE_leaning = rsv.get_NRMSE(readout=fit_log, supervisor=supervisor[init : init + train], name="NRMSE_leaning")
        NRMSE_predict = rsv.get_NRMSE(readout=readout, supervisor=supervisor[init + train :], name="NRMSE_predict")
        np.savez(f"{path_results}/NARMA10/data", input, supervisor, matrix_x, fit_log, readout)
        save_data(f"{path_results}/NARMA10", model, {"NRMSE_leaning": NRMSE_leaning, "NRMSE_predict": NRMSE_predict})
        ax1.plot(np.hstack((fit_log, readout)), color="blue", label="output")
        ax1.plot(supervisor[init:], color="red", alpha=0.5, label="supervisor")

    ax1 = Fig.add_subplot(2, 2, 1, xlabel="time", ylabel="Output value")
    if training_method == "batch":
        batch()
    elif training_method == "force":
        force()
    else:
        sys.exit("The training_method is not existed")
    ax1.set_ylim([0, 1])
    ax1.legend()

    with open(f"{path_results}/model.pickle", mode="wb") as f:
        pickle.dump(model, f)


def task_mc(path_results, model):
    print("==========MC==========")
    os.makedirs(f"{path_results}/MC", exist_ok=True)
    MC, MC_tau = model.get_MC(init=time_dict["init"], train=time_dict["train"], eval=time_dict["eval"])
    save_data(f"{path_results}/MC", model, {"MC": MC})
    ax2 = Fig.add_subplot(2, 2, 2, xlabel=r"$\tau$", ylabel=r"$MC_{\tau}$")
    ax2.plot(MC_tau, color="blue")


def task_ipc(path_results, model):
    print("==========IPC==========")
    os.makedirs(f"{path_results}/IPC", exist_ok=True)
    IPC_total, IPC_linear, IPC_nonlinear, IPC_di = model.get_IPC()
    result_dict = {"IPC_total": IPC_total, "IPC_linear": IPC_linear, "IPC_nonlinear": IPC_nonlinear, "IPC_di": IPC_di}
    save_data(f"{path_results}/IPC", model, result_dict)
    x_axis = np.array([1, 2, 3, 4, 5])
    ax3 = Fig.add_subplot(2, 2, 3, xlabel=r"degree", ylabel=r"capacity")
    ax3.bar(x_axis, IPC_di[: len(x_axis)], align="center")
    ax3.set_ylim(0, 200)
    plt.grid(True)


def save_data(path_results, model, result_dict):
    prof_dict = model.get_profile()
    with open(f"{path_results}/results.txt", "w") as f:
        f.write("-----Parameters-----\n")
        for key, value in prof_dict.items():
            f.write(f"{key}: {value}\n")
        f.write("\n-----Results-----\n")
        for key, value in result_dict.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    np.random.seed(0)
    path_results = "../out/temp"
    os.makedirs(path_results, exist_ok=True)

    #time_dict = {"init": 100, "train": 1500, "eval":900, "total": None}
    time_dict = {"init": 2000, "train": 3000, "eval": 1000, "total": None}
    time_dict["total"] = time_dict["init"] + time_dict["train"] + time_dict["eval"]

    #model = rsv.ESN(num_nodes=399, activation="tanh", minmax_w_in=(-0.1, 0.1), minmax_w_res=(-0.1, 0.1), seed=0)
    #model = rsv.RingReservoir(num_nodes=499, activation="tanh", minmax_w_in=(-0.1, 0.1), w_res=1.03, K=2, bias = 0, seed=0)

    model = rsv.RingReservoir(
             num_nodes=499,
             activation='tanh',
             minmax_w_in=(-0.1, 0.1),
             w_res=0.8,
             K = 5,
             bias = 0.01)

    training_method = "force"  # ('batch' or 'force')
    task_activation = {"NARMA10": True, "MC": True, "IPC": False}

    Fig = plt.figure()
    if task_activation["NARMA10"]:
        task_narma10(path_results, time_dict, model, training_method)
    if task_activation["MC"]:
        task_mc(path_results, model)
    if task_activation["IPC"]:
        task_ipc(path_results, model)
    plt.show()
