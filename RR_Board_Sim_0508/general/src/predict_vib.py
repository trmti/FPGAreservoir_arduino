import matplotlib.pyplot as plt
import numpy as np
import math
import optuna

import reservoir as rsv

##### 設定 #####
div_num = 3
wave_path = "./正常_30.csv"
wave_file = open(wave_path, "r")
error_wave_path = "./異常_30.csv"
error_wave_file = open(error_wave_path, "r")

time_dict = {"init": int(math.floor(2000/div_num)), "train": 2000, "eval": None, "total": None}

##### ファイル読み込み #####
dt_cnt = 0

row_data = wave_file.readline()
split_data = row_data.split(", ")
t = np.array([])
y = np.array([])
while row_data:
    index_dt = int(split_data[0])
    y_dt = float(split_data[1])
    t = np.append(t, index_dt)
    y = np.append(y, y_dt)
    dt_cnt += 1
    row_data = wave_file.readline()
    split_data = row_data.split(", ")
wave_file.close()

t = t[::div_num]
y = y[::div_num]

time_dict["eval"] = int(math.floor((t.shape[0] - time_dict["init"] - time_dict["train"])))
# time_dict["eval"] = 1000
time_dict["total"] = int(time_dict["init"] + time_dict["train"] + time_dict["eval"])

error_dt_cnt = 0

error_row_data = error_wave_file.readline()
error_split_data = error_row_data.split(", ")
error_t = np.array([])
error_y = np.array([])
while error_row_data:
    index_dt = int(error_split_data[0])
    y_dt = float(error_split_data[1])
    error_t = np.append(error_t, index_dt)
    error_y = np.append(error_y, y_dt)
    error_dt_cnt += 1
    error_row_data = error_wave_file.readline()
    error_split_data = error_row_data.split(", ")
error_wave_file.close()

error_t = error_t[::div_num]
error_y = error_y[::div_num]

####################################################################

def task_vib(time_dict, model):
    print("=====Predict vibration=====")
    init = time_dict["init"]
    train = time_dict["train"]
    eval = time_dict["eval"]
    total = time_dict["total"]

    input = y
    supervisor = np.roll(input, -1)

    model.run(input=input, init=init, train=train, eval=eval)

    model.fit_force(supervisor=supervisor)
    readout = model.predict()
    RMSE_predict = rsv.get_RMSE(readout=readout, supervisor=supervisor[init + train : total], name="RMSE_predict")

    return readout, supervisor, RMSE_predict

def objective(trial):
    param = {
        "num_nodes": trial.suggest_int("num_nodes", 100, 1000),
        "activation": "tanh",
        "minmax_w_in": (-1.0, 1.0),
        "w_res": trial.suggest_uniform("w_res", 0.5, 1.0),
        "K": trial.suggest_int("K", 1, 10),
        "bias": trial.suggest_uniform("bias", 0, 0.5),
        "seed": 0
    }
    model = rsv.RingReservoir(**param)
    _, _, RMSE_predict = task_vib(time_dict, model)
    return RMSE_predict

##### メイン #####

if __name__ == "__main__":
    params = {
        "num_nodes": 399,
        "activation": "pwl",
        "minmax_w_in": (-0.1, 0.1),
        "w_res": 0.8,
        "K": 5,
        "bias": 0.01
    }

    is_opt = False # パラメータを最適化するかどうか
    is_only_normal = False # 正常データのみを予測するかどうか
    is_MC = False # MCを計算するかどうか
    show_data_num = 300 # 表示するデータ数
    init_data_index = 300 # 初期データのインデックス

    if (is_opt):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100)
        print(study.best_params)
        params["num_nodes"] = study.best_params["num_nodes"]
        params["w_res"] = study.best_params["w_res"]
        params["K"] = study.best_params["K"]
        params["bias"] = study.best_params["bias"]
        
    np.random.seed(0)

    model = rsv.RingReservoir(**params)

    Fig = plt.figure()

    ##### 正常データの予測 #####
    readout, supervisor, _ = task_vib(time_dict, model)
    ax1 = Fig.add_subplot(2, 2, 1, xlabel="time", ylabel="Output value")

    # ax1.plot(readout[init_data_index:init_data_index + show_data_num], color="blue", label="output")
    ax1.plot(readout, color="blue", label="output")
    ax1.plot(supervisor[time_dict["init"] + time_dict["train"] : time_dict["total"]], color="red", alpha=0.5, label="supervisor")
    # ax1.plot(supervisor[time_dict["init"] + time_dict["train"] + init_data_index : time_dict["init"] + time_dict["train"] + init_data_index + show_data_num], color="red", alpha=0.5, label="supervisor")
    ax1.legend()

    ##### 異常データの予測 #####
    if (not is_only_normal):
        model.run(input=error_y, init=time_dict["init"], train=time_dict["train"], eval=time_dict["eval"])
        readout_error = model.predict()
        rsv.get_RMSE(readout=readout_error, supervisor=np.roll(error_y, -1)[time_dict["init"]+time_dict["train"]: time_dict["total"]], name="RMSE_predict_error")
        ax2 = Fig.add_subplot(2, 2, 2, xlabel="time", ylabel="Output value")

        # ax2.plot(readout_error[init_data_index:init_data_index + show_data_num], color="blue", label="output")
        ax2.plot(readout_error, color="blue", alpha=0.5, label="output")
        # ax2.plot(np.roll(error_y, -1)[time_dict["init"] + time_dict["train"] + init_data_index : time_dict["init"] + time_dict["train"] + init_data_index + show_data_num], color="red", alpha=0.5, label="supervisor")
        ax2.plot(np.roll(error_y, -1)[time_dict["init"] + time_dict["train"] : time_dict["total"]], color="red", alpha=0.5, label="supervisor")
        ax2.legend()

    ##### MCの計算 #####
    if (is_MC):
        print("==========MC==========")
        MC, MC_tau = model.get_MC(init=time_dict["init"], train=time_dict["train"], eval=time_dict["eval"])
        ax3 = Fig.add_subplot(2, 2, 3, xlabel=r"$\tau$", ylabel=r"$MC_{\tau}$")
        ax3.plot(MC_tau, color="blue")

    #### 異常度の計算 #####
    print("========== Calculate anomaly ==========")
    pile_num = 100
    error = np.abs(error_y[time_dict["init"] + time_dict["train"] : time_dict["total"]] - readout)
    Fig2 = plt.figure()
    ax2_1 = Fig2.add_subplot(1, 2, 1, xlabel="time", ylabel="Anomaly")
    ax2_1.plot(error, color="blue", label="normal")

    error_piled = np.array([np.sum(error[0:pile_num])])

    for i, e in enumerate(error[pile_num:]):
        if (i < time_dict["eval"] - pile_num):
            error_piled = np.append(error_piled, error_piled[i] + e - error[i])
        else:
            break
    
    ax2_2 = Fig2.add_subplot(1, 2, 2, xlabel="time", ylabel="Anomaly")
    ax2_2.plot(error_piled, color="blue", label="normal")

    plt.show()
