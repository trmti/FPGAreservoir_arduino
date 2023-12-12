import matplotlib.pyplot as plt
import numpy as np
import optuna

import reservoir as rsv
from make_waves import make_sin

### sin wave params
A = 1
f = 300
sec = 0.5

####################################################################

def task_sin(time_dict, model):
    print("=====Predict sin=====")
    init = time_dict["init"]
    train = time_dict["train"]
    eval = time_dict["eval"]
    total = time_dict["total"]

    sin_wave = make_sin(A, f, sec, total)

    input = sin_wave[1]
    supervisor = np.roll(input, -1)

    model.run(input=input, init=init, train=train, eval=eval)

    fit_log = model.fit_batch(supervisor=supervisor)
    readout = model.predict()
    # NRMSE_leaning = rsv.get_NRMSE(readout=fit_log, supervisor=supervisor[init : init + train], name="NRMSE_leaning")
    rsv.get_NRMSE(readout=readout, supervisor=supervisor[init + train :], name="NRMSE_predict")
    ax1 = Fig.add_subplot(2, 2, 1, xlabel="time", ylabel="Output value")

    ax1.plot(readout, color="blue", label="output")
    ax1.plot(supervisor[init + train :], color="red", alpha=0.5, label="supervisor")

    ax1.legend()

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

    init = 1000
    train = 3000
    eval = 1000
    total = init + train + eval

    sin_wave = make_sin(A, f, sec, total)
    input = sin_wave[1]
    supervisor = np.roll(input, -1)

    model.run(input=input, init=init, train=train, eval=eval)

    fit_log = model.fit_batch(supervisor=supervisor)
    readout = model.predict()
    # NRMSE_leaning = rsv.get_NRMSE(readout=fit_log, supervisor=supervisor[init : init + train], name="NRMSE_leaning")
    NRMSE_predict = rsv.get_NRMSE(readout=readout, supervisor=supervisor[init + train :], name="NRMSE_predict")
    
    return NRMSE_predict

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
    params = {
        "num_nodes": 499,
        "activation": "tanh",
        "minmax_w_in": (-0.1, 0.1),
        "w_res": 0.8,
        "K": 5,
        "bias": 0.01
    }
    is_opt = True

    if (is_opt):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100)
        print(study.best_params)
        params["num_nodes"] = study.best_params["num_nodes"]
        params["w_res"] = study.best_params["w_res"]
        params["K"] = study.best_params["K"]
        params["bias"] = study.best_params["bias"]
        
    np.random.seed(0)

    time_dict = {"init": 1000, "train": 3000, "eval": 1000, "total": None}
    time_dict["total"] = time_dict["init"] + time_dict["train"] + time_dict["eval"]

    model = rsv.RingReservoir(**params)

    Fig = plt.figure()
    task_sin(time_dict, model)
    plt.show()
