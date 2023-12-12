import sys

import numpy as np
from scipy.special import eval_legendre
from tqdm import tqdm


class ESN:
    def __init__(self, num_nodes, activation, minmax_w_in, minmax_w_res, network_densit=1.0, seed=0):
        """
        num_nodes : number of node
        activation : activation function ('tanh', 'pwl' or 'quadratic')
        minmax_w_in : (min, max) wight for input
        minmax_w_res : (min, max) wight for reservoir
        seed : seed for numpy
        """
        self.num_nodes = num_nodes
        self.activation = activation
        self.minmax_w_in = minmax_w_in
        self.minmax_w_res = minmax_w_res
        self.seed = seed
        np.random.seed(seed=seed)

        if activation == "tanh":
            self.act_func = np.tanh
        elif activation == "pwl":
            self.act_func = self.pwl
        elif activation == "quadratic":
            self.act_func = self.quadratic
        else:
            sys.exit("The activation is not existed in this class.")

        self.w_in = np.random.uniform(minmax_w_in[0], minmax_w_in[1], num_nodes)
        w_esn = np.random.uniform(minmax_w_res[0], minmax_w_res[1], (num_nodes + 1, num_nodes + 1))
        sparse_index = np.random.uniform(0, 1, (num_nodes + 1, num_nodes + 1))
        # sparse_index = np.random.uniform(0, 1, (num_nodes, num_nodes))
        sparse_index = np.where(sparse_index <= network_densit, 1, 0)
        # sparse_index = np.hstack([sparse_index, np.ones(num_nodes, 1)])
        # sparse_index = np.vstack([sparse_index, np.ones(1, num_nodes)])
        w_sps = w_esn * sparse_index
        w_eig = np.linalg.eig(w_sps)[0]
        w_eig_abs = np.abs(w_eig)
        w_eig_max = np.max(w_eig_abs)
        self.w_res = 0.95 * (w_esn / w_eig_max)  # spectral radius = 0.9

    # run the reservoir
    def run(self, input, init, train, eval):
        """
        Updating nodes and return 'matrix_x' {total time * (number of nodes+1)}
        """
        self.init = init
        self.train = train
        self.eval = eval
        self.total = init + train + eval
        x = np.random.uniform(-1, 1, self.num_nodes + 1)
        x[-1] = 1.0  # bias
        matrix_x = np.zeros((self.total, self.num_nodes + 1))
        print("--------running the reservoir--------")
        for t in tqdm(range(self.total)):
            x[:-1] = self.act_func(np.dot(self.w_res, x)[:-1] + self.w_in * input[t])
            matrix_x[t, :] = x
        self.matrix_x = matrix_x
        return matrix_x

    def pwl(self, x, slope=1):
        threshold = 1 / slope
        set_max = np.where(x < threshold, slope * x, 1)
        set_min = np.where(set_max > -1, set_max, -1)
        return set_min

    def quadratic(self, x):
        converted = np.where(x > 0, 1 - (1 - x) ** 2, (1 + x) ** 2 - 1)
        set_max = np.where(x < 1, converted, 1)
        set_min = np.where(x > -1, set_max, -1)
        return set_min

    def fit_batch(self, supervisor):
        """
        Training and updating output-wights reffering to 'matrix_x[init:init+train, :]'
        """
        X = self.matrix_x[self.init : self.init + self.train, :]
        Y = supervisor[self.init : self.init + self.train]
        self.w_out = np.linalg.pinv(X).dot(Y)

    def fit_force(self, supervisor):
        """
        Training and updating output-wight reffering to 'matrix_x[init+train:, :]'
        (return 'fit_log', readout value during training)
        """
        w_out = np.random.uniform(-1 / self.num_nodes, 1 / self.num_nodes, self.num_nodes + 1)
        p = np.eye(self.num_nodes + 1)
        pr = np.zeros(self.num_nodes + 1)
        rtp = np.zeros(self.num_nodes + 1)
        prrtp = np.zeros((self.num_nodes + 1, self.num_nodes + 1), "float")
        fit_log = np.zeros(self.train)
        print("--------leaning by FORCE--------")
        for t in tqdm(range(self.init, self.init + self.train)):
            readout = np.dot(w_out, self.matrix_x[t, :])  # scalar
            e = readout - supervisor[t]  # scalar
            pr = np.dot(p, self.matrix_x[t, :])  # vector
            rtp = np.dot(self.matrix_x[t, :], p)  # vector
            prrtp = np.outer(pr, rtp)  # matrix
            rtpr = np.dot(rtp, self.matrix_x[t, :])  # scalar
            p = p - prrtp / (1 + rtpr)  # matrix
            w_out = w_out - e * np.dot(p, self.matrix_x[t, :])  # vector
            fit_log[t - self.init] = readout
        self.w_out = w_out
        return fit_log

    def predict(self):
        """
        Return Predicted value reffering to updated output-wight and matrix_x[init+train:, :]
        """
        return self.matrix_x[self.init + self.train :, :].dot(self.w_out)

    def get_MC(self, init, train, eval):
        """
        Return 'MC' and 'MC_tau'
        """
        total = init + train + eval
        input_MC = np.random.uniform(-1, 1, total)
        self.run(input=input_MC, init=init, train=train, eval=eval)
        MC_tau = np.zeros(100)
        MC = 0
        print("--------calculating MC--------")
        for i in tqdm(range(100)):
            supervisor_MC = np.roll(input_MC, i)
            self.fit_batch(supervisor=supervisor_MC)
            readout_MC = self.predict()
            MC_tau[i] = (np.cov(readout_MC, supervisor_MC[init + train :])[0, 1] ** 2) / (
                np.var(readout_MC) * np.var(supervisor_MC[init + train :])
            )
            MC += MC_tau[i]
        print("MC = {:.1f}".format(MC))
        return MC, MC_tau

    def get_IPC(self, ratio=1):
        """
        ratio : ratio for shrinking {initital, training, evaluation} time ('time' * 'ratio') \n
        Return 'Total Capacity', 'Linear Capacity', 'Non-Linear Capacity' and 'Each Degree Capacity'
        """
        init = 40000 * ratio
        train = 100000 * ratio
        eval = 10000 * ratio
        total = init + train + eval
        input_IPC = np.random.uniform(-1, 1, total)
        self.run(input=input_IPC, init=init, train=train, eval=eval)
        epsilon = 10.0 ** (-2.0)  # 1.7 * (10**(-4))
        N = self.num_nodes + 1
        di = np.zeros(N)
        di_out = di.reshape((1, N))
        di[0] = 1
        C_T_out = np.zeros(1)
        carry = 1
        j = 0
        BF = 0  # break flag
        odd_flag = 0
        even_flag = 0
        while BF == 0:
            print(di)
            di_out = np.append(di_out, di.reshape(1, N), axis=0)
            supervisor = np.ones(total)
            for i in range(N):
                delay_in = np.roll(input_IPC, i)
                supervisor *= eval_legendre(di[i], delay_in)  # supervisor = pi P_di[i]
            # eval
            X = np.linalg.pinv(self.matrix_x[init : init + train, :])
            w_out = np.dot(X, supervisor[init : init + train])
            readout = np.dot(self.matrix_x[init + train :, :], w_out)
            supervisor_ave = np.sum(supervisor[init + train :] ** 2.0) / eval
            MSE = np.sum((readout - supervisor[init + train :]) ** 2.0) / eval
            C_T = 1.0 - (MSE / supervisor_ave)  # C_T[X,{di}]
            print(C_T)
            C_T_out = np.append(C_T_out, C_T)
            if C_T - epsilon < 0:
                # 奇数次数、偶数次数
                if np.sum(di) % 2 == 0:
                    even_flag = 1
                else:
                    odd_flag = 1
                # di：桁上がり [...K, L, 0,...] -> [...0, L+1, 0,...] 合図
                if even_flag == 1 and odd_flag == 1:
                    carry = 0
                    even_flag = 0
                    odd_flag = 0
                    if di[0] == 1 and np.sum(di) == 2:  # [1, 0, 0, 0, ..., 1, 0, 0, ...]
                        BF = 1
                else:
                    di[0] += 1  # di：[K,...] -> [K+1,...]
            else:
                di[0] += 1  # di：[K,...] -> [K+1,...]
            # di：桁上がり [...K, L, 0,...] -> [...0, L+1, 0,...]
            while carry == 0:
                if j == 0 and di[j] == 1:
                    j += 1
                elif j != 0 and di[j] == 0:
                    j += 1
                else:
                    di[0] = 0
                    di[j] = 0
                    di[j + 1] += 1
                    j = 0
                    carry = 1
        C = C_T_out
        di = di_out
        N = self.num_nodes
        epsilon = 10 ** (-2)
        L = 0
        NL = 0
        number_of_C = C.shape[0]
        C_TOT = 0
        C_di = np.zeros(10)
        for i in range(number_of_C):
            if C[i] > epsilon:
                C_TOT += C[i]
                di_sum = np.sum(di[i, :])
                NL += di_sum * C[i] / N
                if di_sum == 1:
                    L += C[i] / N
                di_sum = int(np.sum(di[i, :]))
                if di_sum == 1:
                    C_di[di_sum] += C[i] / N
                elif di_sum == 2:
                    C_di[di_sum] += C[i] / N
                elif di_sum == 3:
                    C_di[di_sum] += C[i] / N
                elif di_sum == 4:
                    C_di[di_sum] += C[i] / N
                elif di_sum == 5:
                    C_di[di_sum] += C[i] / N
                elif di_sum == 6:
                    C_di[di_sum] += C[i] / N
                elif di_sum == 7:
                    C_di[di_sum] += C[i] / N
                elif di_sum == 8:
                    C_di[di_sum] += C[i] / N
                elif di_sum == 9:
                    C_di[di_sum] += C[i] / N
        print("Total capacity: ", C_TOT)
        print("Linear memory capacity: ", L)
        print("Non-linear memory capacity: ", NL)
        for j in range(10):
            print("di_sum = ", j, " : ", C_di[j] * N)
        return C_TOT, L, NL, C_di[j] * N

    def profile(self):
        print(f"num_nodes:{self.num_nodes}")
        print(f"activation:{self.activation}")
        print(f"minmax_w_in:{self.minmax_w_in}")
        print(f"minmax_w_res:{self.minmax_w_res}")
        print(f"seed:{self.seed}")
        print(f"init:{self.init}")
        print(f"train:{self.train}")
        print(f"eval:{self.eval}")
        print(f"total:{self.total}")

    def get_profile(self):
        prof_dict = {
            "num_nodes": self.num_nodes,
            "activation": self.activation,
            "minmax_w_in": self.minmax_w_in,
            "minmax_w_res": self.minmax_w_res,
            "seed": self.seed,
            "init": self.init,
            "train": self.train,
            "eval": self.eval,
            "total": self.total,
        }
        return prof_dict


class RingReservoir(ESN):
    def __init__(self, num_nodes, activation, minmax_w_in, w_res, K, bias, seed=0):
        """
        num_nodes : number of nodes
        activation : activation function ('tanh', 'pwl' or 'quadratic')
        minmax_w_in : {min, max} wight for input
        w_res : wight for reservoir
        K : number of additional nodes for virtual nodes
        bias : bias for reservoir
        seed : seed for numpy
        """
        self.num_nodes = num_nodes
        self.activation = activation
        self.minmax_w_in = minmax_w_in
        self.w_res = w_res
        self.K = K
        self.bias = bias
        self.seed = seed
        np.random.seed(seed=seed)

        if activation == "tanh":
            self.act_func = np.tanh
        elif activation == "pwl":
            self.act_func = self.pwl
        elif activation == "quadratic":
            self.act_func = self.quadratic
        else:
            sys.exit("the activation is not existed in this class")

        self.w_in = np.random.uniform(minmax_w_in[0], minmax_w_in[1], num_nodes)

    # run the reservoir
    def run(self, input, init, train, eval):
        """
        Updating nodes and return 'matrix_x' {total time * (number of nodes+1)}
        """
        self.init = init
        self.train = train
        self.eval = eval
        self.total = init + train + eval
        x = np.random.uniform(-1, 1, self.num_nodes + 1)
        x[-1] = 1.0  # bias
        matrix_x = np.zeros((self.total, self.num_nodes + 1))
        x_vir = np.random.uniform(-1, 1, (self.num_nodes + self.K))
        print("--------running reservoir--------")
        for t in tqdm(range(self.total)):
            x[: self.num_nodes] = self.act_func(self.w_res * x_vir[: self.num_nodes] + self.w_in * input[t]) + self.bias
            x_vir = np.roll(x_vir, self.K)
            x_vir[self.K :] = x[: self.num_nodes]
            matrix_x[t, :] = x
        self.matrix_x = matrix_x
        return matrix_x

    def print_profile(self):
        print(f"num_nodes:{self.num_nodes}")
        print(f"activation:{self.activation}")
        print(f"minmax_w_in:{self.minmax_w_in}")
        print(f"w_res:{self.w_res}")
        print(f"K:{self.K}")
        print(f"bias:{self.bias}")
        print(f"seed:{self.seed}")
        print(f"init:{self.init}")
        print(f"train:{self.train}")
        print(f"eval:{self.eval}")
        print(f"total:{self.total}")

    def get_profile(self):
        prof_dict = {
            "num_nodes": self.num_nodes,
            "activation": self.activation,
            "minmax_w_in": self.minmax_w_in,
            "w_res": self.w_res,
            "K": self.K,
            "bias": self.bias,
            "seed": self.seed,
            "init": self.init,
            "train": self.train,
            "eval": self.eval,
            "total": self.total,
        }
        return prof_dict


class MultiInputReservoir(ESN):
    def __init__(
        self, num_nodes, num_inputs, num_outputs, activation, minmax_w_in, minmax_w_res, network_densit=1.0, seed=0
    ):
        """
        num_nodes : number of node
        num_inputs : number of inputs
        num_outputs : number of outputs
        activation : activation function ('tanh', 'pwl' or 'quadratic')
        minmax_w_in : (min, max) wight for input
        minmax_w_res : (min, max) wight for reservoir
        seed : seed for numpy
        """
        super().__init__(num_nodes, activation, minmax_w_in, minmax_w_res, network_densit, seed)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.w_in = np.random.uniform(minmax_w_in[0], minmax_w_in[1], (num_nodes, num_inputs))

    # run the reservoir
    def run(self, input, init, train, eval):
        """
        Updating nodes and return 'matrix_x' {total time * (number of nodes+1)}
        """
        self.init = init
        self.train = train
        self.eval = eval
        self.total = init + train + eval
        x = np.random.uniform(-1, 1, self.num_nodes + 1)
        x[-1] = 1.0  # bias
        matrix_x = np.zeros((self.total, self.num_nodes + 1))
        print("--------running the reservoir--------")
        for t in tqdm(range(self.total)):
            x[:-1] = self.act_func(np.dot(self.w_res, x)[:-1] + np.dot(self.w_in, input[t, :]))
            matrix_x[t, :] = x
        self.matrix_x = matrix_x
        return matrix_x

    def fit_batch(self, supervisor):
        """
        Training and updating output-wights reffering to 'matrix_x[init:init+train, :]'
        """
        X = self.matrix_x[self.init : self.init + self.train, :]
        Y = supervisor[self.init : self.init + self.train, :]
        self.w_out = np.linalg.pinv(X).dot(Y)  # (num_nodes+1, num_outputs)

    def predict(self):
        """
        Return Predicted value reffering to updated output-wight and matrix_x[init+train:, :]
        """
        return self.matrix_x[self.init + self.train :, :].dot(self.w_out)  # (eval, num_outputs)

    def print_profile(self):
        super().print_profile()
        print(f"num_inputs:{self.num_inputs}")
        print(f"num_outputs:{self.num_outputs}")

    def get_profile(self):
        prof_dict = super().get_profile()
        prof_dict["num_inputs"] = self.num_inputs
        prof_dict["num_outputs"] = self.num_outputs
        return prof_dict


def gen_NARMA10(input):
    """
    Generating NARMA10 list for surpervisor
    """
    Total = len(input)
    supervisor = np.zeros(Total)
    for t in range(Total):
        if t > 8:
            supervisor[t] = (
                0.3 * supervisor[t - 1]
                + 0.05 * supervisor[t - 1] * sum(supervisor[t - 10 : t])
                + 1.5 * input[t - 1] * input[t - 10]
                + 0.1
            )
        else:
            supervisor[t] = input[t]
    return supervisor


def get_NRMSE(readout, supervisor, name="NRMSE"):
    """
    Calculating NRMSE
    """
    T = len(readout)
    MSE = np.sum(np.power(readout - supervisor, 2)) / T
    RMSE = np.sqrt(MSE)
    # NRMSE = RMSE / np.mean(supervisor)
    NRMSE = RMSE / (np.max(supervisor) - np.min(supervisor))
    print("{} = {:.4f}".format(name, NRMSE))
    return NRMSE

def get_RMSE(readout, supervisor, name="NRMSE"):
    """
    Calculating NRMSE
    """
    T = len(readout)
    MSE = np.sum(np.power(readout - supervisor, 2)) / T
    RMSE = np.sqrt(MSE)
    print("{} = {:.4f}".format(name, RMSE))
    return RMSE

class Q:  # quantization
    def __init__(self, WIDTH, EN=True, permit_more_than_1=False):
        self.WIDTH = WIDTH
        self.EN = EN
        self.permit_more_than_1 = permit_more_than_1

    def __call__(self, value, name=""):
        if not self.permit_more_than_1:
            if np.any(value == 1):
                value = np.where(value == 1, value - 2 ** (-(self.WIDTH - 1)), value)
            if np.any(value > 1):
                # sys.exit(f'{name, self.WIDTH} includes a element more than 1')
                print(f"{name, self.WIDTH} includes a element more than 1")
                value = np.where(value > 1, value - 2, value)
            if np.any(value <= -1):
                # sys.exit(f'{name, self.WIDTH} includes a element less than -1')
                print(f"{name, self.WIDTH} includes a element less than -1")
                value = np.where(value <= -1, value + 2, value)
        if self.EN:
            floored_value = np.floor(value * (2 ** (self.WIDTH - 1)))
            out = floored_value * (2 ** (-(self.WIDTH - 1)))
        else:
            out = value
        return out


def to_bin8(value):
    value_dec = np.floor(value * (2 ** 7)).astype("int")
    return "{:08b}".format(value_dec & 0xFF)


def to_bin16(value):
    value_dec = np.floor(value * (2 ** 15)).astype("int")
    return "{:016b}".format(value_dec & 0xFFFF)


def to_bin26(value):
    value_dec = np.floor(value * (2 ** 25)).astype("int")
    return "{:026b}".format(value_dec & 0x03FFFFFF)


def to_bin15_plus10int(value):
    value_dec = np.floor(value * (2 ** 15)).astype("int")
    return "{:025b}".format(value_dec & 0x01FFFFFF)


def my_zfill8(value) -> str:
    value_str = str(value)
    return value_str.zfill(8)


def my_zfill16(value) -> str:
    value_str = str(value)
    return value_str.zfill(16)


def my_zfill26(value) -> str:
    value_str = str(value)
    return value_str.zfill(26)


def my_zfill25(value) -> str:
    value_str = str(value)
    return value_str.zfill(25)


def to_dec_from_bin(value) -> float:
    value_int = -int(value[0]) << len(value) | int(value, 2)
    return value_int / (2 ** (len(value) - 1))


def to_dec_from_bin15_int10(value) -> float:
    value_int = -int(value[0]) << len(value) | int(value, 2)
    return value_int / (2 ** 15)


def to_hex2(value):
    value_dec = np.floor(value * (2 ** 7)).astype("int")
    value_str = "{:02x}".format(value_dec & 0xFF)
    value_hex = value_str.upper()
    return value_hex


def to_hex4(value):
    value_dec = np.floor(value * (2 ** 15)).astype("int")
    value_str = "{:04x}".format(value_dec & 0xFFFF)
    value_hex = value_str.upper()
    return value_hex
