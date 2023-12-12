import csv
import numpy as np
import matplotlib.pyplot as plt

def make_sin(A, f, sec, N):
    """
    A: 振幅
    f: 周波数[Hz]
    sec: 信号の長さ[s]
    N: データ数
    """

    file_name = "hakei_sin.CSV"

    t = np.arange(0, sec, sec/N)
    y = A * np.sin(2*np.pi*f*t)

    res = [t, y]

    with open(f'./{file_name}', "w") as f:
        writer = csv.writer(f)
        writer.writerows(res)

    return res

if __name__ == "__main__":
   res = make_sin(1.0, 500, 1.0, 10000)
   plt.plot(res[0][:100], res[1][:100])
   plt.show()
