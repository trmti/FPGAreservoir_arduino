import numpy as np
import matplotlib.pyplot as plt
import csv

if __name__ == '__main__':
    start = 100
    N = 100
    N_freq = 2000
    with open('正常.csv') as f:
        reader = csv.reader(f)
        data_normal = [float(row[1]) for row in reader]
    with open('異常.csv') as f:
        reader = csv.reader(f)
        data_error = [float(row[1]) for row in reader]

    # 高速フーリエ変換(FFT) 通常
    F_normal = np.fft.fft(data_normal[start:N_freq+start])
    F_abs_normal = np.abs(F_normal)

    # 高速フーリエ変換(FFT) 異常
    F_error = np.fft.fft(data_error[start:N_freq+start])
    F_abs_error = np.abs(F_error)

    fig = plt.figure(figsize = (10,6))
    # 通常時の出力
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(data_normal[start:N+start], label="normal", color="blue")
    ax1.legend(loc = 'upper right')
    ax1.set_ylim(0.05, 0.3)

    # 通常時の周波数成分
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.plot(F_abs_normal[5:int(N_freq/2)+1], label="normal_freq", color="blue")
    ax2.legend(loc = 'upper right')
    
    # 異常時の出力
    ax3 = fig.add_subplot(2, 2, 2)
    ax3.plot(data_error[start:N+start], label="error", color="red")
    ax3.legend(loc = 'upper right')
    ax3.set_ylim(0.05, 0.3)

    # 異常時の周波数成分
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(F_abs_error[5:int(N_freq/2)+1], label="error_freq", color="red")
    ax4.legend(loc = 'upper right')

    plt.show()