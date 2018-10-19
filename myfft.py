import numpy as np


def myfft1d(f):
    N = f.shape[0]
    if N <= 1:
        return f

    elif N % 2 >0:
        print("length of input array is not a power of 2")
    else:
        u = np.arange(0, N)
        N_by2 = int(N / 2)
        even_ft = myfft1d(f[::2])
        odd_ft = myfft1d(f[1::2])
        exp_factor2 = np.exp(-2j * np.pi * u / N)
        return np.concatenate([even_ft + exp_factor2[:N_by2] * odd_ft, even_ft + exp_factor2[N_by2:] * odd_ft])


def myifft1d(f):
    N = f.shape[0]
    if N <= 1:
        return f
    elif N % 2 > 0:
        print("length of input array is not a power of 2")
    else:
        u = np.arange(0, N)
        N_by2 = int(N / 2)
        even_ft = myifft1d(f[::2])
        odd_ft = myifft1d(f[1::2])
        exp_factor2 = np.exp(2j * np.pi * u / N)
        return np.concatenate([even_ft + exp_factor2[:N_by2] * odd_ft, even_ft + exp_factor2[N_by2:] * odd_ft])

f = [0,1,2,3,4,5,6,7,8,9,10,11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
f = np.asarray(f)
f_fft = myfft1d(f)
ffft = np.fft.fft(f)
print(np.allclose(f_fft, ffft))