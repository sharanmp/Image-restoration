import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.measure import compare_ssim as SSIM


# ---------------------ALL LINES ARE WRITTEN BY ME-----------------------------------
def constrained_LS_filter(blurred_image,kernel, gamma):
    # fftType = 0 for faster fft; = 1 for implemented (slower) fft
    fftType =0
    M, N, _ = blurred_image.shape
    deblurred_image = np.zeros([M, N, 3], np.complex128)
    if fftType == 0:
        fft_pad_b_image_B = np.fft.fft2(blurred_image[:, :, 0])
        fft_pad_b_image_G = np.fft.fft2(blurred_image[:, :, 1])
        fft_pad_b_image_R = np.fft.fft2(blurred_image[:, :, 2])
        fft_kernel_B = np.fft.fft2(kernel[:, :, 0], s = (M, N))
        fft_kernel_G = np.fft.fft2(kernel[:, :, 1], s = (M, N))
        fft_kernel_R = np.fft.fft2(kernel[:, :, 2], s = (M, N))

        p_xy = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        fft_pad_p_xy = np.fft.fft2(p_xy, s = (M,N))

        factor_B = np.divide(np.conjugate(fft_kernel_B), gamma*(abs(fft_pad_p_xy))**2 + (abs(fft_kernel_B)) ** 2)
        factor_G = np.divide(np.conjugate(fft_kernel_G), gamma*(abs(fft_pad_p_xy))**2 + (abs(fft_kernel_G)) ** 2)
        factor_R = np.divide(np.conjugate(fft_kernel_R), gamma*(abs(fft_pad_p_xy))**2 + (abs(fft_kernel_R)) ** 2)

        deblurred_image[:, :, 0] = np.fft.ifft2(np.multiply(fft_pad_b_image_B, factor_B))
        deblurred_image[:, :, 1] = np.fft.ifft2(np.multiply(fft_pad_b_image_G, factor_G))
        deblurred_image[:, :, 2] = np.fft.ifft2(np.multiply(fft_pad_b_image_R, factor_R))

    if fftType == 1:
        fft_pad_b_image_B = myfft2d(blurred_image[:, :, 0])
        fft_pad_b_image_G = myfft2d(blurred_image[:, :, 1])
        fft_pad_b_image_R = myfft2d(blurred_image[:, :, 2])
        pad_k = np.zeros([M, N, 3])
        pad_k[kernel.shape[0], kernel.shape[1], :] = kernel

        fft_kernel_B = myfft2d(pad_k[:, :, 0])
        fft_kernel_G = myfft2d(pad_k[:, :, 1])
        fft_kernel_R = myfft2d(pad_k[:, :, 2])
        p_xy = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        fft_pad_p_xy = np.fft.fft2(p_xy, s=(M, N))

        factor_B = np.divide(np.conjugate(fft_kernel_B), gamma * (abs(fft_pad_p_xy)) ** 2 + (abs(fft_kernel_B)) ** 2)
        factor_G = np.divide(np.conjugate(fft_kernel_G), gamma * (abs(fft_pad_p_xy)) ** 2 + (abs(fft_kernel_G)) ** 2)
        factor_R = np.divide(np.conjugate(fft_kernel_R), gamma * (abs(fft_pad_p_xy)) ** 2 + (abs(fft_kernel_R)) ** 2)

        deblurred_image[:, :, 0] = myifft2d(np.multiply(fft_pad_b_image_B, factor_B))
        deblurred_image[:, :, 1] = myifft2d(np.multiply(fft_pad_b_image_G, factor_G))
        deblurred_image[:, :, 2] = myifft2d(np.multiply(fft_pad_b_image_R, factor_R))
    deblurred_image = abs(deblurred_image)
    deblurred_image = deblurred_image / np.max(np.max(np.max(deblurred_image)))
    return deblurred_image


def interactive_constrained_LS_filter(blurred_image, kernel):
    gamma_init = 100
    gamma_min = 1
    gamma_max = 1000
    fig = plt.figure("Restored Image")
    deblurred_image = constrained_LS_filter(blurred_image, kernel, gamma_init)
    deblurred_image_plot = plt.imshow(cv2.cvtColor(deblurred_image.astype(np.float32), cv2.COLOR_BGR2RGB))
    axis = plt.axes([0.1, 0.05, 0.8, 0.05])
    sliderValue = Slider(axis, 'gamma', gamma_min, gamma_max, valinit=gamma_init)

    def update(gamma):
        updated_image = constrained_LS_filter(blurred_image, kernel, gamma).astype(np.float32)
        updated_image = cv2.cvtColor(updated_image, cv2.COLOR_BGR2RGB)
        deblurred_image_plot.set_data(updated_image)
        fig.canvas.draw_idle()

    sliderValue.on_changed(update)
    plt.show()


def myfft2d(f):
    M = f.shape[0]
    N = f.shape[1]
    f_fft = np.zeros([M, N], np.complex128)
    for x in range(M):
        for y in range(N):
            f_fft[x, y] = sum([sum([f[m, n]*np.exp(-2j*np.pi*(m*x/M+n*y/N)) for n in range(N)]) for m in range(M)])
    return f_fft

def myifft2d(f):
    M = f.shape[0]
    N = f.shape[1]
    f_ifft = np.zeros([M, N], np.complex128)
    for x in range(M):
        for y in range(N):
            f_ifft[x, y] = sum([sum([f[m, n]*np.exp(2j*np.pi*(m*x/M+n*y/N)) for m in range(M)]) for n in range(N)])
    f_ifft = f_ifft/(M*N)
    return f_ifft


curr_dir = 'D:/Sem 7/Image Processing EE 610/IP Assignment 2/dataset/'
ground_truth_image = cv2.imread(curr_dir + 'GroundTruth1.jpg')
g = cv2.imread(curr_dir + 'Blurry1_2.png')
h = cv2.imread(curr_dir + 'K2.png')
g = g/255.0
h = h/255.0
ground_truth_image = ground_truth_image/255.0

# interactive_constrained_LS_filter(g, h)

deblurred_image = constrained_LS_filter(g, h, 90)
# cv2.imwrite('D:/Sem 7/Image Processing EE 610/IP Assignment 2/LS_1_2.png', 255*deblurred_image)
MSE = np.mean((ground_truth_image - deblurred_image)**2)
PSNR = 10*np.log10(255.0*255.0/MSE)
print("PSNR =", PSNR)
print("SSIM =", SSIM(ground_truth_image, deblurred_image, data_range=deblurred_image.max() - deblurred_image.min(), multichannel=True))

cv2.imshow('image', deblurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows