import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.measure import compare_ssim as SSIM


# ---------------------ALL LINES ARE WRITTEN BY ME-----------------------------------
def trunc_inv_filter(blurred_image,kernel, radius_):
    # fftType = 0 for faster fft; = 1 for implemented (slower) fft
    fftType = 0
    M, N, _ = blurred_image.shape
    deblurred_image = np.zeros([M, N, 3], np.complex128)
    if fftType == 0:
        fft_pad_b_image_B = np.fft.fft2(blurred_image[:, :, 0])
        fft_pad_b_image_G = np.fft.fft2(blurred_image[:, :, 1])
        fft_pad_b_image_R = np.fft.fft2(blurred_image[:, :, 2])
        fft_kernel_B = np.fft.fft2(kernel[:, :, 0], s = (M,N))
        fft_kernel_G = np.fft.fft2(kernel[:, :, 1], s = (M,N))
        fft_kernel_R = np.fft.fft2(kernel[:, :, 2], s = (M,N))

        trunc_matrix = np.zeros([M, N], np.complex)
        radius = int(np.floor(radius_))
        x, y = np.ogrid[0:M, 0:N]
        mask = x*x + y*y >= (M-radius)**2
        trunc_matrix[mask] = 1.0


        deblurred_image[:, :, 0] = np.fft.ifft2(np.multiply(fft_pad_b_image_B, np.divide(trunc_matrix, fft_kernel_B)))
        deblurred_image[:, :, 1] = np.fft.ifft2(np.multiply(fft_pad_b_image_G, np.divide(trunc_matrix, fft_kernel_G)))
        deblurred_image[:, :, 2] = np.fft.ifft2(np.multiply(fft_pad_b_image_R, np.divide(trunc_matrix, fft_kernel_R)))

    if fftType == 1:
        fft_pad_b_image_B = myfft2d(blurred_image[:, :, 0])
        fft_pad_b_image_G = myfft2d(blurred_image[:, :, 1])
        fft_pad_b_image_R = myfft2d(blurred_image[:, :, 2])
        pad_k = np.zeros([M, N, 3])
        pad_k[0:kernel.shape[0], 0:kernel.shape[1], :] = kernel
        fft_kernel_B = myfft2d(pad_k[:, :, 0])
        fft_kernel_G = myfft2d(pad_k[:, :, 1])
        fft_kernel_R = myfft2d(pad_k[:, :, 2])

        trunc_matrix = np.zeros([M, N], np.complex)
        radius = int(np.floor(radius_))
        m = int(np.floor(M / 2))
        n = int(np.floor(N / 2))
        x1 = int(np.floor((radius - 1) / 2))
        x2 = int(np.floor(radius / 2))
        trunc_matrix[m-x1-1:m+x2, n-x1-1:n+x2] = np.ones([radius, radius], np.complex)

        deblurred_image[:, :, 0] = myifft2d(np.divide(np.multiply(fft_pad_b_image_B, trunc_matrix), fft_kernel_B))
        deblurred_image[:, :, 1] = myifft2d(np.divide(np.multiply(fft_pad_b_image_G, trunc_matrix), fft_kernel_G))
        deblurred_image[:, :, 2] = myifft2d(np.divide(np.multiply(fft_pad_b_image_R, trunc_matrix), fft_kernel_R))
    deblurred_image = abs(deblurred_image)
    deblurred_image = deblurred_image / np.max(np.max(np.max(deblurred_image)))
    return deblurred_image


def interactive_trunc_inv_filter(blurred_image, kernel):
    radius_init = min(blurred_image.shape[0], blurred_image.shape[1])
    radius_min = 2.0
    radius_max = min(blurred_image.shape[0], blurred_image.shape[1])
    fig = plt.figure("Restored Image")
    deblurred_image = trunc_inv_filter(blurred_image, kernel, radius_init)
    deblurred_image_plot = plt.imshow(cv2.cvtColor(deblurred_image.astype(np.float32), cv2.COLOR_BGR2RGB))
    axis = plt.axes([0.1, 0.05, 0.8, 0.05])
    sliderValue = Slider(axis, 'radius', radius_min, radius_max, valinit=radius_init)

    def update(radius):
        updated_image = trunc_inv_filter(blurred_image, kernel, radius).astype(np.float32)
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
ground_truth_image = cv2.imread(curr_dir + 'GroundTruth3.jpg')
g = cv2.imread(curr_dir + 'Blurry1_3.png')
h = cv2.imread(curr_dir + 'K3.png')
g = g/255.0
h = h/255.0
ground_truth_image = ground_truth_image/255.0

# interactive_trunc_inv_filter(g, h)

deblurred_image = trunc_inv_filter(g, h, 60)
# cv2.imwrite('D:/Sem 7/Image Processing EE 610/IP Assignment 2/trunc_1_3.png', 255*deblurred_image)
MSE = np.mean((ground_truth_image - deblurred_image)**2)
PSNR = 10*np.log10(255.0*255.0/MSE)
print("PSNR =", PSNR)
print("SSIM =", SSIM(ground_truth_image, deblurred_image, data_range=deblurred_image.max() - deblurred_image.min(), multichannel=True))

cv2.imshow('image', deblurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows