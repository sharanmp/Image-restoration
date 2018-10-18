import cv2
import numpy as np
from skimage.measure import compare_ssim as SSIM


# ---------------------ALL LINES ARE WRITTEN BY ME-----------------------------------
def inv_filter(blurred_image,kernel):
    # fftType = 0 for faster fft; = 1 for implemented (slower) fft
    fftType = 0
    M, N, _ = blurred_image.shape
    deblurred_image = np.zeros([M, N, 3], np.complex128)
    if fftType == 0:
        fft_pad_b_image_B = np.fft.fft2(blurred_image[:, :, 0])
        fft_pad_b_image_G = np.fft.fft2(blurred_image[:, :, 1])
        fft_pad_b_image_R = np.fft.fft2(blurred_image[:, :, 2])
        fft_kernel_B = np.fft.fft2(kernel[:, :, 0], s=(M, N))
        fft_kernel_G = np.fft.fft2(kernel[:, :, 1], s=(M, N))
        fft_kernel_R = np.fft.fft2(kernel[:, :, 2], s=(M, N))
        deblurred_image[:, :, 0] = np.fft.ifft2(np.divide(fft_pad_b_image_B, fft_kernel_B))
        deblurred_image[:, :, 1] = np.fft.ifft2(np.divide(fft_pad_b_image_G, fft_kernel_G))
        deblurred_image[:, :, 2] = np.fft.ifft2(np.divide(fft_pad_b_image_R, fft_kernel_R))

    if fftType == 1:
        fft_pad_b_image_B = myfft2d(pad_b_image[:, :, 0])
        fft_pad_b_image_G = myfft2d(pad_b_image[:, :, 1])
        fft_pad_b_image_R = myfft2d(pad_b_image[:, :, 2])
        pad_k = np.zeros([M, N, 3])
        pad_k[kernel.shape[0], kernel.shape[1], :] = kernel
        fft_kernel_B = myfft2d(pad_k[:, :, 0])
        fft_kernel_G = myfft2d(pad_k[:, :, 1])
        fft_kernel_R = myfft2d(pad_k[:, :, 2])
        deblurred_image[:, :, 0] = myifft2d(np.divide(fft_pad_b_image_B, fft_kernel_B))
        deblurred_image[:, :, 1] = myifft2d(np.divide(fft_pad_b_image_G, fft_kernel_G))
        deblurred_image[:, :, 2] = myifft2d(np.divide(fft_pad_b_image_R, fft_kernel_R))
    deblurred_image = abs(deblurred_image)
    deblurred_image = deblurred_image / np.max(np.max(np.max(deblurred_image)))
    return deblurred_image


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
ground_truth_image = cv2.imread(curr_dir + 'GroundTruth4.jpg')
g = cv2.imread(curr_dir + 'Blurry1_4.png')
h = cv2.imread(curr_dir + 'K4.png')
g = g/255.0
h = h/255.0
ground_truth_image = ground_truth_image/255.

deblurred_image = inv_filter(g, h)
# cv2.imwrite('D:/Sem 7/Image Processing EE 610/IP Assignment 2/inv_1_4.png', 255*deblurred_image)
deblurred_image = deblurred_image*np.max(np.max(ground_truth_image))/np.max(np.max(deblurred_image))
MSE = np.mean((ground_truth_image - deblurred_image)**2)
PSNR = 10*np.log10(255.0**2/MSE)
print("PSNR =", PSNR)
print("SSIM =", SSIM(ground_truth_image, deblurred_image, data_range=deblurred_image.max() - deblurred_image.min(), multichannel=True))
# print(np.max(np.max(ground_truth_image)))
# print(np.max(np.max(deblurred_image)))
cv2.imshow('image', deblurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows