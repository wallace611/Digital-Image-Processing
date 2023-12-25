import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure
from numpy.fft import *

def gaussian_filter(size, sigma):
    """2D 高斯濾波器"""
    x = np.arange(-size // 2 + 1, size // 2 + 1)
    y = np.arange(-size // 2 + 1, size // 2 + 1)
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2)) #高斯公式
    return kernel / np.sum(kernel) #卷積核加總求平均

def apply_fft(img, kernel):
    img_fft = fft2(img) #將圖像轉到複平面
    kernel_fft = fft2(kernel, s=img.shape) #將卷積核轉到複平面
    return np.abs(ifft2(img_fft * kernel_fft)) #卷積定理 F(M * S) = F(M) * F(S) 

image = io.imread('cameraman.tif')
filter_size = 21
sigma = 5

# 製作高斯卷積核
gaussian_kernel = gaussian_filter(filter_size, sigma)

# 計算結果
filtered_image = apply_fft(image, gaussian_kernel)


plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(132)
plt.imshow(exposure.rescale_intensity(np.log(1 + abs(fftshift(fft2(image)))), out_range=(0.0, 1.0)), cmap='gray')

plt.title('FFT of Original Image')

plt.subplot(133)
plt.imshow(filtered_image, cmap='gray')
plt.title('Image after Gaussian Filtering')

plt.show()