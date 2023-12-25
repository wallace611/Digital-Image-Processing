from skimage import io
import scipy.ndimage as ndi
import numpy as np
from matplotlib import pyplot as plt
import cv2

img = io.imread("D:\coding\Digital-Image-Processing\Lab5\cameraman.tif")
mask = np.array(
    [[1,4,1],
    [4,-20,4],
    [1,4,1]]
)

result1 = ndi.convolve(img, mask, mode='constant')
result2 = ndi.laplace(img)

fig = plt.figure()
plt.gray()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(result1)
ax2.imshow(result2)

plt.show()