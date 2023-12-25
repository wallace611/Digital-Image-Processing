from skimage import io
import scipy.ndimage as ndi
import numpy as np
from matplotlib import pyplot as plt

img = io.imread("D:\coding\Digital-Image-Processing\Exam\cameraman.tif")
mask = np.ones((9, 9)) / 81
result = ndi.convolve(img, mask, mode='constant')
io.imshow(result)
plt.show()