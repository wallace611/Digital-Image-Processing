from skimage import io, img_as_float
import scipy.ndimage as ndi
import numpy as np
from matplotlib import pyplot as plt

img = io.imread("cameraman.tif")

mask = np.array([[1,4,1],[4,-20,4],[1,4,1]])
img = img_as_float(img)
result = ndi.convolve(img, mask, mode='constant')

plt.gray()
plt.imshow(result, vmax=1.0, vmin=0.0)
plt.show()