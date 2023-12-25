from skimage import io, img_as_float
import numpy as np
import scipy.ndimage as ndi
from matplotlib import pyplot as plt

img = io.imread('cameraman.tif')
x = -2
mask = np.array(
    [
        [x, x, x],
        [x, 25, x],
        [x, x, x]
    ]
) / 9

img = img_as_float(img)
result = ndi.convolve(img, mask)

plt.imshow(result, vmax=1.0, vmin=0.0)
plt.gray()
plt.show()