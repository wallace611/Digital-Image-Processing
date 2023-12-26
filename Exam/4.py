import numpy as np
from skimage import io, img_as_float
from matplotlib import pyplot as plt
import scipy.ndimage as ndi 

img = io.imread('cameraman.tif')

arr = np.asarray(img)
r, c = arr.shape
m0 = np.zeros((2*r, 2*c))
m0[::2, ::2] = arr

mask_ne = np.array(
    [
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 0]
    ]
)

mask_bi = np.array(
    [
        [0.25, 0.5, 0.25],
        [0.5, 1, 0.5],
        [0.25, 0.5, 0.25]
    ]
)

mask_bc = np.array(
    [
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ]
)

m_ne = ndi.convolve(m0, mask_ne, mode='reflect')
m_bi = ndi.convolve(m0, mask_bi, mode='reflect')
m_bc = ndi.convolve(m0, mask_bc, mode='reflect')


image = img_as_float(m_bc)
plt.imshow(image)
plt.show()
