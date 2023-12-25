from skimage import io,util
from scipy.signal import wiener
import numpy as np
import matplotlib.pyplot as plt

b = io.imread('cameraman.tif')
bn = util.noise.random_noise(b,mode ='gaussian')
brn = wiener(bn,[3,3])
fig = plt.figure()
ax1 = fig.add_subplot(131)
ax1.imshow(b,cmap='gray')
ax2 = fig.add_subplot(132)
ax2.imshow(bn,cmap='gray')
ax3 = fig.add_subplot(133)
ax3.imshow(brn,cmap='gray')
plt.show()
