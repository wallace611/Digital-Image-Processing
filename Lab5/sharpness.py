from skimage import io, img_as_float
import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt

img = io.imread("D:\coding\Digital-Image-Processing\Lab5\cameraman.tif")
u = np.array([[-2.0,-2.0,-2.0],[-2.0,25.0,-2.0],[-2.0,-2.0,-2.0]]) /9

img = img_as_float(img)
result = ndi.convolve(img, u, mode='constant')

fig = plt.figure()
plt.gray()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.imshow(img)
ax2.imshow(result,vmax=1.0,vmin=0.0)

plt.show()
