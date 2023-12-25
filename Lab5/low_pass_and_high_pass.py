from skimage import io,img_as_float
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np

img = io.imread("D:\coding\Digital-Image-Processing\Lab5\cameraman.tif")

mask = np.array(
    [
        [1,4,1],
        [4,-20,4],
        [1,4,1]
    ]
)
img = img_as_float(img)
result = ndi.convolve(img,mask,mode='constant')

maxres = result.max()
minres = result.min()
result = (result-minres) /(maxres-minres)

fig = plt.figure()
plt.gray()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(img)
ax2.imshow(result, vmax=1.0,vmin=0.0)
plt.show()
