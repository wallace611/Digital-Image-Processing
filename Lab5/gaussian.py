from skimage import io
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

img = io.imread("D:\coding\Digital-Image-Processing\Lab5\cameraman.tif")

result1 = ndi.gaussian_filter(img,0.5,truncate = 4.5)
result2 = ndi.gaussian_filter(img,2,truncate = 1)
result3 = ndi.gaussian_filter(img,1,truncate = 5)
result4 = ndi.gaussian_filter(img,5,truncate = 1)

fig = plt.figure()
plt.gray()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.imshow(result1)
ax2.imshow(result2)
ax3.imshow(result3)
ax4.imshow(result4)
plt.show()
