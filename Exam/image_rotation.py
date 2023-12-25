from skimage import io
from skimage import transform
import matplotlib.pyplot as plt

c = io.imread('cameraman.tif')
c_ne = transform.rotate(c, 60, order=0)
c_bi = transform.rotate(c, 60, order=1)
c_bc = transform.rotate(c, 60, order=2)

fig = plt.figure()
plt.gray()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.imshow(c)
ax2.imshow(c_ne)
ax3.imshow(c_bi)
ax4.imshow(c_bc)
plt.show()