import cv2
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
import numpy as np

def unwrap(image, src, dst):
    h, w = image.shape[:2]
    
    M = cv2.getPerspectiveTransform(src, dst)
    
    warped = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)
    
    return warped

a = io.imread("Digital-Image-Processing\\Lab6\\ambassadors.jpg")
skull = a
fig = plt.figure()

src = np.float32([
    (303, 987),
    (787, 794),
    (287, 1118),
    (812, 878)
])

dst = np.float32([
    (0, 0),
    (1024, 0),
    (0, 1024),
    (1024, 1024)
])

skull = unwrap(skull, src, dst)

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
ax1.plot(x, y, color='red', alpha=1, linewidth=3, solid_capstyle='round', zorder=2)

ax1.imshow(a)
ax2.imshow(skull)

plt.show()