# 引入相關套件
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

# 校正函式
def unwrap(image, src, dst):
    h, w = image.shape[:2]
    
    # 得到轉移矩陣
    M = cv2.getPerspectiveTransform(src, dst)
    
    # 將圖片進行轉移
    warped = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)
    
    return warped

# 讀取檔案
a = io.imread("Digital-Image-Processing\\Lab6\\ambassadors.jpg")
skull = a
fig = plt.figure()

# 擷取要進行校正的範圍
src = np.float32([
    (303, 987),
    (787, 794),
    (287, 1118),
    (812, 878)
])

# 校正後的大小
dst = np.float32([
    (0, 0),
    (1024, 0),
    (0, 1024),
    (1024, 1024)
])

# 進行校正
skull = unwrap(skull, src, dst)

# 定義輸出格式
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
ax1.plot(x, y, color='red', alpha=1, linewidth=3, solid_capstyle='round', zorder=2)

ax1.imshow(a)
ax2.imshow(skull)

# 輸出
plt.show()