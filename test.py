import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

height = 300
width = 300

# 真っ白な300x300の画像
white = np.ones([height, width, 3], dtype = np.uint8) * 255

# 円の中点
p = (150, 150)

# 半径r[px]
r = 4

# 円の描画
# cv2.circle(white, p, 1, (255, 0, 0), -1)
# cv2.circle(white, p, r, (255, 0, 0), 1)

# 角度θ[°]
theta = np.random.randint(0, 359)

# degree → rad に変換
rad = np.deg2rad(theta)

# 移動量を算出
rsinθ = r * np.sin(rad)
rcosθ = r * np.cos(rad)

# 円周上の座標
t = np.array([p[0] + rcosθ, p[1] + rsinθ])

# 円周上の点を描画
cv2.circle(white, (int(t[0]), int(t[1])), 1, (0, 0, 0), -1)


# 画像の可視化
plt.imshow(white)
plt.show()



