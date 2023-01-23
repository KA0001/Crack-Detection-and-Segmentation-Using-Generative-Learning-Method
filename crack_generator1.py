import numpy as np
import cv2
from PIL import Image 
import os


def plot(min, max, lim, s_x1, s_y, w_min, w_max):

  # 初期座標
  x1 = s_x1
  x2 = x1 + np.random.randint(10, 20) #先の太さ
  y = s_y

  # 2回以降
  for i in range(4000):
    p = []
    p.append((x1, y))
    p.append((x2, y))

    x2 += np.random.randint(min, max) # 振れ幅
    x1 = x2 - np.random.randint(w_min, w_max)
    y += np.random.randint(3, 4)

    p.append((x2, y))
    p.append((x1, y))

    points = np.array(p)

    cv2.fillConvexPoly(dst, points, (0,0,0))

    if y>= lim:
      break
  
  return x1, y
  

bg_path = 'generated_data/bg'
ff = os.listdir(bg_path)

height = 256
width = 256

# main
min = -3
max = 3
lim = 500
s_x1 = np.random.randint(50,200)
s_y = 0
w_min = 2
w_max = 5


n=0


for i in ff:
    
  #  if n>=200:
  #      w_min = 1
  #      w_max = 5

    dst = np.zeros((width, height, 3), np.uint8)
    dst.fill(255)
    p = []

    x1r, y = plot(min, max, lim, s_x1, s_y, w_min, w_max)
    img_path = bg_path + "/" + i
    bg = cv2.imread(img_path)

    # リサイズ
    bg = cv2.resize(bg , (256, 256))
    
    
    #回転
    center = (int(width/2), int(height/2))
    angle = np.random.randint(0, 180)
    scale = 1.0
    trans = cv2.getRotationMatrix2D(center, angle, scale)
    dst = cv2.warpAffine(dst, trans, (width,height),borderValue=(255, 255, 255))

    
    # 余白部分の透過
    mask = np.all(dst[:,:,:] == [255,255,255], axis=-1)
    # Affineではみ出した部分を黒で補完
    # dst[np.where((dst != (255,255,255)).all(axis=2))] = (0,0,0)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2BGRA)
    dst[mask, 3] = 0

    # 背景重ねる
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA)
    x1, y1, x2, y2 = 0, 0, dst.shape[1], dst.shape[0]
    bg[y1:y2, x1:x2] = bg[y1:y2, x1:x2] * (1-dst[:, :, 3:]/255) + dst[:, :, :4] * (dst[:, :, 3:]/255)
    dst = 255-dst
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    bg = cv2.blur(bg, (3, 3)) #ブラーをかける


    cv2.imwrite("generated_data/res1/img/"+str(n)+".jpg", bg)
    cv2.imwrite("generated_data/res1/msk/"+str(n)+".jpg", dst)
    
    n+=1