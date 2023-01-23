import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import random
import glob
import os

bgs = glob.glob('generated_data/bg/*.jpg')
n = 0

wmin = 1
wmax = 3
for i in bgs:
    height = 256
    width = 256
    points = []
    points2 = []
    
    # ファイル名取得
    
    if n>=200:
        wmin = 3
        wmax = 5
    else:
        pass
    
    basename = os.path.splitext(os.path.basename(i))[0]
    


    dst = cv2.imread(i)
    dst = cv2.resize(dst, dsize=(256, 256))
    
    k = random.randint(50, 205)
    p = (k, 0)
    points.append(p)

    # 太さの追加
    points2.append((p[0]+random.randint(wmin,wmax), p[1]))

    # 半径の指定
    r = 3

    while True:
      # 円の中心pip 
      sp = (p[0], p[1]+r)

      # 円の描画
      # cv2.circle(dst, (int(sp[0]),int(sp[1])), r, (255, 0, 0), 1)

      # 角度θ[°]
      theta = np.random.randint(0, 180)

      # degree → rad に変換
      rad = np.deg2rad(theta)

      # 移動量を算出
      rsinθ = r * np.sin(rad)
      rcosθ = r * np.cos(rad)

      # 円周上の座標
      p = np.array([sp[0] + rcosθ, sp[1] + rsinθ])
      points.append((int(p[0]), int(p[1])))

      # 太さの追加
      points2.append((int(p[0])+random.randint(wmin,wmax),int(p[1])))

      if p[1] >= 256:
        break
    

    
    points2.reverse()
    points.extend(points2)
    points = np.array(points)
     
    # ひび割れ部分描画
    crack = np.zeros((width, height, 3), np.uint8)
    crack.fill(255)
    cv2.fillConvexPoly(crack, np.int32([points]), color=(0,0,0))
    
    #回転
    center = (int(width/2), int(height/2))
    angle = np.random.randint(0, 180)
    scale = 1.0
    trans = cv2.getRotationMatrix2D(center, angle, scale)
    crack = cv2.warpAffine(crack, trans, (width,height),borderValue=(255, 255, 255))
    
    # 余白部分の透過
    mask = np.all(crack[:,:,:] == [255,255,255], axis=-1)
    # Affineではみ出した部分を黒で補完
    # crack[np.where((crack != (255,255,255)).all(axis=2))] = (0,0,0)
    crack = cv2.cvtColor(crack, cv2.COLOR_BGR2BGRA)
    crack[mask, 3] = 0

    
    # 背景重ねる
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2BGRA)
    x1, y1, x2, y2 = 0, 0, crack.shape[1], crack.shape[0]
    dst[y1:y2, x1:x2] = dst[y1:y2, x1:x2] * (1-crack[:, :, 3:]/255) + crack[:, :, :4] * (crack[:, :, 3:]/255)
    crack = 255-crack
    crack = cv2.cvtColor(crack, cv2.COLOR_BGR2GRAY)
    dst = cv2.blur(dst, (3, 3)) #ブラーをかける
    
    cv2.imwrite('generated_data/res3/img/' + basename, dst)
    cv2.imwrite('generated_data/res3/msk/' + basename, crack)
    n+=1
    





