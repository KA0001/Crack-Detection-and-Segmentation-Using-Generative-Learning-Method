import cv2
import os
import glob
import tensorflow
import sys
import numpy as np
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte

def testGenerator(test_path,num_image = 50,target_size = (256,256),flag_multi_class = False,as_gray = False):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        # img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img_as_ubyte(img))
        

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255


input_dir = 'data/test/img/'
output_dir = 'data/test/res/'


Crack = [255,0,0]
Surface = [0,0,0]

COLOR_DICT = np.array([Crack, Surface])

try:
    files = glob.glob(input_dir + '*.jpg')
    i = 0
    for file in files:
        infile = cv2.imread(file)
        ofile_name = output_dir + str(i) + '.png' # ファイル名を0からの連番、拡張子をpngに変更
        infile = cv2.resize(infile, dsize=(256, 256)) # リサイズ
        cv2.imwrite(ofile_name, infile)
        print('ofile_name = ', ofile_name)
        i += 1
finally:
    pass


SAMPLE = 50 # ファイル数

# 学習済みモデルの読み込み
model = tensorflow.keras.models.load_model('model/m1_unet.hdf5', 
                                compile=False)

# testフォルダーのデータを呼び出す
testGene = testGenerator(output_dir,
                         target_size = (256,256))

results = model.predict(testGene,
                        steps = SAMPLE, # サンプルの総数
                        verbose = 1)
# 予測結果を_predict.pngで保存
saveResult(output_dir, results)