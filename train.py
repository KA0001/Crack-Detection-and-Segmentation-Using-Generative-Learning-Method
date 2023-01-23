from model import *
import skimage.io as io
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

model = unet(input_size = (256, 256, 3))

name = 'm1_unet'

def dice_loss(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + 1) / (union + 1), axis=0)
    return 1. - dice

def iou(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    iou = K.mean((intersection + 1) / (union + 1), axis=0)
    return iou

def mean_iou(y_true, y_pred):
    results = []   
    for t in np.arange(0.5, 1, 0.05):
        t_y_pred = tf.cast((y_pred > t), tf.float32)
        pred = iou(y_true, t_y_pred)
        results.append(pred)
        
    return K.mean(K.stack(results), axis=0)

K.clear_session()

model.compile(optimizer = optimizers.Adam(learning_rate=1e-4), loss = dice_loss, metrics = ['accuracy', iou, dice_loss])



def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = False,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr

        
        
"""
trainフォルダーのimage、labelフォルダーからデータを読み込んでデータを拡張してaugフォルダーに保存する。
"""

# データ拡張 trainから読み込んでデータ拡張してaugに保存する
data_gen_args = dict(rotation_range = 0.5, #回転
                    # width_shift_range = 0.1, #水平移動
                    # height_shift_range = 0.1, #垂直移動
                    shear_range = 0.01, #シアー変換
                    zoom_range = 0.1, #ズーム
                    horizontal_flip = True, #左右反転
                    fill_mode='nearest')
myGenerator = trainGenerator(400,
                             'data/train','img','msk',
                             data_gen_args,
                             save_to_dir = 'data/train/aug',
                             target_size = (256, 256))
num_batch = 4 # 回繰り返す
for i,batch in enumerate(myGenerator):
    if(i >= num_batch):
        break
    
    
"""
augフォルダーの画像を学習して最適なモデルを保存する
"""
# u-netで学習

# チェックポイント
cp = ModelCheckpoint('model/'+ name +'.hdf5', # チェックポイント
                     monitor = 'val_loss', # 監視する値
                     verbose = 1, # 結果表示
                     save_best_only = True)

#過学習を防ぐための早期終了
es = EarlyStopping(monitor = 'val_loss',
                   patience = 500, # このepoch数、性能が向上しなければストップ
                   restore_best_weights = True)

imgs_train,imgs_mask_train = geneTrainNpy('data/train/aug/', 'data/train/aug/')
history = model.fit(imgs_train,
                    imgs_mask_train,
                    batch_size = 10,
                    epochs = 100,
                    verbose = 1,
                    validation_split = 0.2,
                    shuffle = True,
                    callbacks = # チェックポイント、早期終了
                    [cp,
                     es])

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



train_dice = history.history['dice_loss']
valid_dice = history.history['val_dice_loss']

train_IOU = history.history['iou']
valid_IOU = history.history['val_iou']


fig, axes = plt.subplots(1, 2, figsize=(20,7))
axes = axes.flatten()

axes[0].plot(train_IOU, label='training')
axes[0].plot(valid_IOU, label='validation')
axes[0].set_title('IOU Curve [Adam lr : 0.0001]')
axes[0].set_xlabel('epochs')
axes[0].set_ylabel('IOU')
axes[0].legend()


axes[1].plot(train_dice, label='training')
axes[1].plot(valid_dice, label='validation')
axes[1].set_title('Dice coefficient Curve')
axes[1].set_xlabel('epochs')
axes[1].set_ylabel('dice_coef')
axes[1].legend()

plt.show()




