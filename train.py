from keras.applications.vgg16 import VGG16
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from tqdm import tqdm

path = os.getcwd()
dir_list = os.listdir(path + '/SegmentationObject/')
labels = []
images = []
width = 224
height = 224
for i in tqdm(range(len(dir_list))):
  im = Image.open(path + '/SegmentationObject/'+dir_list[i]).resize((width,height)).convert('L')
  image_mask = tf.keras.preprocessing.image.img_to_array(im).reshape(width,height,1)
  for k in range(width):
    for j in range(height):
      if(image_mask[k][j][0]>0):
        image_mask[k][j][0] = 1
  labels.append(image_mask)
  im = Image.open(path + '/JPEGImages/'+dir_list[i][:len(dir_list[i])-3]+'PNG').resize((width,height))
  image = tf.keras.preprocessing.image.img_to_array(im).reshape(width,height,3)
  images.append(image)

images = np.asarray(images)
labels = np.asarray(labels)

fcn = keras.Sequential()
fcn.add(keras.Input(shape=(224,224,3)))
fcn.add(layers.Conv2D(4, kernel_size=3, strides=1, padding='same',activation='relu'))
fcn.add(layers.MaxPool2D(4))
fcn.add(layers.Conv2D(8, kernel_size=3, strides=1, padding='same',activation='relu'))
fcn.add(layers.MaxPool2D(8))
fcn.add(layers.Conv2D(12, kernel_size=3, strides=1, padding='same',activation='relu'))
fcn.add(layers.Conv2DTranspose(12,kernel_size=4,strides=2,padding='same'))
fcn.add(layers.Conv2DTranspose(8,kernel_size=8,strides=4,padding='same'))
fcn.add(layers.Conv2DTranspose(4,kernel_size=8,strides=4,padding='same'))
fcn.add(layers.Conv2D(1,1,activation='sigmoid'))

fcn.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[0], threshold=0.5)],
)

epochs = 40
fcn.fit(images[:400], labels[:400], validation_data=(images[400:],labels[400:]), epochs=epochs, batch_size = 20, shuffle = True)