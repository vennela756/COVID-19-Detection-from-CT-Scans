import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import shutil


from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
import glob

from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau


import argparse
 
parser = argparse.ArgumentParser()

parser.add_argument("-s", "--seg_data",type = str, help = " segmentation_training directory name")
parser.add_argument("-c", "--covid_data",type = str, help = " covid19_training directory name")
 
args = parser.parse_args()


segmentation_training_data = args.seg_data
covid19_data = args.covid_data

img_list = sorted(glob.glob(segmentation_training_data + '/2d_images/*.tif'))
mask_list = sorted(glob.glob(segmentation_training_data + '/2d_masks/*.tif'))

input_data, output_data = np.empty((2, len(img_list), 256, 256, 1), dtype=np.float32)

for i in range(min(len(img_list),len(mask_list))):
    input_data[i] = resize(imread(img_list[i]), output_shape=(256, 256, 1), preserve_range=True)
    output_data[i] = resize(imread(mask_list[i]), output_shape=(256, 256, 1), preserve_range=True)
output_data /= 255.

#print(input_data,output_data)
x_train, x_val, y_train, y_val = train_test_split(input_data, output_data, test_size=0.1)


datagen = ImageDataGenerator(rescale=255.0,samplewise_center=True,samplewise_std_normalization=True)
datagen.fit(x_train)

train_labels=datagen.flow_from_directory(
    directory=covid19_data+'/train', target_size=(256, 256), color_mode='grayscale', classes=['covid','non-covid'],
    class_mode='categorical', batch_size=1)

test_labels=datagen.flow_from_directory(
    directory=covid19_data+'/test', target_size=(256, 256), color_mode='grayscale', classes=['covid', 'non-covid'],
    class_mode='categorical', batch_size=1)


########### MODEL ###############

inputs = Input(shape=(256, 256, 1))

net = Conv2D(32, kernel_size=3, activation='relu', padding='same')(inputs)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(64, kernel_size=3, activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(128, kernel_size=3, activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Dense(128, activation='relu')(net)

net = UpSampling2D(size=2)(net)
net = Conv2D(128, kernel_size=3, activation='sigmoid', padding='same')(net)

net = UpSampling2D(size=2)(net)
net = Conv2D(64, kernel_size=3, activation='sigmoid', padding='same')(net)

net = UpSampling2D(size=2)(net)
outputs = Conv2D(1, kernel_size=3, activation='sigmoid', padding='same')(net)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse'])

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=1, callbacks=[
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)
])

###########       ###############

my_image=100*train_labels[20][0][0]
my_image.resize((1,256,256,1))
predicted_mask=model.predict(my_image)


try:
  os.mkdir('Processed_Images')
except OSError as error:
  shutil.rmtree('Processed_Images')
  os.mkdir('Processed_Images')
os.mkdir('Processed_Images/Test')
os.mkdir('Processed_Images/Test/covid')
os.mkdir('Processed_Images/Test/non-covid')




c=0
for i in range(24):
  my_image = 100*test_labels[i][0][0]
  my_image.resize((1,256,256,1))
  predicted_mask = model.predict(my_image)  
  if test_labels[i][1][0][0]==1:
    matplotlib.image.imsave('Processed_Images/Test/covid/'+str(c)+'.png', predicted_mask.squeeze().squeeze())
    c=c+1
  elif test_labels[i][1][0][0]==0:
    matplotlib.image.imsave('Processed_Images/Test/non-covid/'+str(c)+'.png', predicted_mask.squeeze().squeeze())
    c=c+1

os.mkdir('Processed_Images/Train')
os.mkdir('Processed_Images/Train/covid')
os.mkdir('Processed_Images/Train/non-covid')

c=0
for i in range(722):
  my_image = train_labels[i][0][0]*100
  my_image.resize((1,256,256,1))
  predicted_mask = model.predict(my_image)  
  if train_labels[i][1][0][0]==1:
    matplotlib.image.imsave('Processed_Images/Train/covid/'+str(c)+'.png', predicted_mask.squeeze().squeeze())
    c=c+1
  elif train_labels[i][1][0][0]==0:
    matplotlib.image.imsave('Processed_Images/Train/non-covid/'+str(c)+'.png', predicted_mask.squeeze().squeeze())
    c=c+1

