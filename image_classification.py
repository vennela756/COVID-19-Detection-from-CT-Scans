import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator


from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau



my_classes = os.listdir("Processed_Images/Train/")

base_dir = "Processed_Images"
train_dir = os.path.join(base_dir, 'Train')
test_dir = os.path.join(base_dir, 'Test')



######### Pre-processing  #####################

datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

train_generator = datagen.flow_from_directory(train_dir, target_size=(256, 256), color_mode='rgb', 
                                                    classes=my_classes,
                                                    batch_size=16, shuffle=True)
test_generator = datagen.flow_from_directory(test_dir, target_size=(256, 256), color_mode='rgb', 
                                                    classes=my_classes, batch_size=16, shuffle=True)

############ CovidNET  #################

base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(256, 256, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(2, activation='softmax')(x)


model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['acc', 'mse'])

history = model.fit(train_generator, validation_data=test_generator, epochs=100, callbacks=[
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_lr=1e-05)
])