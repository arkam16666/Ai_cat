import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout


path = 'data/train'

train = ImageDataGenerator(
    rescale = 1./255,
    shear_range =0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)
test = ImageDataGenerator(rescale = 1./255)

train_set = train.flow_from_directory(
    path,
    target_size = (128,128),
    batch_size = 32,
    class_mode = 'binary'
)

X_batch, y_batch = next(train_set) 
#X_train_batch, y_train_batch = next(train_set)

model = Sequential([
    Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'),
    MaxPooling2D(pool_size = (2, 2)),
    Conv2D(32, (3, 3), activation = 'relu'),
    MaxPooling2D(pool_size = (2, 2)),
    Flatten(),
    Dense(units = 128,activation = 'relu'),
    Dropout(0.5),
    Dense(units = 1,activation = 'sigmoid'),

])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

history = model.fit(
    train_set,
    epochs=40
)

model.save('cat.h5')
print(model)
print("-"*50+"\n",history)