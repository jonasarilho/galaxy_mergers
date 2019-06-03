# reference:
# https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
# VGG-Style CNN
import sys
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

img_path = sys.argv[1]

if len(sys.argv) > 2:
    size = int(sys.argv[1])

else:
    size = 32

X_train = []
Y_train = []
training_df_path = img_path + "training_dataframe.csv"
df = pd.read_csv(training_df_path)
for index, data in df.iterrows():
    img = np.load(data["file"])
    X_train.append(img)
    Y_train.append(data["label"])

input_shape = (size, size, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, Y_train, validation_split=0.1)

model.save_weights('experiment1.h5')
