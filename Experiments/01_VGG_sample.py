# reference:
# https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
# VGG-Style CNN
import os
import sys
from datetime import datetime
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
from sklearn.utils import class_weight

img_path = sys.argv[1]
img_path_clean = img_path.replace("/", " ").rstrip().split(" ")[-1]

experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if os.path.getsize(os.path.join(img_path, 'X_train.npy')) > 10*1024*1024*1024:
    mmap_mode = "r+"
else:
    mmap_mode = None

X_train = np.load(os.path.join(img_path, 'X_train.npy'), mmap_mode=mmap_mode)
X_valid = np.load(os.path.join(img_path, 'X_valid.npy'), mmap_mode=mmap_mode)

y_train = np.load(os.path.join(img_path, 'y_train.npy'))
y_valid = np.load(os.path.join(img_path, 'y_valid.npy'))

input_shape = X_train.shape[1:]
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1, mode='auto', restore_best_weights=True)

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=200,
                    verbose=1,
                    validation_data=(X_valid, y_valid),
                    shuffle=True,
                    callbacks=[earlystopping])

model.save('%s_%s_vgg_style.h5' % (experiment_timestamp, img_path_clean))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('%s_%s_learning_curve_acc.png' % (experiment_timestamp, img_path_clean), bbox_inches='tight')
