# reference:
# https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
# VGG-Style CNN
import os
import sys
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import History
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

img_path = sys.argv[1]

if len(sys.argv) > 2:
    size = int(sys.argv[2])

else:
    size = 32

X_list = []
Y_list = []
training_df_path = img_path + "training_dataframe.csv"
df = pd.read_csv(training_df_path)
for index, data in df.iterrows():
    img = np.load(data["file"])
    X_list.append(img)
    if data["label"] == "merger":
        Y_list.append(1)
    else:
        Y_list.append(0)

X_train = np.array(X_list)
Y_train = np.array(Y_list)
print(X_train.shape)
print(Y_train.shape)

training_len = int(len(X_train) * 0.9)
batch_size = 32
epochs = 30
steps = training_len // batch_size
validation_samples = int(0.1 * len(X_train))
validation_steps = validation_samples // batch_size
print(validation_steps)
input_shape = (size, size, 3)
class_weight = class_weight.compute_class_weight(
    'balanced',
    np.unique(Y_train),
    Y_train
    )

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
              optimizer='adam',
              metrics=['accuracy'])

history = History()
model.fit(
    X_train,
    Y_train,
    steps_per_epoch=steps,
    epochs=epochs,
    verbose=1,
    validation_split=0.1,
    validation_steps=validation_steps,
    shuffle=True,
    class_weight=class_weight,
    callbacks=[history]
    )


exp_number = 0
for i in range(20):
    experiment = 'experiment' + str(i) + '.h5'
    if not os.path.exists(experiment):
        model.save_weights(experiment)
        exp_number = i
        break

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
filename = 'experiment' + str(exp_number) + '_accuracy.png'
plt.savefig(filename, bbox_inches='tight')
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
filename = 'experiment' + str(exp_number) + '_loss.png'
plt.savefig(filename, bbox_inches='tight')
plt.clf()

acc = history.history['acc']
v_acc = history.history['val_acc']
loss = history.history['loss']
v_loss = history.history['val_loss']
history_array = np.array([acc, v_acc, loss, v_loss])
filename = 'experiment' + str(exp_number) + '_history.txt'
np.savetxt(filename, history_array)
