import os
import sys
from datetime import datetime
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

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
print(input_shape)

class_weight = class_weight.compute_class_weight(
    'balanced',
    np.unique(y_train),
    y_train
    )

checkpoint = ModelCheckpoint(
    "weights_fromscratch.{epoch:02d}-{val_loss:.2f}.hdf5",
    monitor='val_loss',
    verbose=1,
    period=1,
    mode='auto',
    save_best_only=True,
    save_weights_only=False
    )

earlystopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=3,
    verbose=1,
    mode='auto',
    restore_best_weights=True
    )

base_model = InceptionV3(
    weights=None,
    include_top=False,
    input_shape=input_shape,
    pooling=max
    )

x = base_model.output

x = Flatten()(x)

x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)

predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print(base_model.summary())

opt = SGD(lr=0.01)

model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
    )


history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=200,
    verbose=1,
    validation_data=(X_valid, y_valid),
    shuffle=True,
    class_weight=class_weight,
    callbacks=[earlystopping, checkpoint]
    )

model.save(
    '%s_%s_vgg16_fromscratch.h5' %
    (experiment_timestamp, img_path_clean)
    )

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig(
    '%s_%s_learning_curve_acc.png' %
    (experiment_timestamp, img_path_clean),
    bbox_inches='tight'
    )
