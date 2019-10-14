import os
import sys
from datetime import datetime
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.initializers import Constant
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, LeakyReLU
from keras.constraints import max_norm
from keras.optimizers import SGD
from keras.regularizers import l2
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
    "weights_imagenet.{epoch:02d}-{val_loss:.2f}.hdf5",
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
    weights="imagenet",
    include_top=False,
    input_shape=input_shape,
    pooling=max
    )

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    2048,
    kernel_initializer='glorot_uniform',
    bias_initializer=Constant(value=0.01),
    kernel_constraint=max_norm(3),
    bias_constraint=max_norm(3),
    activity_regularizer=l2(0.1)
    )(x)
x = LeakyReLU()(x)
x = Dropout(0.5)(x)


predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# for i, layer in enumerate(base_model.layers):
#     print(i, layer.name)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
    )

raw_history = model.fit(
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
    '%s_%s_inceptionv3_imagenet.h5' % (experiment_timestamp, img_path_clean)
    )
plt.plot(raw_history.history['acc'])
plt.plot(raw_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig(
    '%s_%s_raw_learning_curve_acc.png' % (
        experiment_timestamp,
        img_path_clean
        ),
    bbox_inches='tight'
    )
plt.clf()

for layer in model.layers[:165]:
    layer.trainable = False
for layer in model.layers[165:]:
    layer.trainable = True

fine_opt = SGD(lr=0.0001, momentum=0.9)

model.compile(
    optimizer=fine_opt,
    loss='binary_crossentropy',
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
    '%s_%s_inceptionv3_imagenet.h5' % (experiment_timestamp, img_path_clean)
    )
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig(
    '%s_%s_ft_learning_curve_acc.png' % (
        experiment_timestamp,
        img_path_clean
        ),
    bbox_inches='tight'
    )
