import os
import sys
from datetime import datetime
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report

img_path = sys.argv[1]
img_path_clean = img_path.replace("/", " ").rstrip().split(" ")[-1]

experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

X_test = np.load(os.path.join(img_path, 'X_test.npy'))
y_test = np.load(os.path.join(img_path, 'y_test.npy'))

trained_model = sys.argv[2]
model = load_model(trained_model)

score = model.evaluate(X_test, y_test)

print("accuracy = %.2f%%" % (score[1]*100))

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred, labels=np.unique(y_pred)))
