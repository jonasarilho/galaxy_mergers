import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import ndimage


def main():
    if len(sys.argv) != 4:
        print("Usage: python 03_split_datasets.py <path> <n_bands> <side>")
        exit(1)

    img_path = sys.argv[1]
    n_bands = int(sys.argv[2])
    side = int(sys.argv[3])

    if not os.path.exists(img_path):
        print("%s does not exist" % img_path)
        exit(1)

    lenc = LabelEncoder()

    data = pd.read_csv("dataframe_jpeg.csv")

    X_objid = data["objid"]
    y = np.array(lenc.fit_transform(data["label"])).astype('uint8')

    print(lenc.classes_)

    X_rest_objid, X_test_objid, y_rest, y_test = train_test_split(
        X_objid,
        y,
        test_size=1600,
        stratify=y,
        random_state=420)

    X_train_objid, X_valid_objid, y_train, y_valid = train_test_split(
        X_rest_objid,
        y_rest,
        test_size=1600,
        stratify=y_rest,
        random_state=420)

    splitted_datasets = zip(
        [X_train_objid, X_valid_objid, X_test_objid],
        ['train', 'valid', 'test'],
        [y_train, y_valid, y_test])

    angles = np.linspace(20, 340, 17)
    size = (len(angles) + 1)

    for split_df, split_name, y_img in splitted_datasets:
        x_angle = np.zeros(
            (len(split_df), side, side, n_bands)
            ).astype('float32')
        X_img = x_angle
        for _ in angles:
            X_img = np.concatenate((X_img, x_angle), axis=0)

        Y_img = np.zeros((len(angles) + 1) * len(y_img))
        print(split_name, X_img.shape, y_img.shape)

        for i, objid in enumerate(split_df):
            print(i, objid, y_img[i])
            x = np.load(os.path.join(img_path, str(objid) + ".npy"))
            x = x.astype('float32')
            X_img[i * size, :, :, :] = x
            Y_img[i * size] = y_img[i]
            for j, angle in enumerate(angles):
                X_rotated = ndimage.rotate(x, angle, reshape=False)
                X_img[i * size + j + 1, :, :, :] = X_rotated.astype('float32')
                Y_img[i * size + j + 1] = y_img[i]

        np.save(os.path.join(img_path, 'X_%s.npy' % split_name), X_img)
        np.save(os.path.join(img_path, 'y_%s.npy' % split_name), Y_img)


if __name__ == '__main__':
    main()
