import os
import sys
import numpy as np
import pandas as pd
from skimage.transform import resize
from astropy.visualization import LogStretch


def bands_list(cfg):
    bands_id = {"g": 0, "i": 1, "r": 2, "u": 3, "z": 4}
    ids_list = []
    for band in cfg:
        ids_list.append(bands_id[band])

    return ids_list


def resize_img(tensor, bands, size):
    img = tensor[bands, :, :]
    image = np.transpose(img, (1, 2, 0))
    image = (image / 65536).astype('float32')
    image = resize(image, (size, size), preserve_range=True)
    stretch = LogStretch()
    image = stretch(image)
    return image


def main():
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    if len(sys.argv) > 2:
        cfg = sys.argv[2]

    else:
        cfg = "rgu"

    if len(sys.argv) > 3:
        size = int(sys.argv[3])

    else:
        size = 32

    dataframe = pd.read_csv("dataframe.csv")
    for index, data in dataframe.iterrows():
        tensor = np.load(data["file"])
        bands = bands_list(cfg)
        img = resize_img(tensor, bands, size)
        file = img_path + str(data["objid"])
        np.save(file, img)
        if index % 500 == 0:
            msg = "Already saved {} files with size {}x{}"
            print(msg.format(index, size, size))


if __name__ == '__main__':
    main()
