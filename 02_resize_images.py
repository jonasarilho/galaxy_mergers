import sys
import numpy as np
import pandas as pd
from skimage.transform import resize
from astropy.visualization import LogStretch


def resize_img(tensor, cfg, size):
    bands_id = {"g": 0, "i": 1, "r": 2, "u": 3, "z": 4}

    tensors_list = []
    for band in cfg:
        id = bands_id[band]
        tensors_list.append(tensor[id, :, :])

    img = np.array(tensors_list)
    image = np.transpose(img, (1, 2, 0))
    image = (image / 65536).astype('float32')
    image = resize(image, (size, size), preserve_range=True)
    stretch = LogStretch()
    image = stretch(image)
    return image


def main():
    img_path = sys.argv[1]
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
        img = resize_img(tensor, cfg, size)
        file = img_path + str(data["objid"])
        np.save(file, img)


if __name__ == '__main__':
    main()
