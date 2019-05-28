import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.transform import resize
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize


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
    print(image.shape)
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
        norm = ImageNormalize(stretch=LogStretch())
        np.save(file, img)
        plt.imshow(img, norm=norm)
        plt.savefig("img" + str(index) + ".png")
        if index == 5:
            break


if __name__ == '__main__':
    main()
