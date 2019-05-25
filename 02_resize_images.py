import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.transform import resize


def resize_img(tensor, cfg):
    bands_id = {"g": 0, "i": 1, "r": 2, "u": 3, "z": 4}

    tensors_list = []
    for band in cfg:
        id = bands_id[band]
        tensors_list.append(tensor[id, :, :])

    img = np.array(tensors_list)
    image = np.transpose(img, (1, 2, 0))
    image = image / 65536
    resize(image, (32, 32), preserve_range=True)
    print(image.shape)
    return image


def main():
    img_path = sys.argv[1]
    if len(sys.argv) == 3:
        cfg = sys.argv[2]

    else:
        cfg = "rgu"

    dataframe = pd.read_csv("dataframe.csv")
    for index, data in dataframe.iterrows():
        tensor = np.load(data["file"])
        img = resize_img(tensor, cfg)
        file = img_path + str(data["objid"])
        np.save(file, img)
        plt.imshow(img)
        plt.show()
        break


if __name__ == '__main__':
    main()
