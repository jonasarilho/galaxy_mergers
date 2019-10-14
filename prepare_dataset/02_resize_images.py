import os
import sys
import numpy as np
import pandas as pd
from skimage.transform import resize


def resize_img(tensor, size):
    image = resize(tensor, (size, size), preserve_range=True)
    return image.astype('float32')


def main():
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    if len(sys.argv) > 2:
        size = int(sys.argv[2])
    else:
        size = 32

    dataframe = pd.read_csv("dataframe_jpeg.csv")
    for index, data in dataframe.iterrows():
        tensor = np.load(data["file"])
        img = resize_img(tensor, size)
        file = os.path.join(img_path, str(data["objid"]))
        np.save(file, img)

        if index % 500 == 0:
            msg = "Already saved {} files with size {}x{}"
            print(msg.format(index, size, size))


if __name__ == '__main__':
    main()
