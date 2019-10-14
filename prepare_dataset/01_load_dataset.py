import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from skimage import io


def info_from_filename(file, input_path, output_path):
    new_name = file.replace('\\', '/')
    new_name = new_name.replace(input_path, '')
    new_name = new_name.replace('/', ' ')
    new_name = new_name.lstrip()
    new_name = new_name.replace(' ', '_')
    new_name = new_name.replace('.jpeg', '')
    label, split, objid = new_name.split('_')
    objid = objid.replace('.jpeg', '')
    new_path = os.path.join(output_path, new_name + '.npy')

    return objid, label, new_path


def main():
    input_path = sys.argv[1]

    if not os.path.exists(input_path):
        print('%s does not exist' % input_path)
        exit(1)

    if len(sys.argv) == 3:
        output_path = sys.argv[2]
    else:
        output_path = "npy_jpeg/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    files = glob(input_path + '/**/*.jpeg', recursive=True)

    data = []
    for f in files:
        objid, label, new_path = info_from_filename(f, input_path, output_path)
        data.append((objid, label, new_path))

        img = io.imread(f)
        img = img / 255.0
        np.save(new_path, img.astype('float32'))

    df = pd.DataFrame(data, columns=["objid", "label", "file"])
    df.to_csv("dataframe_jpeg.csv", encoding='utf-8', index=False)


if __name__ == '__main__':
    main()
