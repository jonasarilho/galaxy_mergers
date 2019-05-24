import sys
import glob
import numpy as np
import pandas as pd
from astropy.io import fits


def convert_to_npy(file, path, out):
    data = {}
    bands = ["g", "i", "r", "u", "z"]
    for band in bands:
        fits_file = file[:-9] + "_" + band + ".fit.gz"
        hdul = fits.open(fits_file, memmap=False)
        hdul.verify('fix')
        data[band] = hdul[0].data
        hdul.close()

    image = np.array([data["g"], data["i"], data["r"], data["u"], data["z"]])
    file_name = out + file[len(path):-9]
    np.save(file_name, image)

    return file_name


def main():

    input_path = sys.argv[1]
    if len(sys.argv) == 3:
        output_path = sys.argv[2]

    else:
        output_path = input_path + "npy/"

    list_files = glob.glob(input_path + '*_g.fit.gz')
    data = []

    for file in list_files[:5]:
        npy_file = convert_to_npy(file, input_path, output_path)
        parsed_file = npy_file[len(output_path):].split("_")
        data.append([parsed_file[2], parsed_file[0], npy_file + ".npy"])

    df = pd.DataFrame(data, columns=["objid", "label", "file"])
    df.to_csv("dataframe.csv", encoding='utf-8', index=False)


if __name__ == '__main__':
    main()
