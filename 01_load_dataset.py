import numpy as np
# import matplotlib
import glob
# import matplotlib.pyplot as plt
from astropy.io import fits

file_path = "F:/merging/fits/"
out_path = "F:/merging/npy/"
ex_file_name = "merger_test_587722982838436141"

list_files = glob.glob(file_path + '*_g.fit.gz')

for file in list_files[:10]:
    data = {}
    bands = ["g", "i", "r", "u", "z"]
    for band in bands:
        fits_file = file[:-9] + "_" + band + ".fit.gz"
        hdul = fits.open(fits_file, memmap=False)
        hdul.verify('fix')
        # hdul.info()

        # print(hdul[0].header)
        data[band] = hdul[0].data
        hdul.close()

    image = np.array([data["g"], data["i"], data["r"], data["u"], data["z"]])

    file_name = file[len(file_path):-9]
    np.save(out_path + file_name, image)
    # print(type(image))
    # print(image.shape)
