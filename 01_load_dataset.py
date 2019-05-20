import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
from astropy.io import fits

file_path = "F:/merging/fits/"
file_name = "merger_test_587722982838436141"

bands = ["g", "i", "r", "u", "z"]
data = {}

for band in bands:
    file = file_path + file_name + "_" + band + ".fit.gz"
    hdul = fits.open(file, memmap=False)
    hdul.verify('fix')
    hdul.info()

    # print(hdul[0].header)
    data[band] = hdul[0].data
    hdul.close()

image_data = np.array([data["g"], data["i"], data["r"], data["u"], data["z"]])

print(type(image_data))
print(image_data.shape)
