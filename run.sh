#!/bin/bash
FITS_PATH="F:/merging/fits/"
NPY_PATH="F:/merging/npy/"
IMG_PATH="F:/merging/img32/"
BANDS_CFG="rgu"
IMG_SIZE="32"

if [ $1 eq "1" ]
then
    python 01_load_dataset.py $FITS_PATH $NPY_PATH
elif [ $1 eq "2" ]
then
    python 02_resize_images.py $IMG_PATH $BANDS_CFG $IMG_SIZE
elif [ $1 eq "3" ]
then
    python 03_separate_test_set.py $IMG_PATH $IMG_SIZE
else
    python 01_load_dataset.py $FITS_PATH $NPY_PATH
    python 02_resize_images.py $IMG_PATH $BANDS_CFG $IMG_SIZE
    python 03_separate_test_set.py $IMG_PATH
fi

# python 01_VGG_CNN.py $IMG_PATH $IMG_SIZE
