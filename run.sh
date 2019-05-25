#!/bin/bash
FITS_PATH="F:/merging/fits/"
NPY_PATH="F:/merging/npy/"
IMG_PATH="F:/merging/img/"
BANDS_CFG="rgu"

if [ $1 eq "1" ]
then
    python 01_load_dataset.py $FITS_PATH $NPY_PATH
else
    python 02_resize_images.py $IMG_PATH $BANDS_CFG
fi
