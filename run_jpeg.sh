#!/bin/bash
FITS_PATH="F:/merging/dataset/"
NPY_PATH="F:/merging/npy_dataset/"
IMG_PATH="F:/merging/img64/"
IMG_SIZE="64"

if [ $1 eq "1" ]
then
    python 01_load_dataset_jpeg.py $FITS_PATH $NPY_PATH
elif [ $1 eq "2" ]
then
    python 02_resize_images_jpeg.py $IMG_PATH $IMG_SIZE
elif [ $1 eq "3" ]
then
    python 03_separate_test_set.py $IMG_PATH
else
    python 01_load_dataset_jpeg.py $FITS_PATH $NPY_PATH
    python 02_resize_images_jpeg.py $IMG_PATH $IMG_SIZE
    python 03_split_dataset_jpeg.py $IMG_PATH
fi

# python 01_VGG_CNN.py $IMG_PATH $IMG_SIZE
# python 02_resize_images.py "F:/merging/img48rrr/" "rrr" "48"
# python 03_separate_test_set.py "F:/merging/img32rrr/"
# python 03_split_dataset_jpeg.py "F:/merging/img64/" "3" "64"

#python 01_load_dataset_jpeg.py "F:/merging/dataset/" "F:/merging/npy_dataset/"
# python 02_resize_images_jpeg.py "F:/merging/jpeg64/" "64"
# python 03_split_dataset_jpeg.py "F:/merging/jpeg64/" "3" "64"

#python 01_load_dataset_jpeg.py "F:/merging/dataset/" "F:/merging/npy_dataset/"
# python 02_resize_images_jpeg.py "F:/merging/jpeg96/" "96"
#python 03_split_dataset_jpeg.py "F:/merging/jpeg96/" "3" "96"
