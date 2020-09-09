# Images in a directory are read and plotted consecutively.
# You decide if the image should be retained or deleted

import os, shutil
from glob import glob
import numpy as np
import rasterio
import rasterio.plot as rp

main_dir = "F:\\Masterarbeit\\DLR\\project\\1_cnn_truck_detection\\training_data\\images"


# plot image by image and decide if image should be dropped or retained
def drop_or_not(d):
    dir_out = os.path.join(d, "checked")
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    files = [os.path.join(d, f) for f in glob(d+os.sep+"*.tif")]
    for i, f in enumerate(files):
        f_out = os.path.join(dir_out, os.path.basename(f).split(".")[0] + "_checked.tif")
        if os.path.exists(f_out):
            continue
        else:
            print("File %s/%s" %(i,len(files)))
            with rasterio.open(f, "r") as src:
                bgr = src.read(1), src.read(2), src.read(3)
                bgr_norm = normalize(bgr[0]), normalize(bgr[1]), normalize(bgr[2])
                rgb_stack = np.array([bgr_norm[2], bgr_norm[1], bgr_norm[0]])
            rp.show(rgb_stack)
            print("Drop or not? Press ENTER for dropping the image")
            in1 = input()
            if in1 == "":
                print("Dropping")
            else:
                print("Ok")
                shutil.copyfile(f, f_out)
            if os.path.exists(f_out):
                os.remove(f)


def normalize(array):
    array_min = array.min()
    return (array - array_min) / (array.max() - array_min)


if __name__ == "__main__":
    drop_or_not(main_dir)
