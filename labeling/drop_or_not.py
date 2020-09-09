# Images in a directory are read and plotted consecutively.
# You decide if the image should be retained or deleted

import os, time
from glob import glob
import numpy as np
import rasterio
import rasterio.plot as rp

main_dir = "F:\\Masterarbeit\\DLR\\project\\1_cnn_truck_detection\\training_data\\images"

def drop_or_not(main_dir):
    files = [os.path.join(main_dir, f) for f in glob(main_dir+os.sep+"*.tif")]
    files = files[3:10]
    for i, f in enumerate(files):
        print("File %s/%s" %(i,len(files)))
        with rasterio.open(f, "r") as src:
            bgr = src.read(1), src.read(2), src.read(3)
            bgr_norm = normalize(bgr[0]), normalize(bgr[1]), normalize(bgr[2])
            rgb_stack = np.array([bgr_norm[2], bgr_norm[1], bgr_norm[0]])
        rp.show(rgb_stack)
        print("Drop or not? Press ENTER for dropping the image")
        in1 = input()
        if in1 == "":
            os.remove(f)
            if not os.path.exists(f):
                print("Deleted")
        else:
            print("Ok")

def normalize(array):
    array_min = array.min()
    return (array - array_min) / (array.max() - array_min)

if __name__ == "__main__":
    drop_or_not(main_dir)