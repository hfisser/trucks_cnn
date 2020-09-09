# postprocess the GeoJSON that contains the labels

import os
import geopandas as gpd

main_dir = "F:\\Masterarbeit\\DLR\\project\\1_cnn_truck_detection\\training_data\\images"
annots_dir = os.path.join(main_dir, "annotations")


def pp_labels(d, ad):
    imgs = glob(ad+os.sep+"*.tif")


if __name__ == "__main__":
    pp_labels(main_dir, annots_dir)
