import os
import glob
import numpy as np
import rasterio as rio
import geopandas as gpd
from shapely.geometry import Polygon, box

from labeling.array.utils import rescale


dir_main = "F:\\Masterarbeit\\DLR\\project\\1_cnn_truck_detection\\training_data\\data"
dir_raw = os.path.join(os.path.dirname(dir_main), "images", "raw")
dir_ready = os.path.join(os.path.dirname(dir_raw), "ready")

# tile ids to process from dir_raw
tiles = ["T32UNA", "T30TVK", "T36VUM", "T23KLQ", "T36TUU", "T31UDS", "T35UQR", "T19HCC"]
# int xy dimension of output image tiles
tile_size = 640
out_dtype = np.float32
min_rescale = 0.
max_rescale = 1.


def generate_training_imgs(file_image, labels, size, dtype, min, max, dir_out):
    """
    :param file_image: str file path of tif image
    :param labels: GeoDataFrame label boxes
    :param size: int xsy dimension of output image tiles
    :param dir_out: str directory where to save training images
    :return:
    """
    tile_id = os.path.basename(file_image).split("_")[5]
    with rio.open(file_image, "r") as src:
        arr = np.zeros((src.count, src.height, src.width))
        for i in range(src.count):
            arr[i] = src.read(i+1)
        src_meta = src.meta.copy()
        bounds = src.bounds
    n_bands = arr.shape[0]
    src_transform = src_meta["transform"]
    lat, lon = np.arange(bounds.top, bounds.bottom, src_transform[4]), np.arange(bounds.left, bounds.right, src_transform[0])
    boxes = gpd.clip(labels, gpd.GeoDataFrame(geometry=[Polygon(box(bounds[0], bounds[1], bounds[2], bounds[3]))],
                                              crs=labels.crs))
    for row in boxes.iterrows():
        file_out_image = os.path.join(dir_out, "truck_image_" + tile_id + "_BOX" + str(row.name) + ".tif")
        bbox = row.geometry.bounds
        box_indices = min_diff(bbox[1], lat), min_diff(bbox[0], lon), min_diff(bbox[3], lat), min_diff(bbox[2], lon)
        box_shape = box_indices[0] - box_indices[2], box_indices[3] - box_indices[1]
        y_diff = size - box_shape[0]
        x_diff = size - box_shape[1]
        y_diff_round, x_diff_round = int(np.round(y_diff / 2)), int(np.round(x_diff / 2))
        box_indices = np.array(box_indices)
        box_indices[0] += y_diff_round - 1  # +1 because size must be precisely matched
        box_indices[2] -= y_diff - y_diff_round
        box_indices[1] -= x_diff_round
        box_indices[3] += x_diff - x_diff_round - 1
        max_frame = src_meta["height"]  # equal for x and y
        for j in range(len(box_indices)):
            box_indices[j] = fix_zero(box_indices[j])
            box_indices[j] = fix_max(box_indices[j], max_frame)
        box_indices = fix_frame(box_indices, max_frame, size)
        arr_box = np.zeros((n_bands, box_indices[0] - box_indices[2] + 1, box_indices[3] - box_indices[1] + 1))
        for z in range(n_bands):
            arr_box[z] = arr[z, box_indices[2]:box_indices[0] + 1, box_indices[1]:box_indices[3] + 1]
        tgt_meta = src_meta.copy()
        tgt_transform = list(src_transform)
        tgt_transform[2], tgt_transform[5] = rio.transform.xy(src_transform, box_indices[2], box_indices[1])
        tgt_transform = rio.transform.Affine(tgt_transform[0], tgt_transform[1], tgt_transform[2], tgt_transform[3],
                                             tgt_transform[4], tgt_transform[5])
        tgt_meta.update(height=arr_box.shape[1],
                        width=arr_box.shape[2],
                        transform=tgt_transform,
                        dtype=dtype)
        arr_box[np.isnan(arr_box)] = 0.
        with rio.open(file_out_image, "w") as tgt:
            for i in range(n_bands):
                arr_out = rescale(arr_box[i].astype(dtype), min, max)
                tgt.write(arr_out, i+1)


def min_diff(a, value):
    diff = np.abs(a - value)
    return int(np.where(diff == diff.min())[0][0])


def fix_zero(value):
    if value < 0:
        return 0
    else:
        return value


def fix_max(value, max_value):
    if value > max_value:
        return max_value
    else:
        return value


def fix_frame(box, max_value, size):
    """
    :param box: list input box bottom,left,up,right
    :param max_value: int max width and/or height of data
    :param size: int side length of frame
    :return: list altered box
    """
    x_size = box[3] - box[1] + 1
    y_size = box[0] - box[2] + 1
    x_diff = size - x_size
    y_diff = size - y_size
    counterpart = [2, 3, 0, 1]
    diffs = [y_diff, x_diff, y_diff, x_diff]
    exceeds = [v == max_value or v == 0 for v in box]
    for cond, diff, i in zip(exceeds, diffs, counterpart):
        if cond and diff > 0:
            box[i] = box[i] - diff if i in [1, 2] else box[i] + diff
    return box


if __name__ == "__main__":
    if not os.path.exists(dir_ready):
        os.mkdir(dir_ready)
    for tile in tiles:
        file_str = dir_raw + os.sep + "*" + tile + "*_y*_x"
        files_img = glob.glob(file_str + "*.tif")  # road-masked data
        label_boxes = gpd.read_file(glob.glob(file_str[0:-6] + "*.gpkg")[0])
        for file_img in files_img:
            generate_training_imgs(file_img,
                                   label_boxes,
                                   tile_size,
                                   out_dtype,
                                   min_rescale,
                                   max_rescale,
                                   dir_ready)



