import os
import numpy as np
import rasterio
import geopandas as gpd
from shapely import geometry

main_dir = "F:\\Masterarbeit\\DLR\\project\\1_cnn_truck_detection\\training_data\\data"
directories = [os.path.join(main_dir, x) for x in os.listdir(main_dir)]
subset_size = 5490


def subset(d, sub_size):
    dir_out = os.path.join(os.path.dirname(os.path.dirname(d)), "images", "raw")
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    d1 = os.path.join(d, os.listdir(d)[0], "GRANULE")
    if os.path.exists(d1):
        d2 = os.path.join(os.path.join(d1, os.listdir(d1)[0]), "IMG_DATA", "R10m")
        files = os.listdir(d2)
        band_names = np.array([f.split("_")[2] for f in files])
        data = []
        kwargs = None
        bands = ["B04", "B03", "B02"]
        for b in bands:
            fname = files[np.where(band_names == b)[0][0]]
            with rasterio.open(os.path.join(d2, fname)) as r:
                data.append(r.read(1))
                kwargs = r.profile
        dim = range(int(kwargs["height"] / sub_size))
        kwargs.update(count=len(bands),
                      driver="GTiff",
                      height=sub_size,
                      width=sub_size,
                      dtype="uint32")
        transform = kwargs["transform"]
        # create subsets of the data
        for y in dim:
            y_low = y * sub_size
            y_up = (y + 1) * sub_size
            for x in dim:
                x_low = x * sub_size
                x_up = (x + 1) * sub_size
                sub = []
                for arr in data:
                    sub.append(arr[y_low:y_up, x * x_low:x_up])
                x_updated, y_updated = rasterio.transform.xy(transform, x_low, y_low)
                t = list(transform)
                kwargs.update(transform=rasterio.Affine(t[0], t[1], x_updated, t[3], t[4], y_updated))
                sub = np.array(sub)
                fname = "_".join([os.path.basename(d), "y" + str(y_low), "x" + str(x_low)]) + ".tif"
                # create empty GPKG for labeling
                gdf = gpd.GeoDataFrame({"index": [1]},
                                       geometry=[geometry.box(x_updated, y_updated, x_updated + 0.1, y_updated + 0.1)],
                                       crs=str(kwargs["crs"]))
                gdf.to_file(os.path.join(dir_out, fname.split(".")[0]+".gpkg"), driver="GPKG")
                file_out = os.path.join(dir_out, fname)
                if not os.path.exists(file_out):
                    with rasterio.open(file_out, "w", **kwargs) as target:
                        for i, band in enumerate(list(sub)):
                            target.write(band.astype(np.uint32), i+1)


def can_have_trucks(blue, green, red):
    blue = blue.astype(np.float32)
    green = green.astype(np.float32)
    red = red.astype(np.float32)
    bg = ((blue - green) / (blue + green)) > 0.02
    br = ((blue - red) / (blue + red)) > 0.02
    bg_br = bg.astype(np.int) * br.astype(np.int)
    return np.count_nonzero(bg_br) > 0


if __name__ == "__main__":
    for directory in directories:
        print("Processing: " + os.path.basename(directory))
        subset(directory, subset_size)
