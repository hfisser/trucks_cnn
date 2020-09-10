import os
import numpy as np
import rasterio

main_dir = "F:\\Masterarbeit\\DLR\\project\\1_cnn_truck_detection\\training_data\\data"
directories = [os.path.join(main_dir, x) for x in os.listdir(main_dir)]
subset_size = 36


def subset(d, sub_size):
    dir_out = os.path.join(d, "images")
    file_out = os.path.join(dir_out, os.path.basename(d)+".jpg")
    if not os.path.exists(file_out):
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)
        d1 = os.path.join(d, os.listdir(d)[0], "GRANULE")
        if os.path.exists(d1):
            d2 = os.path.join(os.path.join(d1, os.listdir(d1)[0]), "IMG_DATA", "R10m")
            files = os.listdir(d2)
            band_names = np.array([f.split("_")[2] for f in files])
            data = []
            kwargs = None
            for b in ["B02", "B03", "B04"]:
                fname = files[np.where(band_names == b)[0][0]]
                with rasterio.open(os.path.join(d2, fname)) as r:
                    data.append(r.read(1))
                    kwargs = r.profile
            kwargs.update(count=4,
                          driver="GTiff",
                          height=sub_size,
                          width=sub_size,
                          dtype="uint16")
            # create subsets of the data
            dim = range(sub_size)
            for y in dim:
                for x in dim:
                    sub = []
                    for arr in data:
                        y_up = (y + 1) * sub_size
                        x_up = (x + 1) * sub_size
                        sub.append(arr[y * sub_size: y_up, x * sub_size: x_up])
                    sub = np.array(sub)
                    fname = "_".join([os.path.basename(d), "y" + str(y), "x" + str(x)]) + ".tif"
                    file_out = os.path.join(dir_out, fname)
                    if not os.path.exists(file_out):
                        # check if there is potential for trucks in the subset
                        proceed = can_have_trucks(sub[0], sub[1], sub[2])
                        if proceed:
                            with rasterio.open(file_out, "w", **kwargs) as target:
                                for i, band in enumerate(list(sub)):
                                    target.write(band.astype(np.uint16), i+1)


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
        subset(directory, subset_size)
