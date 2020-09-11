import os
import numpy as np
import rasterio
import geopandas as gpd
import osr
import utm
from shapely import geometry
from obspy.geodetics.base import kilometers2degrees
from rasterio.crs import CRS

from labeling.osm.utils import get_roads


main_dir = "F:\\Masterarbeit\\DLR\\project\\1_cnn_truck_detection\\training_data\\data"
directories = [os.path.join(main_dir, x) for x in os.listdir(main_dir)]
subset_size = 5490


def subset(d, sub_size):
    home = os.path.dirname(os.path.dirname(d))
    dir_out = os.path.join(home, "images", "raw")
    dir_osm = os.path.join(home, "osm_data")
    for this_dir in [dir_out, dir_osm]:
        if not os.path.exists(this_dir):
            os.mkdir(this_dir)
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
                      dtype="uin16")
        transform = kwargs["transform"]
        crs_origin = kwargs["crs"]
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
                t = list(transform)
                x_low_utm, y_low_utm = rasterio.transform.xy(transform, x_low, y_low)
                ref = osr.SpatialReference()
                ref.ImportFromEPSG(crs_origin.to_epsg())
                utm_zone = ref.ExportToWkt().split("UTM zone ")[1].split(",GEOGCS")[0].split('"')[0]
                upper_y_deg, lower_x_deg = utm.to_latlon(x_low_utm, y_low_utm, int(utm_zone[0:-1]), utm_zone[-1])
                kwargs.update(transform=rasterio.Affine(kilometers2degrees(0.01), t[1], lower_x_deg, t[3],
                                                        -kilometers2degrees(0.01), upper_y_deg),
                              crs=CRS.from_epsg(4326),
                              dtype=np.uint16)
                sub = np.array(sub)
                fname = "_".join([os.path.basename(d), "y" + str(y_low), "x" + str(x_low)]) + ".tif"
                file_out = os.path.join(dir_out, fname)
                if not os.path.exists(file_out):
                    bbox = transform_to_bbox(kwargs)
                    # create empty GPKG for labeling
                    gdf = gpd.GeoDataFrame({"index": [1]},
                                           geometry=[geometry.box(bbox[0], bbox[1], bbox[2], bbox[3])],
                                           crs=str("EPSG:4326"))
                    fname_pure = fname.split(".")[0]
                    gpkg_out = os.path.join(dir_out, fname_pure + ".gpkg")
                    if not os.path.exists(gpkg_out):
                        gdf.to_file(gpkg_out, driver="GPKG")
                    osm_file = get_roads(bbox, ["motorway", "trunk", "primary"], dir_osm, fname_pure + "_osm_roads")
                    osm_vec = gpd.read_file(osm_file)
                    osm_raster = rasterize_osm(osm_vec, )
                    with rasterio.open(file_out, "w", **kwargs) as target:
                        for i, band in enumerate(list(sub)):
                            band_roads = band.astype(np.uint16) #* road_mask
                            target.write(band_roads, i+1)


def transform_to_bbox(kwargs):
    # W,S,E,N
    this_transform = kwargs["transform"]
    upper_y_deg, lower_x_deg = this_transform[5], this_transform[2]
    x_meters = kwargs["width"] * this_transform[0]
    y_meters = kwargs["height"] * this_transform[0]
    x_degrees, y_degrees = kilometers2degrees(x_meters/1000), kilometers2degrees(y_meters/1000)
    return [lower_x_deg, upper_y_deg - y_degrees, lower_x_deg + x_degrees, upper_y_deg]


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
