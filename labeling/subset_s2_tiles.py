import os
import rasterio
import utm
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely import geometry
from rasterio.transform import Affine
from labeling.osm.utils import get_roads, rasterize_osm
from labeling.array.utils import rescale

main_dir = "F:\\Masterarbeit\\DLR\\project\\1_cnn_truck_detection\\training_data\\data"
directories = [os.path.join(main_dir, x) for x in os.listdir(main_dir)]
number_subsets = 4
roads_buffer = 40


def subset(d, n_subs, osm_buffer):
    tgt_crs = "EPSG:4326"
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
        bands = ["B04", "B03", "B02", "B08"]
        file_stack = os.path.join(dir_out, os.path.basename(d) + "_EPSG4326.tif")
        src = rasterio.open(os.path.join(d2, files[0]))
        kwargs = src.meta.copy()
        src_crs = src.crs
        kwargs.update({"count": len(bands), "driver": "GTiff"})
        with rasterio.open(file_stack, "w", **kwargs) as tgt:
            for i, b in enumerate(bands):
                fname = files[np.where(band_names == b)[0][0]]
                with rasterio.open(os.path.join(d2, fname)) as src:
                    tgt.write(src.read(1), i+1)
        src_utm = src.crs.to_wkt().split("/ UTM zone ")[1].split(",")[0][0:-1]
        a, b = src.transform * [0,0], src.transform * [src.height, src.width]
        src_corners = np.array([a[0], b[1], b[0], a[1]]).flatten()
        utm_number, utm_letter = int(src_utm[0:-1]), src_utm[-1]
        y0, x0 = utm.to_latlon(src_corners[0], src_corners[3], utm_number, utm_letter)
        y1, x1 = utm.to_latlon(src_corners[2], src_corners[1], utm_number, utm_letter)
        bbox_epsg4326 = [y1, x0, y0, x1]
        fname = os.path.basename(d) + "_.tif"
        fname_pure = fname.split(".")[0]
        gpkg_out = os.path.join(dir_out, fname_pure + ".gpkg")
        gdf = gpd.GeoDataFrame({"index": [1]},
                               geometry=[geometry.box(src_corners[0], src_corners[1], src_corners[2], src_corners[3])],
                               crs=str("EPSG:4326"))
        gdf.to_file(gpkg_out, driver="GPKG")
        # get OSM data and mask data to roads
        osm_file = get_roads(bbox_epsg4326, ["motorway", "trunk", "primary"], osm_buffer,
                             dir_osm, fname_pure + "osm_roads", str(src_crs))
        bbox_epsg4326 = src_corners
        osm_vec = gpd.read_file(osm_file)
        n = n_subs / 2
        stack = rasterio.open(file_stack, "r")
        n_bands = len(bands)
        h, w = stack.height, stack.width
        data = np.zeros((n_bands, h, w), dtype=np.float32)
        for i in range(n_bands):
            data[i] = stack.read(i+1)
        tgt_pixels_y = int(h/n)
        tgt_pixels_x = int(w/n)
        src_lat = get_lat(bbox_epsg4326, h, (bbox_epsg4326[3]-bbox_epsg4326[1]) / h)
        src_lon = get_lon(bbox_epsg4326, w, (bbox_epsg4326[2]-bbox_epsg4326[0]) / w)
        dim = range(int(n))
        for y in dim:
            for x in dim:
                file_out = os.path.join(dir_out, "_".join([fname_pure, "y"+str(y), "x"+str(x)])+".tif")
                y1, y2 = int(y*tgt_pixels_y), int((y+1)*tgt_pixels_y)
                x1, x2 = int(x*tgt_pixels_x), int((x+1)*tgt_pixels_x)
                tgt_lat = src_lat[y1:y2]
                tgt_lon = src_lon[x1:x2]
                ref_xr = xr.DataArray(data=np.zeros((len(tgt_lat), len(tgt_lon))),
                                      coords={"lat": tgt_lat, "lon": tgt_lon},
                                      dims=["lat", "lon"])
                osm_raster = rasterize_osm(osm_vec, ref_xr)
                osm_raster[osm_raster != 0] = 1
                osm_raster[osm_raster == 0] = np.nan
                t = stack.transform
                transform = Affine(t[0], t[1], tgt_lon[x], t[3], t[4], tgt_lon[y])
                kwargs.update({"crs": tgt_crs, "transform": transform,
                               "height": tgt_pixels_y, "width": tgt_pixels_x,
                               "dtype": np.float32})
                with rasterio.open(file_out, "w", **kwargs) as tgt:
                    for i in range(n_bands):
                        data_band = rescale(data[i, y1:y2, x1:x2].astype(np.float32), 0., 1.)
                        tgt.write((data_band * osm_raster).astype(np.float32), i+1)


def get_lat(bbox, h, step=None):
    if step is None:
        step = (bbox[3] - bbox[1]) / h
        stop = bbox[3]
    else:
        stop = bbox[3] + step
    return np.flip(np.arange(bbox[1], stop, step))


def get_lon(bbox, w, step=None):
    if step is None:
        step = (bbox[2] - bbox[0]) / w
        stop = bbox[2]
    else:
        stop = bbox[2] + step
    return np.arange(bbox[0], stop, step)


def transform_to_bbox(kwargs):
    # W,S,E,N
    this_transform = kwargs["transform"]
    upper_y_deg, lower_x_deg = this_transform[5], this_transform[2]
    res_deg = this_transform[0]
    x_deg = kwargs["width"] * res_deg
    y_deg = kwargs["height"] * res_deg
    return [lower_x_deg, upper_y_deg - y_deg, lower_x_deg + x_deg, upper_y_deg]


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
        subset(directory, number_subsets, roads_buffer)
