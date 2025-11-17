import rasterio 
import numpy as np 
from shapely.geometry import Polygon 
import geopandas as gpd 
from skimage.measure import find_contours
from shapely.geometry.polygon import orient
from osgeo import gdal
import glob
from logger.logger import Logger
logger = Logger.get_logger()

class RasterValidRegionExtractor:
    def __init__(self, tiff_path, output_path):
        self.tiff_path = tiff_path
        self.output_path = output_path
    
    @staticmethod
    def close_ring(coords):
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        return coords
    
    def extract_valid_polygon(self):
        with rasterio .open(self.tiff_path) as src:
            image = src.read(1)
            transform = src.transform
            nodata = src.nodata
            if nodata is not None:
                binary_image = (image!=nodata). astype(np.uint8)
            else:
                nodata = 0
                binary_image = (image!=nodata). astype(np.uint8)
                # binary_image = (~bp.isnan(image)).astype(np.uint8)

            logger.info("extracting contour.")
            external_contours = find_contours(binary_image, level=0.5)
            if not external_contours:
                logger.info("Don't find valid image region.")
                raise
            contour_coords = external_contours[0]

            external_coords = [
                rasterio.transform.xy(transform, row, col, offset='center')
                for row, col in external_contours[0]
            ]
            external_ring = self.close_ring([(x, y) for x, y in external_coords])
            
            holes = []
            if np.any(binary_image == 0):
                inverted_binary_image = 1 - binary_image
                internal_contours = find_contours(inverted_binary_image, level=0.5)

                for contour in internal_contours:
                    internal_coords = [
                        rasterio.transform.xy(transform, row, col, offset='center')
                        for row, col in contour
                    ]
                    hole_ring = self.close_ring([(x, y) for x, y in internal_coords])
                    if Polygon(external_ring).contains(Polygon(hole_ring)) and not Polygon(external_ring).equals(Polygon(hole_ring)):
                        holes.append(hole_ring)
            polygon = Polygon(shell=external_ring, holes=holes)

            logger.info('saving result.')
            gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs=src.crs)
            gdf.to_file(self.output_path)

class VRTGenerator():
    def __init__(self, tiff_dir, output_path, is_overviews=True):
        self.tiff_dir = tiff_dir
        self.output_path = output_path
        self.is_overviews = is_overviews
    
    def create_vrt(self):
        tiff_files = glob.glob(self.tiff_dir + "/*.tif")
        logger.info(f"Totally {len(tiff_files)} tifs will be processed.")
        logger.info("building vrt file.")
        vrt_options = gdal.BuildVRTOptions(srcNodata=0, VRTNodata=0, hideNodata=True)
        gdal.BuildVRT(self.output_path, tiff_files, options=vrt_options)
        if self.is_overviews:
            logger.info("building overviews.")
            vrt_ds = gdal.Open(self.output_path)
            vrt_ds.BuildOverviews('nearest', [2,4,8,16,32,64,128])
            vrt_ds.FlushCache()