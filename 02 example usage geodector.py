
from Ecobound import run_geodetector
import arcpy

# 0 Set environment
arcpy.env.overwriteOutput = True

# 1 Run GeoDetector
run_geodetector(
    y_raster = r".\output\Barren_testout\Sens_Slope.tif",
    x_aligned_folder = r".\output\X_alignment",
    output_dir = r".\output\geodetector",
    slice_types = ["EQUAL_INTERVAL", "EQUAL_AREA", "NATURAL_BREAKS", "GEOMETRIC_INTERVAL"],
    number_zones = [3, 4, 5],
    mode = "qmax",
    alpha = 0.05,
    modules = ["factor", "interaction", "risk", "eco"]
)
