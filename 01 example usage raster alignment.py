from Ecobound import align_rasters
import arcpy

# 0 Set environment
arcpy.env.overwriteOutput = True

# 1 Raster alignment
template = r".\output\Barren_testout\Sens_Slope.tif"

raster_list = [
    r".\input\X 1000m\aspectresample.tif",
    r".\input\X 1000m\clay.tif",
    r".\input\X 1000m\demresample.tif",
    r".\input\X 1000m\FHP.tif",
    r".\input\X 1000m\gravel.tif",
    r".\input\X 1000m\pre.tif",
    r".\input\X 1000m\sand.tif",
    r".\input\X 1000m\sloperesample.tif",
    r".\input\X 1000m\soilph.tif",
    r".\input\X 1000m\temp.tif"
]

continuous = [
    True,  # aspect
    True,  # clay
    True,  # dem
    True,  # FHP
    True,  # gravel
    True,  # pre
    True,  # sand
    True,  # slope
    True,  # soil ph
    True   # temp
]

output_path = r".\output\X_alignment"

aligned_files = align_rasters(
    template=template,
    raster_list=raster_list,
    output_path=output_path,
    continuous=continuous
)
