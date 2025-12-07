# example_usage_alignment.py  (for Ecobound v3.20)
# Purpose:
#   Align all X rasters to the template raster using the new alignment policy.
#   Default policy is "template_only" (recommended).
#
# Key changes in v3.20:
#   - New parameter: align_policy = {"template_only", "global_intersection"}
#       * "template_only" (default): mask by template's valid area only; NoData does NOT propagate across variables.
#       * "global_intersection"     : legacy behavior; any NoData in any X propagates to all outputs.
#   - All messages and ValueError strings are in English.

from Ecobound import align_rasters
import arcpy
import os

# 0) Set ArcGIS environment
arcpy.env.overwriteOutput = True

# 1) Paths (edit these to your project)
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

# 2) Mark each raster as continuous (True) or categorical (False)
#    Continuous rasters use CUBIC resampling; categorical use NEAREST.
#    If all are continuous, you can simplify as: continuous = [True] * len(raster_list)
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
os.makedirs(output_path, exist_ok=True)

# 3) Run alignment
#    - message_callback=arcpy.AddMessage ensures messages appear in the ArcGIS tool pane.
#    - align_policy omitted → defaults to "template_only" (recommended).
aligned_files = align_rasters(
    template=template,
    raster_list=raster_list,
    output_path=output_path,
    continuous=continuous,
    message_callback=arcpy.AddMessage
    # align_policy="template_only"  # (default) mask by template only
)

# (Optional) Echo results to the pane
for p in aligned_files:
    arcpy.AddMessage(f"Aligned file: {p}")

# --- How to switch to the legacy behavior (NOT recommended) ---
# aligned_files = align_rasters(
#     template=template,
#     raster_list=raster_list,
#     output_path=output_path,
#     continuous=continuous,
#     message_callback=arcpy.AddMessage,
#     align_policy="global_intersection"  # legacy: NoData in any X propagates to all outputs
# )
