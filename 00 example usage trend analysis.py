from Ecobound import trend_analysis
import arcpy

# 0. Set environment
arcpy.env.overwriteOutput = True

# 1. Set relative paths
input_folder = r".\input\Barren_resample"
output_folder = r".\output\Barren_testout"
radius = 5000
unit = "MAP"

# 2. Run trend analysis
trend_analysis(input_folder, output_folder, radius, unit)

