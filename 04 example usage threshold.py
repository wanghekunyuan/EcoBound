from Ecobound import ecobound_threshold
import arcpy

# 0 Set environment
arcpy.env.overwriteOutput = True

# 1 Test run for EcoBound threshold analysis (Example usage)
ecobound_threshold(
    x_folder = r".\output\X_alignment",
    y_raster = r".\output\Barren_testout\Sens_Slope.tif",
    output_folder = r".\output\ecobound",
    num_bins = 100,     # Number of bins for first-layer segmentation (entropy scan)
    b_bins = 30,        # Number of bins for second-layer evaluation (used for VR calculation)
    permutations = 0,   # Permutation test iterations (set to 0 to skip significance test)
    svg_only = True,
    ecobound = True
)
