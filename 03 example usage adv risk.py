from Ecobound import adv_risk
import arcpy

# 0 Set environment
arcpy.env.overwriteOutput = True

# 1 Run advanced risk detector
adv_risk(
    x_folder = r".\output\X_alignment",
    y_raster = r".\output\Barren_testout\Sens_Slope.tif",
    output_folder = r".\output\adv_risk",
    num_bins = 100,
    std_factor = 1.5,
    sma_short_bins = 10,
    sma_long_bins = 20,
    macd_short_period = 12,
    macd_long_period = 26,
    macd_signal_period = 9,
    k_factor = 1.0,
    svg_only = True    
)



