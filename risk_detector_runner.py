import arcpy
from Ecobound import adv_risk

def main():
    # 获取参数
    x_folder = arcpy.GetParameterAsText(0)
    y_raster = arcpy.GetParameterAsText(1)
    output_folder = arcpy.GetParameterAsText(2)
    num_bins = int(arcpy.GetParameterAsText(3))
    std_factor = float(arcpy.GetParameterAsText(4))
    sma_short_bins = int(arcpy.GetParameterAsText(5))
    sma_long_bins = int(arcpy.GetParameterAsText(6))
    macd_short_period = int(arcpy.GetParameterAsText(7))
    macd_long_period = int(arcpy.GetParameterAsText(8))
    macd_signal_period = int(arcpy.GetParameterAsText(9))
    k_factor = float(arcpy.GetParameterAsText(10))
    svg_only = arcpy.GetParameter(11)

    arcpy.AddMessage("🚀 Running Advanced Risk Detector...")
    adv_risk(
        x_folder=x_folder,
        y_raster=y_raster,
        output_folder=output_folder,
        num_bins=num_bins,
        std_factor=std_factor,
        sma_short_bins=sma_short_bins,
        sma_long_bins=sma_long_bins,
        macd_short_period=macd_short_period,
        macd_long_period=macd_long_period,
        macd_signal_period=macd_signal_period,
        k_factor=k_factor,
        svg_only=svg_only
    )
    arcpy.AddMessage("✅ Advanced Risk Detection Completed.")

if __name__ == '__main__':
    main()
