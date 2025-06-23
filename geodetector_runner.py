# geodetector_runner.py
import arcpy
from Ecobound import run_geodetector

def script_tool():
    y_raster = arcpy.GetParameterAsText(0)
    x_aligned_folder = arcpy.GetParameterAsText(1)
    output_dir = arcpy.GetParameterAsText(2)
    slice_types_str = arcpy.GetParameterAsText(3)
    number_zones_str = arcpy.GetParameterAsText(4)
    mode = arcpy.GetParameterAsText(5)
    alpha = float(arcpy.GetParameterAsText(6))
    modules_str = arcpy.GetParameterAsText(7)

    # 解析列表参数
    slice_types = [s.strip() for s in slice_types_str.split(";")] if slice_types_str else ["GEOMETRIC_INTERVAL"]
    number_zones = [int(z.strip()) for z in number_zones_str.split(",")] if number_zones_str else [3, 4, 5]
    modules = [m.strip() for m in modules_str.split(";")] if modules_str else ["factor", "interaction", "risk", "eco"]

    def notify(msg):
        arcpy.AddMessage(msg)

    run_geodetector(
        y_raster=y_raster,
        x_aligned_folder=x_aligned_folder,
        output_dir=output_dir,
        slice_types=slice_types,
        number_zones=number_zones,
        mode=mode,
        alpha=alpha,
        modules=modules,
        message_callback=notify
    )

if __name__ == "__main__":
    script_tool()
