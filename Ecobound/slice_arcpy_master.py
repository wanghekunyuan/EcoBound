import arcpy
from arcpy.sa import Slice
from arcpy.sa import Int
import os


# 启用覆盖输出
arcpy.env.overwriteOutput = True

def slice_rasters(raster_list, slice_types=None, number_zones=None, output_dir=None, message_callback=print):
    def notify(msg):
        try:
            message_callback(msg)
        except:
            notify(msg)
    # 设置默认值
    if slice_types is None:
        slice_types = ["EQUAL_INTERVAL", "EQUAL_AREA", "NATURAL_BREAKS", "GEOMETRIC_INTERVAL"]
    if number_zones is None:
        number_zones = [3, 4, 5]

    sliced_rasters = []
    
    slice_type_short = {
        "EQUAL_INTERVAL": "ei",
        "EQUAL_AREA": "ea",
        "NATURAL_BREAKS": "nb",
        "GEOMETRIC_INTERVAL": "gi"
    }

    # 检查并设置输出目录
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    arcpy.env.workspace = output_dir

    for raster in raster_list:
        raster_name = os.path.splitext(os.path.basename(raster))[0]
        for zones in number_zones:
            for s_type in slice_types:
                short = slice_type_short.get(s_type, "unknown")
                out_raster_name = f"{raster_name}_{zones}_{short}.tif"
                out_raster_path = os.path.join(output_dir, out_raster_name) if output_dir else out_raster_name

                # 执行 Slice 函数
                out_slice = Slice(raster, zones, s_type)
                out_slice.save(out_raster_path)

                sliced_rasters.append(out_raster_path)
                notify(f"Sliced raster ready: {out_raster_path}")

    # 删除无效分割
    rasters_to_delete = []
    for raster_path in sliced_rasters:
        try:
            raster = arcpy.Raster(raster_path)
            desc = arcpy.Describe(raster)
            if desc.pixelType in ["SInt", "UInt"]:
                int_raster = raster
            else:
                int_raster = Int(raster)

            temp_table = "in_memory/temp_table"
            if arcpy.Exists(temp_table):
                arcpy.Delete_management(temp_table)

            arcpy.sa.ZonalStatisticsAsTable(int_raster, "Value", int_raster, temp_table, "DATA", "ALL")

            unique_values = sorted(set(row[0] for row in arcpy.da.SearchCursor(temp_table, ["Value"])))
            num_strata = len(unique_values)
            unique_counts = sorted(set(row[0] for row in arcpy.da.SearchCursor(temp_table, ["COUNT"])))

            if num_strata < 2 or 1 in unique_counts:
                notify(f"Sliced raster has been deleted with slice count = 1 or any sliced raster count = 1: {raster_path}")
                rasters_to_delete.append(raster_path)

            arcpy.management.Delete(temp_table)

        except Exception as e:
            notify(f"Error processing {raster_path}: {e}")
            continue

    for raster_path in rasters_to_delete:
        sliced_rasters.remove(raster_path)
        arcpy.management.Delete(raster_path)

    return sliced_rasters

if __name__ == "__main__":
    # 示例输入
    raster_list = [
        r"C:\Users\yuan wang\OneDrive\geodect\X without reclass resample\Aspect.tif",
        r"C:\Users\yuan wang\OneDrive\geodect\X without reclass resample\Soil.tif"
    ]
    output_dir = r"G:\Yangtze vulnerability\myana\out6"
    

    # 进行栅格分割
    #可选参数
    #slice_types = ["GEOMETRIC_INTERVAL"]
    #number_zones = [6]
    
    #sliced_rasters = slice_rasters(raster_list, slice_types, number_zones, output_dir=output_dir)

    #默认参数
    sliced_rasters = slice_rasters(raster_list, output_dir=output_dir)

