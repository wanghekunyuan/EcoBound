import arcpy
import numpy as np
from scipy.ndimage import label
from arcpy.sa import *

# 允许Spatial Analyst扩展模块和3D Analyst扩展模块
arcpy.CheckOutExtension("Spatial")
arcpy.CheckOutExtension("3D")

# 允许覆盖输出
arcpy.env.overwriteOutput = True

def threshold_segmentation(input_raster, threshold):
    """根据阈值将栅格分为两个区域，并确保区域完整性"""
    # 读取栅格数据为numpy数组
    raster = arcpy.RasterToNumPyArray(input_raster, nodata_to_value=np.nan)
    
    # 打印读取的数据形状以帮助调试
    print(f"Raster shape after reading: {raster.shape}")

    # 创建一个掩码，标识无效值
    mask_valid = ~np.isnan(raster)
    valid_raster = np.where(mask_valid, raster, np.nan)  # 使用np.nan填充无效值，保持二维结构

    # 创建结果数组，初始化为np.nan
    result = np.full_like(valid_raster, np.nan)

    # 根据阈值分割
    low_mask = valid_raster <= threshold
    high_mask = valid_raster > threshold
    
    # 标记连通区域
    low_labeled, num_low_labels = label(low_mask)
    high_labeled, num_high_labels = label(high_mask)

    # 选择面积最大的连通区域
    if num_low_labels > 0:
        largest_low_label = np.argmax(np.bincount(low_labeled.flat)[1:]) + 1
        result[low_labeled == largest_low_label] = 1
    
    if num_high_labels > 0:
        largest_high_label = np.argmax(np.bincount(high_labeled.flat)[1:]) + 1
        result[high_labeled == largest_high_label] = 2
    
    return result

def save_raster(array, template_raster_path, output_raster_path):
    """保存numpy数组为栅格"""
    template_raster = arcpy.Raster(template_raster_path)
    spatial_ref = template_raster.spatialReference
    lower_left = arcpy.Point(template_raster.extent.XMin, template_raster.extent.YMin)
    cell_size = template_raster.meanCellWidth

    # 创建输出栅格，使用np.nan作为NoData值
    out_raster = arcpy.NumPyArrayToRaster(array, lower_left, cell_size, cell_size, value_to_nodata=np.nan)
    arcpy.DefineProjection_management(out_raster, spatial_ref)
    out_raster.save(output_raster_path)

def extract_and_smooth_boundary(original_raster, segmented_raster, threshold, output_boundary_shapefile):
    """提取并平滑栅格数据中值为1和2的边界线，并保存为Shapefile"""
    # 将浮点型栅格值转换为整型
    int_raster = Int(Raster(segmented_raster))

    # 将值为1的区域转换为多边形
    region1_raster = SetNull(int_raster != 1, int_raster)
    region1_polygon = "in_memory/region1_polygon"
    arcpy.RasterToPolygon_conversion(region1_raster, region1_polygon, "NO_SIMPLIFY", "Value")

    # 将值为2的区域转换为多边形
    region2_raster = SetNull(int_raster != 2, int_raster)
    region2_polygon = "in_memory/region2_polygon"
    arcpy.RasterToPolygon_conversion(region2_raster, region2_polygon, "NO_SIMPLIFY", "Value")

    # 提取两个多边形的边界
    region1_boundary = "in_memory/region1_boundary"
    region2_boundary = "in_memory/region2_boundary"
    arcpy.PolygonToLine_management(region1_polygon, region1_boundary)
    arcpy.PolygonToLine_management(region2_polygon, region2_boundary)

    # 找到两个多边形边界的交界线
    initial_boundary_intersect = "in_memory/initial_boundary_intersect"
    arcpy.Intersect_analysis([region1_boundary, region2_boundary], initial_boundary_intersect, "ALL", "", "LINE")

    # 提取自然间断点处的等值线
    contour_lines = "in_memory/contour_lines"
    arcpy.ddd.ContourList(original_raster, contour_lines, [threshold])

    # 创建图层
    arcpy.management.MakeFeatureLayer(initial_boundary_intersect, "initial_boundary_layer")
    arcpy.management.MakeFeatureLayer(contour_lines, "contour_layer")

    # 找到等值线和初始交界线的相交部分
    arcpy.management.SelectLayerByLocation("contour_layer", "INTERSECT", "initial_boundary_layer")

    # 将选中的部分保存为最终结果
    arcpy.management.CopyFeatures("contour_layer", output_boundary_shapefile)

    # 删除过程数据
    arcpy.Delete_management("in_memory/region1_polygon")
    arcpy.Delete_management("in_memory/region2_polygon")
    arcpy.Delete_management("in_memory/region1_boundary")
    arcpy.Delete_management("in_memory/region2_boundary")
    arcpy.Delete_management("in_memory/initial_boundary_intersect")
    arcpy.Delete_management("in_memory/contour_lines")

def generate_natural_boundary(input_raster, break_value, output_boundary_shapefile):
    """根据输入的栅格和间断值生成自然地理界限"""
    segmented_raster_path = "in_memory/segmented_raster"
    
    # 执行阈值分割
    result_array = threshold_segmentation(input_raster, break_value)
    
    # 保存分割结果为中间栅格
    save_raster(result_array, input_raster, segmented_raster_path)
    
    # 提取并平滑自然地理界线
    extract_and_smooth_boundary(input_raster, segmented_raster_path, break_value, output_boundary_shapefile)

if __name__ == "__main__":
    # 示例用法
    input_raster_path = r"G:\Yangtze vulnerability\myana\input\EVI_mytest2.tif"
    output_boundary_shapefile_path = r"G:\Yangtze vulnerability\myana\output\natural_boundary_line.shp"

    # 间断值，只处理一个间断值
    break_value = 0.450

    # 生成自然地理界限
    generate_natural_boundary(input_raster_path, break_value, output_boundary_shapefile_path)

    # 释放Spatial Analyst扩展模块和3D Analyst扩展模块
    arcpy.CheckInExtension("Spatial")
    arcpy.CheckInExtension("3D")
