import arcpy
import os

# 定义输入和输出文件夹路径
input_folder = r"G:\SJY LEN\EcoBound Python Package\input\grassland"
#input_folder = r"G:\SJY LEN\EcoBound Python Package\input\X"
output_folder = r"G:\SJY LEN\EcoBound Python Package\input\Grassland_resample"
#output_folder = r"G:\SJY LEN\EcoBound Python Package\input\X_resample"

# 设置重采样的单元大小和重采样类型
cell_size = "1000 1000"
resampling_type = "MAJORITY"
#resampling_type = "BILINEAR"

# 遍历输入文件夹中的所有栅格文件
for filename in os.listdir(input_folder):
    if filename.endswith(".tif"):
        # 构建输入和输出栅格文件的路径
        in_raster = os.path.join(input_folder, filename)
        out_raster = os.path.join(output_folder, filename)
        
        # 执行重采样
        arcpy.management.Resample(
            in_raster=in_raster,
            out_raster=out_raster,
            cell_size=cell_size,
            resampling_type=resampling_type
        )
        print(f"已完成重采样: {filename}")
        
print("所有栅格文件的重采样已完成。")
