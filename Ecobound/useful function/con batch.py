import arcpy
import os
from arcpy.sa import Con

# 设置工作空间
arcpy.CheckOutExtension("Spatial")

#input_folder = r"G:\SJY LEN\SJY LEN landscape pattern\SJY LEN ana\landuse resample SJY" #resample data
input_folder = r"G:\SJY LEN\SYJ LEN slope\input" #30m original resample data



#output_folder = r"G:\SJY LEN\EcoBound\input\grassland"
#output_folder = r"G:\SJY LEN\EcoBound\input\Barren"
output_folder = r"G:\SJY LEN\EcoBound Python Package\input\Grassland_withoutresample"



os.makedirs(output_folder, exist_ok=True)

landID = 4
#landID = 7

landID_where = f"VALUE = {landID}"

# 遍历输入文件夹中的所有.tif文件
for filename in os.listdir(input_folder):
    if filename.endswith(".tif"):
        # 构建输入栅格路径
        in_raster = os.path.join(input_folder, filename)
        
        # 提取年份：假设文件名为 "CLCD_2000.tif"
        basename = os.path.splitext(filename)[0]
        year = basename.split("_")[2]
        #out_filename = f"Grass_{year}.tif"
        out_filename = f"Barren_{year}.tif"
        
        out_raster = os.path.join(output_folder, out_filename)

        # 执行Con操作：将指定 landID 的像元赋值为1，其余为0
        con_ras = Con(in_raster, 1, 0, landID_where)

        # 保存结果栅格
        con_ras.save(out_raster)

        print(f"Processed: {filename} → {out_filename}")

arcpy.CheckInExtension("Spatial")
