import os
import sys
import time


# 0 指定ngdpy包的路径
ngdpy_path = "C:/Users/yuan wang/OneDrive/geodect"
sys.path.append(ngdpy_path)

# 用户选择环境变量
os.environ['NGDPY_ENV'] = "arcpy"

import ngdpy as ng
import arcpy

# 1 用户定义
#10 定义输入输出
arcpy.env.overwriteOutput = True

#定义输入文件
# 所有X和Y的路径，存入一个List
raster_list = [
    r"G:\SJY LEN\SJY Geodector\input\aspectresample.tif",
    r"G:\SJY LEN\SJY Geodector\input\clay.tif",
    r"G:\SJY LEN\SJY Geodector\input\demresample.tif",
    r"G:\SJY LEN\SJY Geodector\input\FHP.tif",
    r"G:\SJY LEN\SJY Geodector\input\gravel.tif",
    r"G:\SJY LEN\SJY Geodector\input\pre.tif",
    r"G:\SJY LEN\SJY Geodector\input\sand.tif",
    r"G:\SJY LEN\SJY Geodector\input\sloperesample.tif",
    r"G:\SJY LEN\SJY Geodector\input\soilph.tif",
    r"G:\SJY LEN\SJY Geodector\input\temp.tif"
]

#定义输出文件夹
outpath = r"G:\SJY LEN\EcoBound Python Package\output\X_alignment"

#11 定义参数
#定义模板栅格
template = r"G:\SJY LEN\EcoBound Python Package\output\Barren\Sens_Slope.tif"

#每个栅格对应的数据类型：True 为连续数据，False 为分类数据，顺序与raster_list一致
continuous = [
    True, #aspect
    True, #clay
    True, #dem
    True, #FHP
    True, #gravel
    True, #pre
    True, #sand
    True, #slope
    True, #soil ph
    True, #temp
]

#执行raster alignment 函数

# 开始计时
start_time = time.time()

ng.keep_extent(template, raster_list, outpath, continuous)

# 结束计时并打印运行时间
end_time = time.time()
print(f"Total execution time: {end_time - start_time} seconds")
