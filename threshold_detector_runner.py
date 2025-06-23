import arcpy
from Ecobound import ecobound_threshold

def main():
    # 参数输入
    x_folder = arcpy.GetParameterAsText(0)  # 输入 X 文件夹
    y_raster = arcpy.GetParameterAsText(1)  # 输入 Y 栅格
    output_folder = arcpy.GetParameterAsText(2)  # 输出文件夹
    num_bins = int(arcpy.GetParameterAsText(3))  # 熵增扫描分箱数
    b_bins = int(arcpy.GetParameterAsText(4))    # ΔV 分箱数
    permutations = int(arcpy.GetParameterAsText(5))  # 置换次数
    generate_boundary = arcpy.GetParameter(6)    # 是否生成自然边界线

    # 执行主函数
    ecobound_threshold(
        x_folder=x_folder,
        y_raster=y_raster,
        output_folder=output_folder,
        num_bins=num_bins,
        b_bins=b_bins,
        permutations=permutations,
        svg_only=True,
        ecobound=generate_boundary
    )

if __name__ == '__main__':
    main()
