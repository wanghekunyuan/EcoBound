# ecr_tool_runner.py
import arcpy
from Ecobound import trend_analysis

def script_tool():
    # 读取参数（注意顺序和参数表一致）
    input_folder = arcpy.GetParameterAsText(0)
    output_folder = arcpy.GetParameterAsText(1)
    radius = float(arcpy.GetParameterAsText(2))
    unit = arcpy.GetParameterAsText(3) # 新增参数


    # 输出提示信息
    arcpy.AddMessage("✓ Parameters received, starting trend analysis...")
    arcpy.AddMessage(f"→ Input Folder: {input_folder}")
    arcpy.AddMessage(f"→ Output Folder: {output_folder}")
    arcpy.AddMessage(f"→ Radius: {radius}")

    # 调用主函数
    trend_analysis(
        input_folder=input_folder,
        output_folder=output_folder,
        radius=radius,
        unit=unit,
        message_callback=arcpy.AddMessage  # 👈 传入消息回调
    )


# ArcGIS 会自动调用此脚本，需显式执行主函数
script_tool()
