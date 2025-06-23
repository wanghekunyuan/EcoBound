import arcpy
import os
import re
from arcpy.sa import FocalStatistics, NbrCircle, Raster, ExtractByMask

def trend_analysis(
    input_folder,
    output_folder,
    radius=5000,
    unit="MAP",  # 新增单位参数
    slope_output="Sens_Slope.tif",
    zscore_output="Z_score.tif",
    message_callback=print
):
    def notify(msg):
        #print(msg)
        try:
            message_callback(msg)
        except:
            pass

    notify("🌱 EcoBound module initializing: analyzing ecological trend responses, please wait...")
    notify("🌱 EcoBound 模块启动中：正在分析生态响应趋势，请稍候...")

    arcpy.CheckOutExtension("Spatial")
    arcpy.CheckOutExtension("ImageAnalyst")

    arcpy.env.workspace = input_folder
    arcpy.env.overwriteOutput = True
    arcpy.env.scratchWorkspace = "in_memory"
    arcpy.env.compression = "LERC"
    os.makedirs(output_folder, exist_ok=True)

    focal_neighborhood = NbrCircle(radius, unit.upper()) # 支持 MAP 或 CELL

    notify(f"→ Using radius = {radius} with unit = {unit.upper()}")

    def parse_year_from_filename(rname):
        m = re.search(r'\d{4}', rname)
        return int(m.group()) if m else None

    notify(">>> [1] Reading raster files from input folder...")
    notify(">>> [1] 正在读取输入文件夹中的栅格数据...")

    all_rasters = arcpy.ListRasters("*", "TIF")
    if not all_rasters:
        raise ValueError("未找到任何 .tif 文件。\nNo .tif files found in input folder.")

    raster_info = [(r, parse_year_from_filename(r)) for r in all_rasters]
    raster_info.sort(key=lambda x: x[1] if x[1] is not None else 0)

    notify(f">>> [2] Detected {len(raster_info)} valid yearly rasters. Starting FocalStatistics computation...")
    notify(f">>> [2] 读取到 {len(raster_info)} 个有效年份的栅格。开始进行 FocalStatistics 计算...")

    focal_rasters = []
    for idx, (rname, yval) in enumerate(raster_info, 1):
        ras = Raster(os.path.join(input_folder, rname))
        focal = FocalStatistics(ras, focal_neighborhood, "MEAN", "DATA")
        out_path = os.path.join(output_folder, f"focal_{rname}")
        focal.save(out_path)
        focal_rasters.append(focal)

        notify(f"   → Raster {idx}/{len(raster_info)} processed: {rname} (Year: {yval})")
        notify(f"   → 第 {idx}/{len(raster_info)} 张栅格处理完成：{rname}（年份: {yval}）")

    notify(">>> [3] Creating Mosaic Dataset and adding all focal rasters...")
    notify(">>> [3] 正在创建 Mosaic Dataset 并添加所有焦点栅格...")

    loggdb = os.path.join(output_folder, "log.gdb")
    arcpy.management.CreateFileGDB(output_folder, "log.gdb")
    arcpy.management.CreateMosaicDataset(loggdb, "Mosaic", focal_rasters[0])
    mosaic_path = os.path.join(loggdb, "Mosaic")
    arcpy.management.AddRastersToMosaicDataset(mosaic_path, "Raster Dataset", focal_rasters)

    years = [y for (_, y) in raster_info]
    with arcpy.da.UpdateCursor(mosaic_path, ["ZOrder"]) as cursor:
        for i, row in enumerate(cursor):
            row[0] = years[i]
            cursor.updateRow(row)

    arcpy.md.BuildMultidimensionalInfo(mosaic_path, "Tag", "ZOrder")

    notify(">>> [4] Starting trend analysis using GenerateTrendRaster (this may take a while)...")
    notify(">>> [4] 开始趋势分析 GenerateTrendRaster（这一步可能较慢）...")

    trend = arcpy.ia.GenerateTrendRaster(mosaic_path, "ZOrder", "Dataset", "MANN-KENDALL")

    notify(">>> [5] Extracting and saving trend results: Sens Slope and Z-score...")
    notify(">>> [5] 提取并保存趋势结果：Sens Slope 和 Z-score...")

    slope_ras = ExtractByMask(trend.getRasterBands(1), all_rasters[0])
    slope_ras.save(os.path.join(output_folder, slope_output))

    zscore_ras = ExtractByMask(trend.getRasterBands(5), all_rasters[0])
    zscore_ras.save(os.path.join(output_folder, zscore_output))

    notify(">>> [✓] All processing steps complete. Cleaning up temporary files...")
    notify(">>> [✓] 全部处理完成。准备清理中间文件...")

    try:
        arcpy.management.Delete(loggdb)
    except Exception as e:
        notify(f"【警告，可忽略】⚠ 无法删除 {loggdb}，可能正被锁定。请稍后手动删除。详细信息：{e}")
        notify(f"[Warning: Can be ignored] ⚠ Unable to delete {loggdb}. It may be locked. Please delete it manually later. Details: {e}")

    for ras in focal_rasters:
        try:
            arcpy.management.Delete(ras)
        except:
            pass

    arcpy.CheckInExtension("Spatial")
    arcpy.CheckInExtension("ImageAnalyst")

    notify("完成100%")
    notify("Complete 100%")
