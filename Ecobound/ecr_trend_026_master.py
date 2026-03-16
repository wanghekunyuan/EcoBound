import arcpy
import os
import re
from arcpy.sa import FocalStatistics, NbrCircle, Raster, ExtractByMask

def trend_analysis(
    input_folder,
    output_folder,
    radius=5000,
    unit="MAP",  # 支持 MAP 或 CELL
    slope_output="Sens_Slope.tif",
    zscore_output="Z_score.tif",
    message_callback=print
):
    def notify(msg):
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

    focal_neighborhood = NbrCircle(radius, unit.upper())
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

    focal_paths = []
    for idx, (rname, yval) in enumerate(raster_info, 1):
        ras = Raster(os.path.join(input_folder, rname))
        focal = FocalStatistics(ras, focal_neighborhood, "MEAN", "DATA")
        out_path = os.path.join(output_folder, f"focal_{rname}")
        focal.save(out_path)
        focal_paths.append(out_path)
        notify(f"   → Raster {idx}/{len(raster_info)} processed: {rname} (Year: {yval})")
        notify(f"   → 第 {idx}/{len(raster_info)} 张栅格处理完成：{rname}（年份: {yval}）")

    notify(">>> [3] Building multidimensional input (try Mosaic Dataset; fallback to multiband+NetCDF)...")
    notify(">>> [3] 构建多维输入（优先 Mosaic Dataset；否则使用 多波段+NetCDF 方案）...")

    years = [y for (_, y) in raster_info]
    focal_names = [os.path.basename(p) for p in focal_paths]
    name_to_year = {name: y for name, y in zip(focal_names, years)}

    md_input = None
    md_kind = None

    # ✅ 维度名固定为 ZOrder，维度值为真实年份（2000..2024）
    md_dimension = "ZOrder"
    md_variable = "RSEI"

    loggdb = os.path.join(output_folder, "log.gdb")
    md_gdb = os.path.join(output_folder, "md.gdb")
    out_multiband = os.path.join(md_gdb, "RSEI_multiband")
    out_netcdf = os.path.join(output_folder, "RSEI_multiband.nc")

    def _safe_delete(path_):
        try:
            if arcpy.Exists(path_):
                arcpy.management.Delete(path_)
            elif os.path.exists(path_):
                os.remove(path_)
        except:
            pass

    def _try_build_mosaic_md():
        # Create Mosaic Dataset 需要 Standard/Advanced；Basic 会在这里抛错，然后走 fallback
        _safe_delete(loggdb)
        arcpy.management.CreateFileGDB(output_folder, "log.gdb")

        sr = arcpy.Describe(focal_paths[0]).spatialReference
        arcpy.management.CreateMosaicDataset(
            in_workspace=loggdb,
            in_mosaicdataset_name="Mosaic",
            coordinate_system=sr,
            num_bands=1
        )
        mosaic_path = os.path.join(loggdb, "Mosaic")

        arcpy.management.AddRastersToMosaicDataset(
            in_mosaic_dataset=mosaic_path,
            raster_type="Raster Dataset",
            input_path=output_folder,
            update_cellsize_ranges="NO_CELL_SIZES",
            update_boundary="NO_BOUNDARY",
            update_overviews="NO_OVERVIEWS",
            filter="focal_*.tif",
            sub_folder="NO_SUBFOLDERS",
            build_pyramids="NO_PYRAMIDS",
            calculate_statistics="NO_STATISTICS"
        )

        existing_fields = [f.name for f in arcpy.ListFields(mosaic_path)]
        if md_dimension not in existing_fields:
            arcpy.management.AddField(mosaic_path, md_dimension, "LONG")
        if "Tag" not in existing_fields:
            arcpy.management.AddField(mosaic_path, "Tag", "TEXT", field_length=32)

        with arcpy.da.UpdateCursor(mosaic_path, ["Name", md_dimension, "Tag"]) as cur:
            for name, zval, tag in cur:
                base = os.path.basename(str(name))
                y = None
                if base in name_to_year:
                    y = name_to_year[base]
                elif (base + ".tif") in name_to_year:
                    y = name_to_year[base + ".tif"]
                else:
                    y = parse_year_from_filename(base)

                if y is None:
                    y = zval if zval is not None else 0

                cur.updateRow((name, int(y), md_variable))

        arcpy.md.BuildMultidimensionalInfo(
            in_mosaic_dataset=mosaic_path,
            variable_field="Tag",
            dimension_fields=md_dimension
        )
        return mosaic_path

    def _build_netcdf_md():
        if not arcpy.Exists(md_gdb):
            arcpy.management.CreateFileGDB(output_folder, "md.gdb")

        _safe_delete(out_multiband)
        arcpy.management.CompositeBands(focal_paths, out_multiband)

        _safe_delete(out_netcdf)
        arcpy.md.RasterToNetCDF(
            in_raster=out_multiband,
            out_netCDF_file=out_netcdf,
            variable=md_variable,
            variable_units="",
            x_dimension="x",
            y_dimension="y",
            band_dimension=md_dimension,
            fields_to_dimensions="#",
            compression_level=0
        )

        # 尝试把维度坐标写成真实年份（缺 netCDF4 会给 warning，但常规年度等间隔仍能保证数值一致）
        try:
            from netCDF4 import Dataset
            import numpy as np
            with Dataset(out_netcdf, "r+") as ds:
                if md_dimension in ds.variables:
                    v = ds.variables[md_dimension]
                else:
                    v = ds.createVariable(md_dimension, "i4", (md_dimension,))
                year_vals = [int(y) if y is not None else 0 for y in years]
                v[:] = np.array(year_vals, dtype=v.dtype)
                try:
                    v.long_name = "year"
                    v.units = "year"
                except Exception:
                    pass
        except Exception as e:
            notify(
                f"   ⚠ Unable to write real-year values into NetCDF dimension '{md_dimension}'. "
                f"Trend results will still be correct for equally spaced annual data, "
                f"but slope units may be 'per index'. Details: {e}"
            )

        return out_netcdf

    try:
        notify("   → Trying Mosaic Dataset workflow...")
        md_input = _try_build_mosaic_md()
        md_kind = "mosaic"
        notify("   ✓ Mosaic Dataset workflow succeeded.")
    except Exception as e:
        notify(f"   ⚠ Mosaic Dataset workflow unavailable. Falling back to multiband+NetCDF. Details: {e}")
        md_input = _build_netcdf_md()
        md_kind = "netcdf"
        notify("   ✓ multiband+NetCDF workflow ready.")

    notify(">>> [4] Starting trend analysis using GenerateTrendRaster (this may take a while)...")
    notify(">>> [4] 开始趋势分析 GenerateTrendRaster（这一步可能较慢）...")

    trend = arcpy.ia.GenerateTrendRaster(
        in_multidimensional_raster=md_input,
        dimension=md_dimension,
        variables=md_variable,
        line_type="MANN-KENDALL",
        frequency=None,
        ignore_nodata="DATA",
        cycle_length=None,
        cycle_unit="",
        rmse=None,
        r2=None,
        slope_p_value=None,
        seasonal_period=""
    )

    notify(">>> [5] Extracting and saving trend results: Sens Slope and Z-score...")
    notify(">>> [5] 提取并保存趋势结果：Sens Slope 和 Z-score...")

    mask_path = os.path.join(input_folder, raster_info[0][0])
    mask_ras = Raster(mask_path)

    slope_ras = ExtractByMask(trend.getRasterBands(1), mask_ras)
    slope_ras.save(os.path.join(output_folder, slope_output))

    zscore_ras = ExtractByMask(trend.getRasterBands(5), mask_ras)
    zscore_ras.save(os.path.join(output_folder, zscore_output))

    notify(">>> [✓] All processing steps complete. Cleaning up temporary files...")
    notify(">>> [✓] 全部处理完成。准备清理中间文件...")

    try:
        if md_kind == "mosaic":
            _safe_delete(loggdb)
    except Exception as e:
        notify(f"【警告，可忽略】⚠ 无法清理中间文件，可能正被锁定。请稍后手动删除。详细信息：{e}")
        notify(f"[Warning: Can be ignored] ⚠ Unable to clean intermediate files. They may be locked. Please delete them manually later. Details: {e}")

    for p in focal_paths:
        _safe_delete(p)

    arcpy.CheckInExtension("Spatial")
    arcpy.CheckInExtension("ImageAnalyst")

    notify("完成100%")
    notify("Complete 100%")