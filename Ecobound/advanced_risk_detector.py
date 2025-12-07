# advanced_risk_detector.py
# ---------------------------------------------------------------------
# Batch runner for Advanced Risk Detector (Figure 8-7) with optional
# GERI synthesis using pure ArcPy/Spatial Analyst (C2 pixelwise renorm).
#
# Key design (locked with user):
#  - Risk_j = |MACD - Signal|  (already produced per-X in plotting step)
#  - Per-layer robust normalization: Risk*_j = min(Risk_j / p98(Risk_j), 1)
#  - GERI(p) = (Σ_j q_j * Safe(Risk*_j(p))) / (Σ_j Mask(Risk*_j(p), q_j))
#       Safe(R) = Con(IsNull(R), 0, R)
#       Mask(R,q) = Con(IsNull(R), 0, q)
#  - If W(p)=0 (no valid layers at p), output 0 by design.
#
# Inputs:
#  - x_folder: folder of X rasters to analyze (non-recursive)
#  - y_raster: Y raster path
#  - output_folder: where figures and risk rasters are written by fig8_7
#  - qmax_csv: filtered_q_results.csv (Qmax), fields: Base_X, Q_statistic, Significant
#
# Outputs:
#  - Per-X figures (SVG/PNG) from fig8_7 (layout=["large"/"small"/both"])
#  - Optional: GERI.tif, GERI_valid_count.tif, GERI_effective_weight.tif
#              GERI_risk_norm_thresholds.csv, GERI_used_layers.csv
#
# Notes:
#  - Pure ArcPy SA is used for GERI to avoid large-array memory pressure.
#  - p98 is computed via ZonalStatisticsAsTable on a single-zone raster,
#    so it scales to large rasters without loading full arrays to memory.
#  - We DO NOT add {Base_X}_risk.tif as an input parameter; we auto-scan
#    risk rasters generated in this batch under output_folder.
# ---------------------------------------------------------------------

import os
import glob
import csv
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt

import arcpy
from arcpy.sa import (
    Raster, Con, IsNull, SetNull, Plus, Times, Divide, Int, Float,
    ZonalStatisticsAsTable
)

# Import the plotting/analysis function (Figure 8-7)
# This is assumed to produce the per-X risk rasters alongside figures.
from .fig8_7 import advanced_risk_detector


# ------------------------ Utilities ------------------------

def _notify(msg: str) -> None:
    try:
        arcpy.AddMessage(msg)
    except Exception:
        print(msg)


def _warn(msg: str) -> None:
    try:
        arcpy.AddWarning(msg)
    except Exception:
        print("[WARN] " + msg)


def _error(msg: str) -> None:
    try:
        arcpy.AddError(msg)
    except Exception:
        print("[ERROR] " + msg)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _is_tif(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {".tif", ".tiff"}


def _list_x_rasters(x_folder: str) -> List[str]:
    """List *.tif files under x_folder (non-recursive)."""
    items = []
    for name in os.listdir(x_folder):
        p = os.path.join(x_folder, name)
        if os.path.isfile(p) and _is_tif(p):
            items.append(p)
    items.sort()
    return items


def _scan_risk_rasters(output_folder: str) -> Dict[str, str]:
    """
    Find per-X risk rasters produced during plotting.
    Preferred pattern: '<Base_X>_risk.tif'
    Fallback pattern:  '.../<Base_X>/risk_value.tif'
    Returns: {Base_X: path_to_risk_tif}
    """
    mapping: Dict[str, str] = {}

    # Preferred: <Base_X>_risk.tif anywhere under output_folder
    for path in glob.glob(os.path.join(output_folder, "**", "*_risk.tif"), recursive=True):
        base = os.path.basename(path)
        base_x = base[:-9]  # strip '_risk.tif'
        if base_x:
            mapping[base_x] = path

    # Fallback: .../<Base_X>/risk_value.tif
    for path in glob.glob(os.path.join(output_folder, "**", "risk_value.tif"), recursive=True):
        base_x = os.path.basename(os.path.dirname(path))
        mapping.setdefault(base_x, path)

    return mapping


def _load_qmax_table(qmax_csv: str, sig_filter: bool = True) -> Dict[str, Tuple[float, bool]]:
    """
    Load filtered_q_results.csv (Qmax) to a dict:
      { Base_X: (q_value, significant_bool) }
    Required columns:
      - Base_X
      - Q_statistic
      - Significant (optional; if absent, treated as True)
    """
    if not os.path.isfile(qmax_csv):
        raise ValueError(
            "未找到 Qmax CSV：{0}\n"
            "Qmax CSV not found: {0}".format(qmax_csv)
        )

    with open(qmax_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in reader.fieldnames] if reader.fieldnames else []
        colmap = {h: h for h in headers}

        required = {"Base_X", "Q_statistic"}
        if not required.issubset(set(headers)):
            raise ValueError(
                "Qmax 缺少必要字段 {0}\n"
                "Qmax missing required columns {0}".format(required)
            )

        has_sig = "Significant" in headers
        result: Dict[str, Tuple[float, bool]] = {}

        for row in reader:
            base_x = row["Base_X"].strip()
            try:
                qv = float(row["Q_statistic"])
            except Exception:
                continue
            sig = True
            if has_sig:
                sig_str = str(row["Significant"]).strip().lower()
                sig = sig_str in ("true", "1", "yes", "y")

            if sig_filter and not sig:
                continue
            # Keep first occurrence
            if base_x not in result:
                result[base_x] = (qv, sig)

    return result


# ---------------------- Percentile p98 ----------------------

def _p98_via_zonal_stats(value_raster: Raster, scratch_name: str) -> Optional[float]:
    """
    Compute the global 98th percentile of a raster using ZonalStatisticsAsTable.
    We create a single-zone raster (constant 1) matching the template.
    """
    try:
        arcpy.CheckOutExtension("Spatial")
    except Exception:
        _warn("⚠️ 无法检出 Spatial Analyst 许可；无法计算 p98。\n"
              "⚠️ Could not check out Spatial Analyst; cannot compute p98.")
        return None

    try:
        # Single-zone raster aligned to value_raster
        zone = Con(IsNull(value_raster), 1, 1)  # constant 1 with same grid
        out_tbl = os.path.join("in_memory", f"ztbl_p98_{scratch_name}")

        # Zonal statistics with percentile
        # ignore_nodata="DATA" ensures NoData cells are ignored
        ZonalStatisticsAsTable(
            in_zone_data=zone,
            zone_field="Value",          # zone raster must use 'Value'
            in_value_raster=value_raster,
            out_table=out_tbl,
            ignore_nodata="DATA",
            statistics_type="PERCENTILE",
            percentile_values=98
        )

        # The percentile field is typically named like 'PCT_98' (defensive parse)
        fields = [f.name for f in arcpy.ListFields(out_tbl)]
        pct_fields = [fn for fn in fields if "PCT" in fn.upper() or "PERCENT" in fn.upper()]
        target_field = pct_fields[0] if pct_fields else None

        p98_val = None
        if target_field:
            with arcpy.da.SearchCursor(out_tbl, [target_field]) as cur:
                for (v,) in cur:
                    p98_val = float(v) if v is not None else None
                    break

        # Clean up
        try:
            arcpy.management.Delete(out_tbl)
        except Exception:
            pass

        return p98_val

    except Exception as e:
        _warn(f"⚠️ p98 计算失败：{e}\n"
              f"⚠️ Failed to compute p98: {e}")
        return None
    finally:
        try:
            arcpy.CheckInExtension("Spatial")
        except Exception:
            pass


# ---------------------- GERI (pure ArcPy) ----------------------

def _compute_geri_arcsa(
    output_folder: str,
    qmax_csv: str,
    risk_norm_quantile: float = 0.98,   # locked at 0.98 (p98)
    sig_filter: bool = True,
    export_sidecars: bool = True,
    out_name_prefix: str = "GERI"
) -> Optional[str]:
    """
    Compute GERI via ArcPy SA using C2 pixelwise renormalization.
    Returns GERI path if successful; else None.
    """
    try:
        arcpy.CheckOutExtension("Spatial")
    except Exception:
        _error("❌ 未能检出 Spatial Analyst 许可，无法计算 GERI。\n"
               "❌ Could not check out Spatial Analyst; cannot compute GERI.")
        return None

    try:
        risk_map = _scan_risk_rasters(output_folder)
        if not risk_map:
            _warn("⚠️ 未找到任何风险栅格（*_risk.tif 或 */risk_value.tif）。\n"
                  "⚠️ No risk rasters found (*_risk.tif or */risk_value.tif).")
            return None

        qdict = _load_qmax_table(qmax_csv, sig_filter=sig_filter)

        # intersect keys
        common = sorted(set(risk_map.keys()) & set(qdict.keys()))
        skipped_q = sorted(set(risk_map.keys()) - set(qdict.keys()))
        if not common:
            _warn("⚠️ 未发现同时具备风险栅格与 Qmax 权重的变量，GERI 无法计算。\n"
                  "⚠️ No overlap between risk rasters and Qmax weights; cannot compute GERI.")
            return None
        if skipped_q:
            _notify(f"ℹ️ 以下风险图层在 Qmax 中找不到权重，已跳过（{len(skipped_q)}）：{skipped_q}\n"
                    f"ℹ️ Risk layers without q in Qmax (skipped, {len(skipped_q)}): {skipped_q}")

        # Accumulators
        S = None   # numerator sum: Σ q * Safe(Rnorm)
        W = None   # denominator sum: Σ Mask(Rnorm, q)
        C = None   # valid_count: Σ Con(IsNull(Rnorm), 0, 1)

        thresholds_records: List[Tuple[str, float]] = []
        used_records: List[Tuple[str, str, float, bool]] = []

        eps = 1e-12

        for base_x in common:
            r_path = risk_map[base_x]
            qv, sig = qdict[base_x]
            qv = float(qv)

            _notify(f"• 处理风险图层 / Processing: {base_x}")

            R = Raster(r_path)

            # --- compute p98 robust threshold (global over valid cells) ---
            p98 = _p98_via_zonal_stats(R, scratch_name=base_x) if abs(risk_norm_quantile - 0.98) < 1e-9 else None
            if p98 is None or p98 <= 0:
                # fallback: use MAX as conservative threshold
                try:
                    p98 = float(arcpy.management.GetRasterProperties(R, "MAXIMUM").getOutput(0))
                except Exception:
                    p98 = 1.0
            thresholds_records.append((base_x, p98))

            # --- normalized risk: Risk* = min(R / (p98+eps), 1), preserving NoData ---
            R_div = R / (p98 + eps)
            R_norm_raw = Con(R_div > 1, 1, R_div)    # clip upper bound at 1
            R_norm = SetNull(IsNull(R), R_norm_raw)  # keep NoData where R is NoData


            # --- Safe / Mask ---
            Safe = Con(IsNull(R_norm), 0, R_norm)   # contributes only where valid
            Mask = Con(IsNull(R_norm), 0, qv)       # contributes q where valid else 0
            C_i  = Con(IsNull(R_norm), 0, 1)        # count valid layers

            # --- accumulate ---
            term_S = Times(Safe, qv)                # q * Safe(R_norm)
            S = term_S if S is None else Plus(S, term_S)
            W = Mask   if W is None else Plus(W, Mask)
            C = C_i    if C is None else Plus(C, C_i)

            used_records.append((base_x, r_path, qv, sig))

        if S is None or W is None:
            _warn("⚠️ 没有可用的有效图层用于 GERI 计算。\n"
                  "⚠️ No valid layers to compute GERI.")
            return None

        # --- GERI = Con(W>0, S/W, 0) ---
        GERI = Con(W > 0, Divide(S, W), 0)

        # --- Write outputs ---
        _ensure_dir(output_folder)
        out_geri = os.path.join(output_folder, f"{out_name_prefix}.tif")
        out_valid = os.path.join(output_folder, f"{out_name_prefix}_valid_count.tif")
        out_wsum = os.path.join(output_folder, f"{out_name_prefix}_effective_weight.tif")

        # Save rasters
        GERI.save(out_geri)
        Int(C).save(out_valid)
        Float(W).save(out_wsum)

        # Sidecar CSVs
        if export_sidecars:
            thr_csv = os.path.join(output_folder, f"{out_name_prefix}_risk_norm_thresholds.csv")
            with open(thr_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Base_X", "Risk_p98"])
                for bx, tval in thresholds_records:
                    w.writerow([bx, "{:.10g}".format(tval)])

            used_csv = os.path.join(output_folder, f"{out_name_prefix}_used_layers.csv")
            with open(used_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Base_X", "risk_path", "q", "Significant"])
                for rec in used_records:
                    bx, rpath, qv, sig = rec
                    w.writerow([bx, rpath, "{:.10g}".format(qv), str(bool(sig))])

        _notify(f"✅ 已输出 / Written: {out_geri}")
        _notify(f"✅ 已输出 / Written: {out_valid}")
        _notify(f"✅ 已输出 / Written: {out_wsum}")
        return out_geri

    except Exception as e:
        _error(f"❌ 计算 GERI 失败：{e}\n"
               f"❌ Failed to compute GERI: {e}")
        return None
    finally:
        try:
            arcpy.CheckInExtension("Spatial")
        except Exception:
            pass


# ---------------------- Batch Runner ----------------------

def batch_advanced_risk_detector(
    x_folder: str,
    y_raster: str,
    output_folder: str,
    num_bins: int = 100,
    std_factor: float = 1.5,
    sma_short_bins: int = 10,
    sma_long_bins: int = 20,
    macd_short_period: int = 12,
    macd_long_period: int = 26,
    macd_signal_period: int = 9,
    k_factor: float = 1.0,
    layout: List[str] = None,
    svg_only: bool = False,
    # --- Optional GERI ---
    compute_geri: bool = False,
    qmax_csv: Optional[str] = None,
    risk_norm_quantile: float = 0.98,   # locked to 0.98 (p98)
    sig_filter: bool = True,
    export_sidecars: bool = True,
):
    """
    Batch runner with optional GERI synthesis (pure ArcPy SA).
    """
    if layout is None:
        layout = ["large"]

    _ensure_dir(output_folder)

    x_paths = _list_x_rasters(x_folder)
    if not x_paths:
        _warn("⚠️ 在 x_folder 中未找到任何 .tif 输入。\n"
              "⚠️ No .tif files found in x_folder.")
        return

    _notify(f"🟦 批处理开始：{len(x_paths)} 个 X\n"
            f"🟦 Batch start: {len(x_paths)} X rasters.")

    for x_path in x_paths:
        base_name = os.path.splitext(os.path.basename(x_path))[0]

        svg_path = os.path.join(output_folder, f"{base_name}.svg")
        png_path = os.path.join(output_folder, f"{base_name}.png")

        risk_value_tif = os.path.join(output_folder, f"{base_name}_risk.tif")         # 连续风险（我们后续用它做 GERI）
        risk_level_tif = os.path.join(output_folder, f"{base_name}_risk_level.tif")   # 风险等级（分级栅格）
      
        modes = [m for m in ("large", "small") if m in layout]
        if not modes:
            modes = ["large"]

        for _m in modes:
            _notify(f"→ {base_name} [{_m}] ...")

            risk_df, fig = advanced_risk_detector(
                x_path,
                y_raster,
                num_bins=num_bins,
                std_factor=std_factor,
                sma_short_bins=sma_short_bins,
                sma_long_bins=sma_long_bins,
                macd_short_period=macd_short_period,
                macd_long_period=macd_long_period,
                macd_signal_period=macd_signal_period,
                k_factor=k_factor,
                layout=[_m],
                output_risk_level_tif_path=risk_level_tif,   # ★ 新增：分级风险输出
                output_risk_value_tif_path=risk_value_tif    # ★ 新增：连续风险输出（供 GERI 使用）
            )





            # Save figs with suffix if needed
            if len(modes) == 1:
                fig.savefig(svg_path, format="svg")
                if not svg_only:
                    fig.savefig(png_path, dpi=300)
            else:
                root_svg, ext_svg = os.path.splitext(svg_path)
                root_png, ext_png = os.path.splitext(png_path)
                fig.savefig(f"{root_svg}_{_m}{ext_svg}", format="svg")
                if not svg_only:
                    fig.savefig(f"{root_png}_{_m}{ext_png}", dpi=300)

            plt.close(fig)

        _notify(f"✅ Done: {base_name}")

    # Optional GERI synthesis (pure ArcPy)
    if compute_geri:
        if not qmax_csv or not os.path.isfile(qmax_csv):
            _warn("⚠️ compute_geri=True 但未提供有效的 qmax_csv（filtered_q_results.csv）。\n"
                  "⚠️ compute_geri=True but qmax_csv (filtered_q_results.csv) is missing.")
            return

        _notify("🟩 开始计算 GERI ...\n"
                "🟩 Computing GERI ...")
        _compute_geri_arcsa(
            output_folder=output_folder,
            qmax_csv=qmax_csv,
            risk_norm_quantile=risk_norm_quantile,  # currently locked to 0.98
            sig_filter=sig_filter,
            export_sidecars=export_sidecars,
            out_name_prefix="GERI",
        )
        _notify("🟩 GERI 计算完成。\n"
                "🟩 GERI done.")
