# risk_detector_runner.py
# ------------------------------------------------------------
# Advanced Risk Detector runner (ArcGIS Script Tool entrypoint)
# - Adds layout control for figures: ["large"], ["small"], or both
# - Adds optional GERI synthesis (pure ArcPy implementation in adv_risk)
# ------------------------------------------------------------

import os
import arcpy
from Ecobound import adv_risk  # == batch_advanced_risk_detector(...)

def _parse_layout(text_value):
    """
    Parse layout text into a list of flags from {"large","small"}.
    Examples accepted (case-insensitive):
      "", "large", "small", "large,small", "large; small", "Large Small"
    Default -> ["large"]
    """
    if not text_value:
        return ["large"]
    raw = text_value.replace(";", ",").replace("|", ",").strip().lower()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    valid = {"large", "small"}
    picked = [p for p in parts if p in valid]
    return picked or ["large"]

def main():
    # ---------------- Params from ArcGIS Script Tool ----------------
    # Required (existing)
    x_folder           = arcpy.GetParameterAsText(0)
    y_raster           = arcpy.GetParameterAsText(1)
    output_folder      = arcpy.GetParameterAsText(2)
    num_bins           = int(arcpy.GetParameterAsText(3))
    std_factor         = float(arcpy.GetParameterAsText(4))
    sma_short_bins     = int(arcpy.GetParameterAsText(5))
    sma_long_bins      = int(arcpy.GetParameterAsText(6))
    macd_short_period  = int(arcpy.GetParameterAsText(7))
    macd_long_period   = int(arcpy.GetParameterAsText(8))
    macd_signal_period = int(arcpy.GetParameterAsText(9))
    k_factor           = float(arcpy.GetParameterAsText(10))
    svg_only           = arcpy.GetParameter(11)  # bool

    # New (added)
    layout_text        = arcpy.GetParameterAsText(12)  # e.g., "large,small"
    compute_geri       = arcpy.GetParameter(13)        # bool
    qmax_csv           = arcpy.GetParameterAsText(14)  # path to filtered_q_results.csv
    risk_q_text        = arcpy.GetParameterAsText(15)  # e.g., "0.98"
    sig_filter         = arcpy.GetParameter(16)        # bool
    export_sidecars    = arcpy.GetParameter(17)        # bool

    # ---------------- Parse & defaults ----------------
    layout = _parse_layout(layout_text)
    try:
        risk_norm_quantile = float(risk_q_text) if risk_q_text not in (None, "") else 0.98
    except Exception:
        risk_norm_quantile = 0.98

    # ---------------- Run ----------------
    arcpy.AddMessage("🚀 Running Advanced Risk Detector...")
    arcpy.AddMessage(f"• Layout: {layout}")
    if compute_geri:
        arcpy.AddMessage("• GERI synthesis: ENABLED")
        if not qmax_csv:
            arcpy.AddWarning("⚠ compute_geri=True but qmax_csv is empty; GERI will be skipped in adv_risk.")
        else:
            arcpy.AddMessage(f"• Qmax table: {qmax_csv}")
            arcpy.AddMessage(f"• Risk normalization quantile (p): {risk_norm_quantile}")
            arcpy.AddMessage(f"• GeoDetector significance filter: {bool(sig_filter)}")
    else:
        arcpy.AddMessage("• GERI synthesis: DISABLED")

    adv_risk(
        x_folder=x_folder,
        y_raster=y_raster,
        output_folder=output_folder,
        num_bins=num_bins,
        std_factor=std_factor,
        sma_short_bins=sma_short_bins,
        sma_long_bins=sma_long_bins,
        macd_short_period=macd_short_period,
        macd_long_period=macd_long_period,
        macd_signal_period=macd_signal_period,
        k_factor=k_factor,
        layout=layout,
        svg_only=svg_only,

        # ---- Optional GERI ----
        compute_geri=bool(compute_geri),
        qmax_csv=qmax_csv,
        risk_norm_quantile=risk_norm_quantile,
        sig_filter=bool(sig_filter),
        export_sidecars=bool(export_sidecars),
    )

    arcpy.AddMessage("✅ Advanced Risk Detection Completed.")

if __name__ == "__main__":
    main()
