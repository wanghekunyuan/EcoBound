# Example: Advanced Risk Detector + optional GERI (pure ArcPy backend)
# -----------------------------------------------------------------------------
# This script runs the advanced risk detector for all X rasters and (optionally)
# synthesizes a Global Ecological Risk Index (GERI) using C2 per-pixel renormalization.
# Figures: exported as SVG (and optionally PNG) in the output folder.
# GERI:    exported as rasters (GERI.tif, GERI_valid_count.tif, GERI_effective_weight.tif).
# -----------------------------------------------------------------------------

from Ecobound import adv_risk   # adv_risk == batch_advanced_risk_detector(...)
import arcpy
import os

# 0) ArcGIS/ArcPy environment --------------------------------------------------
arcpy.env.overwriteOutput = True
# Spatial Analyst is required for GERI; if you're in ArcGIS Pro Script Tool, this is auto-managed.
# Otherwise you can uncomment the next line:
# arcpy.CheckOutExtension("Spatial")

# NOTE: Because you've already done raster alignment, you typically don't need to set
# env.snapRaster / env.cellSize / env.extent here. If desired, you may still set them.

# 1) Paths ---------------------------------------------------------------------
x_folder      = r".\output\X_alignment"  # Folder containing aligned X rasters (*.tif), non-recursive
y_raster      = r".\output\Barren_testout\Sens_Slope.tif"  # Aligned Y raster
output_folder = r".\output\adv_risk"     # Destination for figures and per-X risk rasters
qmax_csv      = r".\output\geodetector\filtered_q_results.csv"
# ^ Qmax table produced by GeoDetector (filtered_q_results.csv), must contain:
#   - Base_X       : standardized X name (used to match risk rasters)
#   - Q_statistic  : q weight for each X
#   - Significant  : (optional) True/False; used when sig_filter=True

# 2) Run advanced risk detector (with all parameters explicitly set) ----------
adv_risk(
    x_folder=x_folder,                 # Folder of input X rasters (*.tif), non-recursive, already aligned
    y_raster=y_raster,                 # Target Y raster (aligned with X)
    output_folder=output_folder,       # Output directory for figures and risk rasters
    num_bins=100,                      # Number of equal-width bins for X when building response curves
    std_factor=1.5,                    # σ multiplier for Bollinger band (controls band width/risk sensitivity)
    sma_short_bins=10,                 # Short simple moving average window (in bins)
    sma_long_bins=20,                  # Long simple moving average window (in bins)
    macd_short_period=12,              # MACD short EMA period (in bins)
    macd_long_period=26,               # MACD long  EMA period (in bins)
    macd_signal_period=9,              # MACD signal EMA period (in bins)
    k_factor=1.0,                      # Risk-class threshold factor (kept for compatibility in plotting)
    layout=["large", "small"],         # Figure layouts to export; choose one or both: "large", "small"
    svg_only=True,                     # If True, export SVG only (no PNG) for each figure

    # --- Optional GERI synthesis (C2 per-pixel renormalization) ---------------
    compute_geri=True,                 # If True, compute GERI.tif from this batch's per-X risk rasters
    qmax_csv=qmax_csv,                 # Path to GeoDetector output "filtered_q_results.csv" (Qmax).
                                       # If you don’t have it, supply a CSV with columns: Base_X (must match <Base_X>_risk.tif),
                                       # Q_statistic (float), and optional Significant (True/False). Required only if Compute GERI = True.
    risk_norm_quantile=0.98,           # Per-layer robust normalization threshold (p98 of Risk_j)
    sig_filter=True,                   # If True, include only rows with Significant=True from Qmax
    export_sidecars=True               # If True, also export thresholds and used-layers CSVs and diagnostics
)

print("✔ Finished. Outputs are in:", os.path.abspath(output_folder))
