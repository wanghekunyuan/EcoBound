# align_rasters_master.py  (v3.20)
import arcpy
from arcpy.sa import Raster
import os

def align_rasters(
    template,
    raster_list,
    output_path,
    continuous,
    message_callback=print,
    align_policy="template_only"  # default: recommended
):
    """
    Align rasters by spatial reference, extent, cell size, and pixel grid.

    Parameters
    ----------
    template : str
        Path to the template raster used as reference.
    raster_list : list[str]
        List of input raster paths to be aligned.
    output_path : str
        Folder to save aligned rasters.
    continuous : list[bool]
        For each raster, True if continuous (use CUBIC), False if categorical (use NEAREST).
    message_callback : callable
        Function to emit messages (e.g., print or arcpy.AddMessage).
    align_policy : {"template_only","global_intersection"}
        - "template_only" (default; recommended): use template to unify projection/cell size/snap,
          and mask by template's valid area only. NoData from other rasters will NOT propagate.
        - "global_intersection" (legacy): build a zero raster from the global intersection of
          all inputs. Any NoData in any raster will propagate to all outputs.

    Returns
    -------
    list[str]
        Paths of aligned rasters saved to output_path.
    """

    def notify(msg: str):
        # No fallback print to avoid duplicate outputs in ArcGIS Script Mode
        message_callback(msg)

    # --- Basic checks ---
    if len(raster_list) != len(continuous):
        raise ValueError("Length mismatch: `raster_list` and `continuous` must have the same length.")

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # --- Backup & set environments ---
    _env = {
        "overwriteOutput": arcpy.env.overwriteOutput,
        "snapRaster": arcpy.env.snapRaster,
        "cellSize": arcpy.env.cellSize,
        "extent": arcpy.env.extent,
        "outputCoordinateSystem": arcpy.env.outputCoordinateSystem,
        "resamplingMethod": getattr(arcpy.env, "resamplingMethod", None),
    }
    arcpy.env.overwriteOutput = True

    tmpl = Raster(template)
    arcpy.env.snapRaster = tmpl
    arcpy.env.cellSize = tmpl.meanCellWidth  # assuming square cells; ArcGIS handles X/Y accordingly
    arcpy.env.extent = tmpl.extent
    arcpy.env.outputCoordinateSystem = tmpl.spatialReference

    try:
        # --- Build zero template ("tem") according to policy ---
        if align_policy not in ("template_only", "global_intersection"):
            raise ValueError("align_policy must be either 'template_only' or 'global_intersection'.")

        notify(f"Align policy: {align_policy}")
        tem = Raster(template)

        if align_policy == "global_intersection":
            notify("Building 'global intersection' zero raster (legacy behavior)...")
            for rp in raster_list:
                tem = tem + Raster(rp)
            tem = tem - tem  # valid where ALL inputs are valid; elsewhere NoData
        else:
            notify("Building 'template-only' zero raster (recommended)...")
            tem = tem - tem  # valid where template is valid; independent of other rasters' NoData

        out_raster_paths = []

        # --- Align each raster ---
        for raster_path, is_continuous in zip(raster_list, continuous):
            name = os.path.basename(raster_path)
            output_raster = os.path.join(output_path, name)

            # Choose resampling method (effective when projection/resample is needed)
            arcpy.env.resamplingMethod = "CUBIC" if is_continuous else "NEAREST"
            notify(f"→ Aligning {name} as {'continuous' if is_continuous else 'categorical'} raster...")

            # Raster algebra triggers projection/resampling/grid snap under current env:
            # - Inside valid area: tem=0 → tem + X == X
            # - Outside valid area: tem=NoData → result is NoData (controlled by policy)
            aligned = tem + Raster(raster_path)
            aligned.save(output_raster)
            out_raster_paths.append(output_raster)

            notify(f"✓ Saved: {output_raster}")

        notify("✓ All rasters aligned successfully.")
        return out_raster_paths

    finally:
        # --- Restore environments ---
        arcpy.env.overwriteOutput = _env["overwriteOutput"]
        arcpy.env.snapRaster = _env["snapRaster"]
        arcpy.env.cellSize = _env["cellSize"]
        arcpy.env.extent = _env["extent"]
        arcpy.env.outputCoordinateSystem = _env["outputCoordinateSystem"]
        if _env["resamplingMethod"] is not None:
            arcpy.env.resamplingMethod = _env["resamplingMethod"]
