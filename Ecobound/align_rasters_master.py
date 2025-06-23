import arcpy
from arcpy.sa import Raster
import os

def align_rasters(template, raster_list, output_path, continuous, message_callback=print):
    """
    Align rasters by spatial reference, extent, and cell size to a template raster.

    Parameters:
        template (str): Path to the template raster used as reference.
        raster_list (list of str): List of input raster paths to be aligned.
        output_path (str): Folder to save aligned rasters.
        continuous (list of bool): List indicating whether each raster is continuous (True) or categorical (False).
        message_callback (function): Function for handling messages (e.g., print or arcpy.AddMessage).

    Returns:
        list: List of aligned raster output paths.
    """
    def notify(msg):
        try:
            message_callback(msg)
        except:
            print(msg)

    # Basic validation
    if len(continuous) != len(raster_list):
        raise ValueError("The length of 'continuous' must match the number of input rasters.")

    arcpy.env.overwriteOutput = True
    arcpy.CheckOutExtension("Spatial")

    # Create output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        notify(f"✓ Output folder created: {output_path}")

    # Load template raster and set cell size environment
    tem = Raster(template)
    arcpy.env.cellSize = tem

    # Use template + all rasters to compute full processing extent
    for raster_path in raster_list:
        tem = tem + Raster(raster_path)
    tem = tem - tem  # generates a 0-valued raster with full extent

    out_raster_paths = []

    for raster_path, is_continuous in zip(raster_list, continuous):
        name = os.path.basename(raster_path)
        output_raster = os.path.join(output_path, name)

        # Set resampling method
        arcpy.env.resamplingMethod = "CUBIC" if is_continuous else "NEAREST"
        notify(f"→ Aligning {name} as {'continuous' if is_continuous else 'categorical'} raster...")

        # Align raster by adding to template extent raster
        aligned = tem + Raster(raster_path)
        aligned.save(output_raster)
        out_raster_paths.append(output_raster)

        notify(f"✓ Saved: {output_raster}")

    notify("✓ All rasters aligned successfully.")
    return out_raster_paths