import os
import shutil
import pandas as pd
from .geodetector_arcpy import Geodetector
from .slice_arcpy_master import slice_rasters
import arcpy

def slice_from_folder(
    input_folder,
    output_dir,
    slice_types=None,
    number_zones=None,
    message_callback=print
):
    def notify(msg):
        try:
            message_callback(msg)
        except:
            print(msg)

    arcpy.env.workspace = input_folder
    rasters = arcpy.ListRasters("*", "TIF")

    if not rasters:
        raise ValueError(f"No .tif rasters found in {input_folder}")

    full_paths = [os.path.join(input_folder, r) for r in rasters]
    notify(f"✓ Found {len(full_paths)} rasters in {input_folder}. Starting slicing...")

    return slice_rasters(
        raster_list=full_paths,
        slice_types=slice_types,
        number_zones=number_zones,
        output_dir=output_dir,
        message_callback=message_callback
    )

class GeoDetectorAdvanced:
    def __init__(self, x_rasters, y_raster, mode="qmax", alpha=0.05):
        self.geodetector = Geodetector(x_rasters, y_raster, mode=mode, alpha=alpha)

    def run_all(self, output_dir, modules=["factor", "interaction", "risk", "eco"], message_callback=print):
        def notify(msg):
            try:
                message_callback(msg)
            except:
                print(msg)

        if "factor" in modules:
            notify("🔍 Running factor detector...")
            self.geodetector.calculate_q_for_list()
            self.geodetector.save_q_results(os.path.join(output_dir, "q_results.csv"))
            self.geodetector.filter_q_results()
            self.geodetector.save_filtered_q_results(os.path.join(output_dir, "filtered_q_results.csv"))

            if self.geodetector.mode == "qmax":
                notify("🧹 Cleaning up non-optimal sliced rasters in X_sliced...")
                keep_files = set(os.path.basename(row["Raster"]) for _, row in self.geodetector.filtered_q_results.iterrows())
                sliced_dir = os.path.join(output_dir, "X_sliced")
                for fname in os.listdir(sliced_dir):
                    if fname.endswith(".tif") and fname not in keep_files:
                        fpath = os.path.join(sliced_dir, fname)
                        try:
                            #os.remove(fpath)
                            arcpy.management.Delete(fpath)
                            notify(f"🗑 Deleted non-optimal slice: {fname}")
                        except Exception as e:
                            notify(f"⚠ Could not delete {fname}: {e}")

            

        if "interaction" in modules:
            notify("🔁 Running interaction detector...")
            self.geodetector.detection_of_interaction()
            self.geodetector.save_interaction_results(os.path.join(output_dir, "interaction_results.csv"))

        if "risk" in modules:
            notify("⚠ Running risk detector...")
            self.geodetector.risk_detection()
            self.geodetector.save_risk_results(os.path.join(output_dir, "risk_results.csv"))

        if "eco" in modules:
            notify("🌿 Running ecological detector...")
            self.geodetector.ecological_detection()
            self.geodetector.save_ecological_results(os.path.join(output_dir, "eco_results.csv"))

        notify("✅ GeoDetector full analysis complete.")

def run_geodetector(
    y_raster,
    x_aligned_folder,
    output_dir,
    slice_types=["GEOMETRIC_INTERVAL"],
    number_zones=[3, 4, 5],
    mode="qmax",
    alpha=0.05,
    modules=["factor", "interaction", "risk", "eco"],
    message_callback=print
):
    def notify(msg):
        try:
            message_callback(msg)
        except:
            print(msg)

    os.makedirs(output_dir, exist_ok=True)
    sliced_dir = os.path.join(output_dir, "X_sliced")
    os.makedirs(sliced_dir, exist_ok=True)

    notify("🧩 Step 1: Slicing X rasters...")
    x_sliced = slice_from_folder(
        input_folder=x_aligned_folder,
        output_dir=sliced_dir,
        slice_types=slice_types,
        number_zones=number_zones,
        message_callback=notify
    )

    notify("📊 Step 2: Running GeoDetectorAdvanced...")
    gda = GeoDetectorAdvanced(x_sliced, y_raster, mode=mode, alpha=alpha)
    gda.run_all(output_dir, modules=modules, message_callback=notify)
