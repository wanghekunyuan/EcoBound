# EcoBound v 3.16: A Complete Guide for GeoDetector, Ecological Risk Detection, and Natural Geographical Boundary Extraction (ArcGIS Pro Compatible)

This repository provides a complete implementation and usage guide for **GeoDetector-based ecological analysis**, powered by the **EcoBound toolkit**, a modular Python framework for ecological boundary extraction, threshold detection, and advanced risk mapping based on ArcGIS Pro.

> 🗞 This README also serves as Supplementary Material 2 (SM2) of our manuscript.  
>
> ​        The mathematical foundations of EcoBound sees SM1 of our manuscript.



---

## 📦 Installation

EcoBound relies on **Python 3.7+**, and must be executed within an ArcGIS Pro environment (**version 2.7 or later with Spatial Analyst Extensions**). Please ensure your system is properly configured to run ArcPy-based scripts.

#### Step 1: Clone Python Environment

Follow the official Esri guide to clone the default Python environment:

🔗 [Clone an environment—ArcGIS Pro | Documentation](https://pro.arcgis.com/en/pro-app/latest/arcpy/get-started/clone-an-environment.htm)

This ensures that your custom packages will not affect ArcGIS Pro’s default environment and that updates will not overwrite your setup.

#### Step 2: Install Required Packages

The following packages are required by EcoBound:

| Package                 | Description                            | Availability             |
| ----------------------- | -------------------------------------- | ------------------------ |
| `numpy`                 | Numerical computing                    | ✅ Included in ArcGIS Pro |
| `matplotlib`            | Plotting and visualization             | ✅ Included in ArcGIS Pro |
| `os`, `glob`, `logging` | Standard Python libraries              | ✅ Built-in               |
| `rasterio`              | Geospatial raster I/O and manipulation | ❌ **Not included**       |

All packages except `rasterio` are included in the default ArcGIS Pro Python environment. You may verify or manage them using the **ArcGIS Package Manager**.

For details see the official help website of Esri:

🔗[Add or remove a package—ArcGIS Pro | Documentation](https://pro.arcgis.com/en/pro-app/latest/arcpy/get-started/add-a-package.htm)

---

### 🛠 Installing `rasterio` in ArcGIS Pro Environment (Required)

`rasterio` is not included in ArcGIS Pro by default and must be manually installed. Below is a tested procedure for installing `rasterio` in a cloned ArcGIS Pro environment (e.g., `arcgispro-py3-clone`) running Python 3.9.

#### ✅ Step-by-Step Guide

1. **Open Command Prompt and set temporary directories** *(required for path safety)*:

> 📌 Open **Command Prompt (cmd.exe)** as administrator before running the following commands.

```bash
set TEMP=C:\Temp
set TMP=C:\Temp
```

2. **Activate the cloned environment**:

> 🔧 The path below is the default ArcGIS Pro installation. If you installed to a custom location, please modify it accordingly.

```bash
cd "C:\Program Files\ArcGIS\Pro\bin\Python\Scripts"
activate.bat
conda activate arcgispro-py3-clone  # Replace with your environment name defined in Step 1
```

> ✅ This method has been fully tested and confirmed to work on Windows 10 with ArcGIS Pro 3.2 using the default cloned environment (`arcgispro-py3-clone`).

this indicates the environment is activated and ready for package installation.

3. **Verify environment**:

```bash
conda info --envs
```

Make sure `arcgispro-py3-clone` is activated. If successful, you will see:

<img src="./images\Verify environment.png" alt="Verify environment" style="zoom:50%;" />

4. **Download the `.whl` file** for rasterio (from [https://pypi.org/project/rasterio](https://pypi.org/project/rasterio)), matching your Python version.

Example for Python 3.9:

```
rasterio-1.3.11-cp39-cp39-win_amd64.whl
```

5. **Install using pip**:

```bash
> 💡 Run this command inside **Command Prompt (cmd.exe)** after activating your environment.

python -m pip install "C:\Path\To\rasterio-1.3.11-cp39-cp39-win_amd64.whl"
```

> 📎 Replace `"C:\Path\To\..."` with the actual path where you saved the `.whl` file.

6. **Check installation**:

```bash
python -c "import rasterio; print(rasterio.__version__)"
```

Expected output:

```
1.3.11
```

7. **Test functionality** *(optional)*:

```bash
python -c "import rasterio; print(rasterio.open)"
```

---

### Step 3: Open and Run EcoBound

You can use EcoBound in two different ways depending on your workflow and technical preference:

---

#### ▶️ Script Mode

This is the recommended mode for users familiar with Python scripting or command-line operations.

##### 🧪 How to open the EcoBound environment in IDLE

1. After cloning the environment in **Step 1**, locate the `idle.bat` file inside your conda environment.

> Default path (modify according to your user name and environment name):

```
C:\Users\your_username\AppData\Local\ESRI\conda\envs\arcgispro-py3-clone\Lib\idlelib\idle.bat
```

2. Double-click `idle.bat` to launch **Python IDLE** in the correct ArcGIS Pro environment.

3. Create a new script file (e.g., `open_ecobound.py`) in the same folder where the `Ecobound/` package is located.

​	For example, if your EcoBound code is stored at:

```
G:\SJY LEN\EcoBound Python Package\Ecobound\
```

​	Then your script should be placed at:

```
G:\SJY LEN\EcoBound Python Package\open_ecobound.py
```

​	This ensures that the line `import Ecobound` works correctly.

4. In your script, import EcoBound:

```python
import Ecobound
```

5. Press `F5` or select **Run > Run Module** to execute.

   If  successful, you will see:

   <img src=".\images\onen ecobound script mode.png" alt="onen ecobound script mode" style="zoom:50%;" />

> 💡 Tip: You can also run EcoBound from a terminal by activating the cloned environment and running Python scripts directly:

```bash
conda activate arcgispro-py3-clone
python open_ecobound.py
```

This method supports batch processing, reproducibility, and integration with other tools like Jupyter or GEE Python API.

---

#### 🧰 Toolbox Mode (ArcGIS Pro GUI)

For users who prefer working in the graphical interface of ArcGIS Pro, EcoBound is also available as a Toolbox file (`Ecobound.atbx`).

##### 📦 Step-by-step to run EcoBound in Toolbox Mode:

1. **Switch to the cloned Python environment with `rasterio` installed**  
   (created in Step 1 and configured in Step 2):

   - In ArcGIS Pro, go to **Project > Package Manager**
   - Under **Active Environment**, select your cloned environment (e.g., `arcgispro-py3-clone`)
   - Restart ArcGIS Pro

2. **Add the EcoBound Toolbox to your project**:

   - Go to **Catalog > Toolboxes**
   - Right-click and choose **Add Toolbox...**
   - Browse to the location of `".\EcoBound Python Package\Ecobound.atbx"` and click **Open**

3. **Use the EcoBound tools in ArcGIS GUI**:

   The toolbox contains the following tools, each corresponding to a core EcoBound function:

   - `Ecological Trend Analysis`
   - `GeoDetector Advanced`
   - `Entropy-Based Threshold Detector`
   - `Advanced Risk Detector`

   Click to open any tool and fill in the parameters using the GUI.

   ![open toolbox model](.\images\open toolbox model.png)

> ✅ This mode is ideal for GIS analysts who prefer visual workflows without writing code.

> ⚠️ Make sure to activate the correct environment before running any tool, otherwise `rasterio`-dependent functions will fail.

---

## 🧭 Run EcoBound with Example Data

### Over view

EcoBound consists of several modular components, each corresponding to a core function in ecological boundary and risk detection.

| Module               | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `trend_analysis`     | Detect long-term ecological trends using Mann-Kendall + Sens Slope |
| `align_rasters`      | Align all input rasters to a common grid before analysis     |
| `run_geodetector`    | Run GeoDetector (factor, interaction, risk, and eco modules) |
| `adv_risk`           | Conduct advanced risk detection using MACD + Bollinger Band logic |
| `ecobound_threshold` | Identify ecological thresholds and generate boundary maps    |

> 🧪 Each module is accessible via both Script Mode and Toolbox Mode.

#### 🧪 Example input rasters

The folder `.EcoBound Python Package/input/` contains resampled example rasters (1000m resolution), derived from open-access datasets listed in the manuscript.

- Total file size: 32.1MB
- Spatial resolution: 1000 meters
- Time period: 1990–2022 (subset)

The full-resolution input data (30m and 100m) are not provided due to size, but can be freely accessed from sources listed in the paper.

---

### 🔍 Trend analysis

Detects long-term ecological trends in time-series raster data using the Mann-Kendall test and Sen’s slope estimator.

#### 🔧Parameters

trend_analysis(input_folder, output_folder, radius, unit)

| Parameter       | Type  | Required | Description                                                  |
| --------------- | ----- | -------- | ------------------------------------------------------------ |
| `input_folder`  | `str` | ✅ Yes    | Folder path containing ecological rasters with temporal sequence (e.g., NDVI from 2002–2023). <br/>Each raster filename must contain a 4-digit number (e.g., `NDVI_2001.tif`, `LST_0930.tif`), which is used to define the temporal sequence. These numbers are sorted in ascending order to construct the time series. The number does not have to be a year — it can represent a day, hour, or other custom time marker. Missing intervals are allowed. |
| `output_folder` | `str` | ✅ Yes    | Folder to store the result rasters: `Sens_Slope.tif` and `Z_score.tif`. |
| `radius`        | `int` | ✅ Yes    | Radius for focal statistics. A circular neighborhood is constructed using Esri’s `NbrCircle` method, which defines a circle with the given radius centered on each cell. The unit of radius is specified separately below. |
| `unit`          | `str` | ✅ Yes    | Unit for the radius: <br/>- `MAP` = map units (e.g., meters or degrees depending on raster CRS)  <br/>- `CELL` = number of raster cells |

#### 📁 Folder structure

Your project folder should look like this:

```
EcoBound Python Package/
├── Ecobound/                      # Core code package
├── 00 example_usage_trend.py        # Example script
├── input/
│   └── Barren_resample/          # Example input rasters
│       ├── Barren_1990.tif
│       ├── Barren_1991.tif
│       └── ...
├── output/
│   └── Barren_testout/           # Will be auto-created by the script
└── README.md
```

#### 💻 Code Sample (Script Mode)

```python
from Ecobound import trend_analysis
import arcpy

arcpy.env.overwriteOutput = True

input_folder = r".\input\Barren_resample"
output_folder = r".\output\Barren_testout"
radius = 5000
unit = "MAP"

trend_analysis(input_folder, output_folder, radius, unit)
```

This script uses the built-in `IDLE` environment from ArcGIS Pro. No additional environment setup is required beyond **Step 1–2** in this guide.

##### 💡 Example: Ecological Trend Analysis

You can test EcoBound by running the following example script:

```bash
example_usage_trend.py
```

This script performs ecological trend analysis using time-series raster data.

#### 🧰 ArcGIS Pro GUI (Toolbox Mode)

EcoBound also provides a toolbox interface for users preferring GUI-based workflows. In ArcGIS Pro:

1. Open **Project > Package Manager**, and activate your clone environment (e.g., `arcgispro-py3-clone`)
2. Go to **Catalog > Toolboxes**, right-click → **Add Toolbox...**
3. Select and load `/EcoBound Python Package/Ecobound.atbx`
4. Run the tool **Ecological Trend Analysis**
5. Input parameters are the same as in the script version

> ✅ The underlying logic is identical to the Python function `trend_analysis()`. This mode is ideal for GIS analysts and students unfamiliar with coding.

![ECR tool box mod](.\images\ECR tool box mod.png)

#### 📤 Output Files

The function generates the following files in the `output_folder`:

- `Sens_Slope.tif`: Raster of Sen’s slope, indicating the direction and magnitude of ecological change
- `Z_score.tif`: Raster of Mann-Kendall Z-scores, indicating the **statistical significance** of the trend

> 📊 Values of `Z_score` greater than **1.65**, **1.96**, and **2.58** correspond to significance at the **90%**, **95%**, and **99%** confidence levels, respectively.

---

### 📐 Raster Alignment

Aligns multiple input rasters to a common grid using a template raster. This ensures consistent spatial extent, resolution, and alignment across all input layers before analysis.

---

#### 🔧 Parameters

| Parameter     | Type   | Description                                                  |
| ------------- | ------ | ------------------------------------------------------------ |
| `template`    | `str`  | Path to the template raster. Raster layer to be used as spatial alignment template. All input rasters will be resampled and aligned to match this raster’s grid. |
| `raster_list` | `list` | List of paths to input rasters to align.                     |
| `output_path` | `str`  | Directory to save aligned rasters.                           |
| `continuous`  | `list` | List of `True/False` flags indicating whether each raster is continuous (True) or categorical (False). |

---

#### 📁 Folder Structure

```
EcoBound Python Package/
├── input/
│   ├── X 1000m/
│   │   ├── aspectresample.tif
│   │   ├── clay.tif
│   │   ├── demresample.tif
│   │   ├── FHP.tif
│   │   └── ... #(The example input for parameter raster_list)
├── output/
	├── Barren_testout/
		└── Sens_Slope.tif #(The output of the previous stpe, parameter template)
	└── X_alignment/ #(parameter output_path)
└── 01 example usage raster alignment.py (Code sample)
```

---

#### 💻 Code Sample (Script Mode)

```python
from Ecobound import align_rasters
import arcpy

# 0 Set environment
arcpy.env.overwriteOutput = True

# 1 Raster alignment
template = r".\output\Barren_testout\Sens_Slope.tif"

raster_list = [
    r".\input\X 1000m\aspectresample.tif",
    r".\input\X 1000m\clay.tif",
    r".\input\X 1000m\demresample.tif",
    r".\input\X 1000m\FHP.tif",
    r".\input\X 1000m\gravel.tif",
    r".\input\X 1000m\pre.tif",
    r".\input\X 1000m\sand.tif",
    r".\input\X 1000m\sloperesample.tif",
    r".\input\X 1000m\soilph.tif",
    r".\input\X 1000m\temp.tif"
]

continuous = [
    True,  # aspect
    True,  # clay
    True,  # dem
    True,  # FHP
    True,  # gravel
    True,  # pre
    True,  # sand
    True,  # slope
    True,  # soil ph
    True   # temp
]

output_path = r".\output\X_alignment"

aligned_files = align_rasters(
    template = template,
    raster_list = raster_list,
    output_path = output_path,
    continuous = continuous
)
```

---

#### 🧰 ArcGIS Pro GUI (Toolbox Mode)

In the toolbox `Ecobound.atbx`, use the tool **Raster Alignment**.

1. Select a template raster
2. Add all input rasters
3. Specify whether each one is continuous
4. Choose output folder

![onen ecobound script mode](.\images\Raster aliment tool box mod.png)

---

#### 📤 Output Files

- Aligned raster layers saved in the specified output folder
- File names and formats are preserved (e.g., `GDP.tif`, `NDVI.tif`, `LST.tif`)

> ✅ This prepares input data for further analysis such as trend detection and GeoDetector. Recommended as the **first step** in any workflow using heterogeneous raster datasets.

---

### 🧪 GeoDetector Analysis

Performs geographical detection of explanatory variables for a target ecological indicator using spatial stratification. The mathematical foundation of all calculations is fully consistent with the original GeoDetector framework (see [Wang et al., 2020](https://doi.org/10.1080/15481603.2020.1760434)).

Included modules: **Factor Detector**, **Interaction Detector**, **Risk Detector**, and **Ecological Detector**.

---

#### 🔧 Parameters

| Parameter          | Type    | Required | Description                                                  |
| ------------------ | ------- | -------- | ------------------------------------------------------------ |
| `y_raster`         | `str`   | ✅ Yes    | Path to the dependent variable raster (e.g., trend output such as Sens_Slope.tif). |
| `x_aligned_folder` | `str`   | ✅ Yes    | Folder containing aligned explanatory variables (e.g., output from Raster Alignment). |
| `output_dir`       | `str`   | ✅ Yes    | Directory to store output CSVs and raster results.           |
| `slice_types`      | `list`  | ✅ Yes    | List of discretization methods (e.g., `["EQUAL_INTERVAL", "EQUAL_AREA", "NATURAL_BREAKS", "GEOMETRIC_INTERVAL"]`). |
| `number_zones`     | `list`  | ✅ Yes    | List of class break numbers for stratification (e.g., `[3, 4, 5]`). |
| `mode`             | `str`   | ✅ Yes    | Detection mode: `"qmax"` selects the stratification with the highest q value for each variable. `"all"` keeps all significant slice results.  <br/>> ⚠️ `mode="all"` may generate dozens of outputs per variable by retaining all statistically significant stratifications.  <br/>> For clarity and reproducibility, we recommend using the default `"qmax"` mode, which selects only the most explanatory result for each factor. |
| `alpha`            | `float` | ✅ Yes    | Significance level for q-statistic filtering (commonly set to `0.05`). |
| `modules`          | `list`  | ✅ Yes    | List of modules to run: `["factor", "interaction", "risk", "eco"]`.<br/><br/>> 📌 Module reference:  <br/>> - `"factor"` → **Factor Detector** (required)  <br/>> - `"interaction"` → **Interaction Detector**  <br/>> - `"risk"` → **Risk Detector**  <br/>> - `"eco"` → **Ecological Detector**<br/><br/>> ⚠️ `"factor"` is mandatory for all workflows and cannot be excluded. |

---

#### 📁 Folder Structure

```
EcoBound Python Package/
├── output/
│   ├── Barren_testout/
│   │   └── Sens_Slope.tif         # Example Y raster, from trend analysis output
│   ├── X_alignment/               # Example X rasters, from raster alignment output
│   │   ├── aspectresample.tif
│   │   ├── clay.tif
│   │   └── ...
│   └── geodetector/               # Output folder for GeoDetector results
└── 02 example usage geodector.py  # Code sample for this module
```

---

#### 💻 Code Sample (Script Mode)

```python
from Ecobound import run_geodetector
import arcpy

# 0 Set environment
arcpy.env.overwriteOutput = True

# 1 Run GeoDetector
run_geodetector(
    y_raster = r".\output\Barren_testout\Sens_Slope.tif",
    x_aligned_folder = r".\output\X_alignment",
    output_dir = r".\output\geodetector",
    slice_types = ["EQUAL_INTERVAL", "EQUAL_AREA", "NATURAL_BREAKS", "GEOMETRIC_INTERVAL"],
    number_zones = [3, 4, 5],
    mode = "qmax",
    alpha = 0.05,
    modules = ["factor", "interaction", "risk", "eco"]
)
```

---

#### 🧰 ArcGIS Pro GUI (Toolbox Mode)

From the toolbox `Ecobound.atbx`, choose the **GeoDetector Analysis** tool.

1. Input the Y raster (e.g., `from trend analysis output`)
2. Select the folder of aligned explanatory variables (e.g., 'from raster alignment output')
3. Choose slicing method(s) and number of classes
4. Set significance threshold (`alpha`)
5. Select modules to run
   ![GEODECTOR tool box mod](.\images\GEODECTOR tool box mod.png)

---

#### 📤 Output Files

- `q_results.csv`: Raw output from the **Factor Detector**, including all slicing schemes and their corresponding q-values and p-values.
- `filtered_q_results.csv`: Filtered results keeping only the best slicing (highest q-value) for each variable. This is used when `mode="qmax"`.
- `interaction_results.csv`: Output from the **Interaction Detector**, showing the interaction types and strength (Δq) between variable pairs.
- `risk_results.csv`: Output from the **Risk Detector**, including spatial risk values and fluctuation metrics.
- `eco_results.csv`: Output from the **Ecological Detector**, including entropy-based thresholds and boundary locations.
- `X_sliced/`: Raster files of the best slicing scheme for each variable (used for spatialization and threshold detection).

### 📊 Advanced Risk Detector

This module detects dynamic ecological risk regions using an adapted MACD (Moving Average Convergence Divergence) model combined with Bollinger Band logic.  
It is designed for tracking shifts in ecological indicator fluctuations over stratified classes of explanatory variables.

> ✅ Designed for adaptive risk zoning based on indicator dynamics across environmental gradients.

---

#### 🔧 Parameters

| Parameter            | Type    | Required | Description                                                  |
| -------------------- | ------- | -------- | ------------------------------------------------------------ |
| `x_folder`           | `str`   | ✅ Yes    | Path to the folder containing aligned explanatory variables (e.g., output from Raster Alignment). |
| `y_raster`           | `str`   | ✅ Yes    | Raster file representing the dependent ecological indicator (e.g., Sens_Slope.tif from trend analysis). |
| `output_folder`      | `str`   | ✅ Yes    | Folder to store output risk rasters and SVG plots.           |
| `num_bins`           | `int`   | ✅ Yes    | C1: Number of bins for X segmentation (default: 100).        |
| `std_factor`         | `float` | ✅ Yes    | C2: Width of Bollinger bands in standard deviations (default: 1.5). |
| `sma_short_bins`     | `int`   | ✅ Yes    | C3: Short-term simple moving average (SMA) window size (default: 10 bins). |
| `sma_long_bins`      | `int`   | ✅ Yes    | C4: Long-term SMA window size (default: 20 bins).            |
| `macd_short_period`  | `int`   | ✅ Yes    | C5: Short EMA period for MACD computation (default: 12).     |
| `macd_long_period`   | `int`   | ✅ Yes    | C6: Long EMA period for MACD computation (default: 26).      |
| `macd_signal_period` | `int`   | ✅ Yes    | C7: MACD signal line period (default: 9).                    |
| `k_factor`           | `float` | ✅ Yes    | Threshold factor for Risk Level classification (default: 1.0). |
| `svg_only`           | bool    | ✅ Yes    | Whether to export only SVG plots (default: True). Set to False to export both `.svg` and `.png`. |

---

#### 📁 Folder Structure

```
EcoBound Python Package/
├── output/
│   ├── X_alignment/               # Input X rasters (aligned)
│   ├── Barren_testout/           # Input Y raster: Sens_Slope.tif
│   └── adv_risk/                 # Output from advanced risk detector
└── 03 example usage adv risk.py  # Code sample
```

---

#### 💻 Code Sample (Script Mode)

```python
from Ecobound import adv_risk
import arcpy

# 0 Set environment
arcpy.env.overwriteOutput = True

# 1 Run advanced risk detector
adv_risk(
    x_folder = r".\output\X_alignment",
    y_raster = r".\output\Barren_testout\Sens_Slope.tif",
    output_folder = r".\output\adv_risk",
    num_bins = 100,
    std_factor = 1.5,
    sma_short_bins = 10,
    sma_long_bins = 20,
    macd_short_period = 12,
    macd_long_period = 26,
    macd_signal_period = 9,
    k_factor = 1.0,
    svg_only = True
)
```

---

#### 🧰 ArcGIS Pro GUI (Toolbox Mode)

From the toolbox `Ecobound.atbx`, select **Advanced Risk Detector**.

1. Provide aligned explanatory rasters (X)

2. Input the ecological indicator raster (Y)

3. Specify binning and smoothing parameters

4. Click Run

   ![risk_adv tool box mod](.\images\risk_adv tool box mod.png)

---

#### 📤 Output Files

- `adv_risk/*_Risk_Value.tif`: Continuous raster of MACD-based ecological risk values.
- `adv_risk/*_Risk_Level.tif`: Classified raster of risk levels based on the threshold factor (`k_factor`).
- `adv_risk/*_Risk.svg`: Visual diagram showing MACD curve, signal line, Bollinger bands, and detected thresholds.

---

### 🌄 EcoBound and Natural Geographic Boundary Detector

This tool identifies **primary ecological thresholds** of environmental variables (X) in relation to a response variable (Y) using entropy-guided segmentation and variance reduction. It optionally generates **natural boundary shapefiles** based on the detected thresholds.

---

#### 🔧 Parameters

| Parameter           | Type   | Required | Description                                                  |
| ------------------- | ------ | -------- | ------------------------------------------------------------ |
| `x_folder`          | `str`  | ✅ Yes    | Path to the folder containing input X rasters (e.g., environmental drivers). |
| `y_raster`          | `str`  | ✅ Yes    | Path to the dependent variable raster (e.g., NDVI, EVI, Sens_Slope). |
| `output_folder`     | `str`  | ✅ Yes    | Folder to store output CSVs, SVG plots, and boundary shapefiles. |
| `num_bins`          | `int`  | ✅ Yes    | Number of bins for entropy-guided segmentation (default: 100). |
| `b_bins`            | `int`  | ✅ Yes    | Number of bins for variance reduction evaluation (default: 30). |
| `permutations`      | `int`  | ✅ Yes    | Number of permutations for ΔV significance test. Set to 0 to skip (default: 0 for demo; recommend 9999 for formal analysis). |
| `svg_only`          | `bool` | ✅ Yes    | Whether to export only SVG plots (default: True). Set to False to export both `.svg` and `.png`. |
| `generate_boundary` | `bool` | ✅ Yes    | Whether to generate natural boundary shapefiles from detected thresholds. |

---

### 📁 Folder Structure

Example:

```
EcoBound Python Package/
├── output/
│   ├── Barren_testout/
│   │   └── Sens_Slope.tif         # Example Y raster, from trend analysis output
│   ├── X_alignment/               # Example X rasters, from raster alignment output
│   │   ├── aspectresample.tif
│   │   ├── clay.tif
│   │   └── ...
│   └── ecobound/                  # Output folder for Ecobound results
└── 04 example usage threshold.py  # Code sample for this module
```

---

### 💻 Code Sample (Script Mode)

```python
from Ecobound import ecobound_threshold
import arcpy

# 0 Set environment
arcpy.env.overwriteOutput = True

# 1 Test run for EcoBound threshold analysis (Example usage)
ecobound_threshold(
    x_folder = r".\output\X_alignment",
    y_raster = r".\output\Barren_testout\Sens_Slope.tif",
    output_folder = r".\output\ecobound",
    num_bins = 100,     # Number of bins for first-layer segmentation (entropy scan)
    b_bins = 30,        # Number of bins for second-layer evaluation (used for VR calculation)
    permutations = 0,   # Permutation test iterations (set to 0 to skip significance test)
    svg_only = True,
    ecobound = True
)
```

---

### 🧰 ArcGIS Pro GUI (Toolbox Mode)

This tool can also be accessed via ArcGIS Pro Toolbox. Parameters are identical to those shown above, with descriptions provided in both English and Chinese.

- Simply point to the aligned `X` folder and `Y` raster

- Select number of bins, set permutation count

- Check the option to generate boundary shapefiles (optional)

  ![EcoBound tool box mod](.\images\EcoBound tool box mod.png)

---

### 📤 Output Files

Each X raster produces the following outputs in `output_folder`:

- `threshold_results.csv`: Summary of entropy thresholds, variance reduction, and p-values.

- `*_Risk_Threshold.svg`: Curve plots visualizing entropy scan and ΔV reduction. The result CSV from the EcoBound and Natural Geographic Boundary Detector tool includes the following fields:

  | Column      | Description (English)                                        |
  | ----------- | ------------------------------------------------------------ |
  | `T_entropy` | **Threshold (T\*) determined by entropy scanning**. The optimal segmentation point along the environmental gradient X that maximizes structural information gain (Equation S1-25). |
  | `VR`        | **Variance Reduction index**. Measures how much of the variance in ecological response Y is explained by splitting at T\* (Equation S1-28). Higher values indicate stronger separability. |
  | `p_val`     | **p-value of permutation test on ΔV**. Statistical significance of the threshold T\* based on repeated random permutations of Y (Equation S1-29). |

  This file is typically named `threshold_results.csv` and saved to the output folder specified by the user.

- `*_Boundary.shp`: Shapefile of detected ecological threshold (if `generate_boundary=True`).

---

## 🛡 License

All code, figures, and documentation in the **EcoBound** package are licensed under the  
[Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

This means:

- ✅ You **may use, modify, and redistribute** the content **for non-commercial research or educational purposes**.
- ❌ **Commercial use is strictly prohibited** without prior permission.
- ✍️ **You must retain attribution** to the original authors in any derivative works or redistributed copies.

## 📖 Citation Requirement

If you use **EcoBound** or any of its components (e.g., GeoDetector, MACD-based risk, entropy threshold detection) in academic research, reports, or presentations, you **must cite** the following reference:

> Wanghe, K., et al. *EcoBound: A Python package for GeoDetector, Ecological Risk Detection, and Natural Geographical Boundary Extraction*  （Unpublished）
>
> This GitHub page before the above paper published.

Failure to properly cite the original work may constitute a violation of this license.

## 🔗 Contact

For questions, feedback, or collaboration inquiries, please contact:

**Kunyuan Wanghe**  
Northwest Institute of Plateau Biology, Chinese Academy of Sciences  
📧 Email: [wanghekunyuan@189.cn](mailto:wanghekunyuan@189.cn)  
📧 CC: [wanghekunyuan@gmail.com](mailto:wanghekunyuan@gmail.com)
🔗 GitHub: https://github.com/wanghekunyuan
