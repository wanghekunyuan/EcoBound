import os
import pandas as pd
from .ecobound_analysis import EcoBoundAnalyzer
from .segmentation import generate_natural_boundary
import matplotlib.pyplot as plt

def batch_ecobound_threshold(
    x_folder,
    y_raster,
    output_folder,
    num_bins=100,
    b_bins=30,
    permutations=999,
    svg_only=True,
    ecobound = True
):
    """
    批量执行 EcoBound 边界阈值识别（Entropy-based ecological threshold detection）

    参数：
        x_folder : str
            存放 X 环境变量栅格（.tif）的文件夹
        y_raster : str
            响应变量 Y 的栅格路径（.tif）
        output_folder : str
            输出图表和 CSV 的路径
        num_bins : int
            分箱数量，默认 100
        permutations : int
            置换检验次数，默认 999，设为 0 可跳过检验
        svg_only : bool
            是否只保存 SVG 图（默认 True）
    """
    os.makedirs(output_folder, exist_ok=True)
    result_rows = []

    for file in os.listdir(x_folder):
        if file.lower().endswith(".tif"):
            x_path = os.path.join(x_folder, file)
            basename = os.path.splitext(file)[0]

            analyzer = EcoBoundAnalyzer(x_path, y_raster)
            T_entropy, VR, best_k = analyzer.run_ecobound(C1=num_bins, B_bins=b_bins)
            if ecobound:
                natural_boundary_shp = os.path.join(output_folder, f"boundary_{os.path.basename(x_path).replace('.tif', '.shp')}")
                generate_natural_boundary(x_path, T_entropy, natural_boundary_shp)

            if permutations > 0:
                p_val, _ = analyzer.run_permutation_test(repeat=permutations)
            else:
                p_val = "-"

            svg_path = os.path.join(output_folder, f"{basename}_curve.svg")
            analyzer.plot(save_path=svg_path, show=False, dpi=300)
            if not svg_only:
                jpg_path = os.path.join(output_folder, f"{basename}_curve.jpg")
                analyzer.plot(save_path=jpg_path, show=False, dpi=300)

            result_rows.append({
                "X_name": basename,
                "T_entropy": T_entropy,
                "VR": VR,
                "p_val": p_val
            })

    # 保存汇总 CSV
    df = pd.DataFrame(result_rows)
    df.to_csv(os.path.join(output_folder, "ecobound_summary.csv"), index=False)
    print("✅ EcoBound threshold analysis complete.")
