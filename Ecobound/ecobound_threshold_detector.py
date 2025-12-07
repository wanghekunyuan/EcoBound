import os
import pandas as pd
from .ecobound_analysis import EcoBoundAnalyzer
from .segmentation import generate_natural_boundary
import matplotlib.pyplot as plt
import re

def _find_original_x(x_aligned_path, x_original_folder):
    """在 x_original_folder 中按“同名去后缀”匹配原始 X；匹配不到返回 None。"""
    if not x_original_folder or not os.path.isdir(x_original_folder):
        return None

    base = os.path.splitext(os.path.basename(x_aligned_path))[0]
    # 去掉常见对齐后缀：_align/_aligned/-align/-aligned/.align/.aligned（大小写不敏感）
    normalized = re.sub(r'(?i)[._-]?align(ed)?$', '', base)

    # 1) 先尝试严格同名 + 去后缀同名
    strict_candidates = [
        os.path.join(x_original_folder, base + ".tif"),
        os.path.join(x_original_folder, normalized + ".tif"),
    ]
    for c in strict_candidates:
        if os.path.exists(c):
            return c

    # 2) 退而扫描整个文件夹，做“去后缀后的不区分大小写完全相等”匹配
    for fn in os.listdir(x_original_folder):
        name, _ = os.path.splitext(fn)
        name_norm = re.sub(r'(?i)[._-]?align(ed)?$', '', name)
        if name_norm.lower() == normalized.lower():
            return os.path.join(x_original_folder, fn)

    return None





def batch_ecobound_threshold(
    x_folder,
    y_raster,
    output_folder,
    num_bins=100,
    b_bins=30,
    permutations=999,
    svg_only=True,
    ecobound = True,
    x_original_folder = None   # 👈 新增：原始 X（未对齐）文件夹，仅用于生成边界线
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
            # === 生成自然地理边界线（可选） ===
            if ecobound and (T_entropy is not None):
                # ① 默认用当前对齐版 X 出线
                raster_for_line = x_path

                # ② 如用户提供了原始 X 文件夹，则尝试按“同名去后缀”匹配原始 X
                original_match = _find_original_x(x_path, x_original_folder)
                if original_match:
                    print(f"🟢 Using ORIGINAL X for boundary: {os.path.basename(original_match)}")
                    raster_for_line = original_match
                else:
                    if x_original_folder:
                        print("⚠️ No matching ORIGINAL X found in x_original_folder; "
                              "falling back to aligned X for boundary. Geometry may be fragmented by NoData.")

                # ③ 调用现有的出线函数（不改其实现）
                out_shp = os.path.join(output_folder, f"{basename}_EcoBound.shp")
                try:
                    generate_natural_boundary(raster_for_line, T_entropy, out_shp)
                    print(f"✅ Boundary saved: {out_shp}")
                except Exception as e:
                    print(f"❌ Failed to generate boundary for {basename}: {e}")


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
