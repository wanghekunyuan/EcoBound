# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import matplotlib.pyplot as plt  # 保持与原文件兼容；当前未额外使用

from .ecobound_analysis import EcoBoundAnalyzer
from .segmentation import generate_natural_boundary


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


def _export_threshold_profile(analyzer, basename, output_folder, best_k):
    """
    v3.30 Debug 3:
    默认导出每个切点的完整 threshold profile（CSV），不改前端、不改函数签名。

    输出：
        {basename}_threshold_profile.csv

    若 analyzer 中不存在 profile_* 属性，则返回 None，并打印 warning。
    """
    if not hasattr(analyzer, "profile_k") or not hasattr(analyzer, "profile_T") or not hasattr(analyzer, "profile_score"):
        print(f"⚠️ Threshold profile not found in analyzer for {basename}; skipping profile export.")
        return None

    if analyzer.profile_k is None or analyzer.profile_T is None or analyzer.profile_score is None:
        print(f"⚠️ Threshold profile is None for {basename}; skipping profile export.")
        return None

    if len(analyzer.profile_k) == 0:
        print(f"⚠️ Threshold profile is empty for {basename}; skipping profile export.")
        return None

    df_profile = pd.DataFrame({
        "X_name": basename,
        "k_full": analyzer.profile_k,
        "candidate_threshold": analyzer.profile_T,
        "weighted_information_gain": analyzer.profile_score,
    })

    df_profile["is_best"] = (df_profile["k_full"].astype(int) == int(best_k)).astype(int)
    df_profile["rank_desc"] = df_profile["weighted_information_gain"].rank(method="dense", ascending=False).astype(int)
    df_profile = df_profile.sort_values("candidate_threshold").reset_index(drop=True)

    out_csv = os.path.join(output_folder, f"{basename}_threshold_profile.csv")
    df_profile.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Threshold profile saved: {out_csv}")
    return df_profile


def batch_ecobound_threshold(
    x_folder,
    y_raster,
    output_folder,
    num_bins=100,
    b_bins=30,
    permutations=999,
    svg_only=True,
    ecobound=True,
    x_original_folder=None  # 原始 X（未对齐）文件夹，仅用于生成边界线
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
            第一次分箱数量，默认 100
        b_bins : int
            第二次分箱数量，默认 30
        permutations : int
            置换检验次数，默认 999，设为 0 可跳过检验
        svg_only : bool
            是否只保存 SVG 图（默认 True）
        ecobound : bool
            是否尝试输出边界（默认 True）
        x_original_folder : str or None
            原始 X（未对齐）文件夹，仅用于生成边界线

    Debug 3（v3.30）新增默认输出：
        - 每个 X 一份完整 threshold profile CSV
        - 一份合并后的总 profile CSV

    前端 / 函数输入 / UI 完全不变。
    """
    os.makedirs(output_folder, exist_ok=True)

    result_rows = []
    all_profile_rows = []

    for file in os.listdir(x_folder):
        if not file.lower().endswith(".tif"):
            continue

        x_path = os.path.join(x_folder, file)
        basename = os.path.splitext(file)[0]

        analyzer = EcoBoundAnalyzer(x_path, y_raster)
        T_entropy, VR, best_k = analyzer.run_ecobound(C1=num_bins, B_bins=b_bins)

        # === Debug 3：默认导出完整 threshold profile（不改前端） ===
        df_profile = _export_threshold_profile(analyzer, basename, output_folder, best_k)
        if df_profile is not None:
            all_profile_rows.append(df_profile)

        # === 生成自然地理边界（可选） ===
        if ecobound and (T_entropy is not None):
            # ① 默认用当前对齐版 X 出线/面
            raster_for_line = x_path

            # ② 如用户提供了原始 X 文件夹，则尝试按“同名去后缀”匹配原始 X
            original_match = _find_original_x(x_path, x_original_folder)
            if original_match:
                print(f"✅ Using ORIGINAL X for boundary: {os.path.basename(original_match)}")
                raster_for_line = original_match
            else:
                if x_original_folder:
                    print(
                        "⚠️ No matching ORIGINAL X found in x_original_folder; "
                        "falling back to aligned X for boundary. Geometry may be fragmented by NoData."
                    )

            out_shp = os.path.join(output_folder, f"{basename}_EcoBound.shp")
            try:
                generate_natural_boundary(raster_for_line, T_entropy, out_shp)
                print(f"✅ Boundary saved: {out_shp}")
            except Exception as e:
                print(f"❌ Failed to generate boundary for {basename}: {e}")

        # === permutation ===
        if permutations > 0:
            p_val, _ = analyzer.run_permutation_test(repeat=permutations)
        else:
            p_val = "-"

        # === 曲线图 ===
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

    # === 保存汇总 CSV ===
    df = pd.DataFrame(result_rows)
    summary_csv = os.path.join(output_folder, "ecobound_summary.csv")
    df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Summary saved: {summary_csv}")

    # === 保存合并后的 profile CSV ===
    if len(all_profile_rows) > 0:
        df_all_profile = pd.concat(all_profile_rows, ignore_index=True)
        all_profile_csv = os.path.join(output_folder, "ecobound_threshold_profiles_all.csv")
        df_all_profile.to_csv(all_profile_csv, index=False, encoding="utf-8-sig")
        print(f"✅ Combined threshold profiles saved: {all_profile_csv}")
    else:
        print("⚠️ No threshold profiles were exported.")

    print("✅ EcoBound threshold analysis complete.")
