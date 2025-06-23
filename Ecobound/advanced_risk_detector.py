
import os
import matplotlib.pyplot as plt
from .fig8_7 import advanced_risk_detector

def batch_advanced_risk_detector(
    x_folder,
    y_raster,
    output_folder,
    num_bins=100,
    std_factor=1.5,
    sma_short_bins=10,
    sma_long_bins=20,
    macd_short_period=12,
    macd_long_period=26,
    macd_signal_period=9,
    k_factor=1.0,
    svg_only=True
):
    """
    对一个文件夹中的所有 X 栅格执行高级环境风险探测，输出两个 .tif 和一个 svg 图。

    参数：
        x_folder : str
            包含多个 X 栅格的文件夹路径（.tif）
        y_raster : str
            响应变量栅格（Y）
        output_folder : str
            输出目录
        svg_only : bool
            是否仅保存 svg 图（默认 True，若 False 还保存 png）
    """
    os.makedirs(output_folder, exist_ok=True)

    x_raster_list = [
        os.path.join(x_folder, f) for f in os.listdir(x_folder)
        if f.lower().endswith(".tif")
    ]

    for x_path in x_raster_list:
        base_name = os.path.splitext(os.path.basename(x_path))[0]
        risk_level_tif = os.path.join(output_folder, f"{base_name}_Risk_Level.tif")
        risk_value_tif = os.path.join(output_folder, f"{base_name}_Risk_Value.tif")
        svg_path = os.path.join(output_folder, f"{base_name}_Risk.svg")
        png_path = os.path.join(output_folder, f"{base_name}_Risk.png")

        print(f"🌀 Processing: {base_name}")

        risk_df, fig = advanced_risk_detector(
            rasX_path = x_path,
            rasY_path = y_raster,
            output_risk_level_tif_path = risk_level_tif,
            output_risk_value_tif_path = risk_value_tif,
            num_bins=num_bins,
            std_factor=std_factor,
            sma_short_bins=sma_short_bins,
            sma_long_bins=sma_long_bins,
            macd_short_period=macd_short_period,
            macd_long_period=macd_long_period,
            macd_signal_period=macd_signal_period,
            k_factor=k_factor
        )

        fig.savefig(svg_path, format="svg")
        if not svg_only:
            fig.savefig(png_path, dpi=300)

        plt.close(fig)
        print(f"✅ Done: {base_name}\n")
