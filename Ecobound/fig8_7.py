import rasterio
import numpy as np
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator        # small：减少主刻度数量
from matplotlib.patches import Patch             # 风险色块图例

from scipy.interpolate import make_interp_spline
import pandas as pd
#from matplotlib.patches import Patch
#from rasterio.transform import from_origin
import os

def advanced_risk_detector(
    rasX_path,
    rasY_path,
    output_risk_level_tif_path,    # 输出风险等级栅格的路径
    output_risk_value_tif_path,    # 输出风险值栅格的路径
    num_bins=100,                  # 分箱数量 (C1)
    std_factor=1.5,                # 布林带倍数 (C2)
    sma_short_bins=10,             # 短期SMA窗口 (C3)
    sma_long_bins=20,              # 长期SMA窗口 (C4)
    macd_short_period=12,          # MACD短期EMA窗口 (C5)
    macd_long_period=26,           # MACD长期EMA窗口 (C6)
    macd_signal_period=9,          # MACD信号线EMA窗口 (C7)
    k_factor=1.0,                   # 风险分类阈值系数 (k)，默认=1
    layout=["large"]               # 新增参数：版式，支持 ["large"] 或 ["small"]
):
    """
    以栅格 rasX (环境因子) 与 rasY (生态或环境属性) 为输入，使用Advanced Risk Detector方法
    完成以下步骤：
      1) 分箱并计算 Y 的均值与标准差
      2) 计算布林带 (mean ± std_factor * std)
      3) 计算10-bin与20-bin SMA（可自定义）
      4) 计算MACD指标 (12, 26, 9)（可自定义）
      5) 动态阈值风险分类：High / Medium / Low
         - High Risk:   abs(MACD - Signal) >  + k_factor * threshold
         - Low Risk:    abs(MACD - Signal) < - k_factor * threshold
         - Medium Risk: 其余
      6) 输出 Risk_Value 和 Risk_Level 栅格文件
    
    参数：
    -------
    rasX_path : str
        环境因子（例如Aspect.tif）的文件路径
    rasY_path : str
        目标生态或环境属性（例如EVI.tif）的文件路径
    output_risk_level_tif_path : str
        输出风险等级栅格文件的路径
    output_risk_value_tif_path : str
        输出风险值栅格文件的路径
    num_bins : int
        将X值等距分箱的数量，默认100
    std_factor : float
        用于构建“布林带”的倍数 (默认1.5)
    sma_short_bins : int
        短期SMA窗口大小 (默认10)
    sma_long_bins : int
        长期SMA窗口大小 (默认20)
    macd_short_period : int
        MACD短期EMA窗口大小 (默认12)
    macd_long_period : int
        MACD长期EMA窗口大小 (默认26)
    macd_signal_period : int
        MACD信号线的EMA窗口大小 (默认9)
    k_factor : float
        风险分类时threshold的倍数系数 (默认1.0)
    
    返回：
    -------
    risk_df : pd.DataFrame
        每个bin的中心值、均值、标准差、样本数、SMA、MACD等信息，以及风险分类结果
    fig : matplotlib.figure.Figure
        绘制的图形，可以根据需要自行plt.show()或保存
    """

    # =============== 读取栅格并掩膜NoData ===============
    def extract_raster_values(ras_path):
        with rasterio.open(ras_path) as src:
            arr = src.read(1)
            nodata_value = src.nodata
            transform = src.transform
            crs = src.crs
        return arr, nodata_value, transform, crs

    # 读取X与Y栅格
    rasX_values, nodata_value_X, transform_X, crs_X = extract_raster_values(rasX_path)
    rasY_values, nodata_value_Y, transform_Y, crs_Y = extract_raster_values(rasY_path)
    
    # 确定NoData值并应用掩膜
    if nodata_value_X != nodata_value_Y:
        nodata_value = nodata_value_X
    else:
        nodata_value = nodata_value_X

    mask = (rasX_values != nodata_value) & (rasY_values != nodata_value)
    rasX_flat = rasX_values[mask]
    rasY_flat = rasY_values[mask]

    # =============== 分箱并计算 Y 的统计量和样本数 ===============
    bins = np.linspace(np.min(rasX_flat), np.max(rasX_flat), num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    y_means = []
    y_stds = []
    sample_counts = []

    for i in range(num_bins):
        bin_mask = (rasX_flat >= bins[i]) & (rasX_flat < bins[i+1])
        y_values_in_bin = rasY_flat[bin_mask]
        count = len(y_values_in_bin)
        sample_counts.append(count)
        if count > 0:
            y_means.append(np.mean(y_values_in_bin))
            y_stds.append(np.std(y_values_in_bin))
        else:
            y_means.append(np.nan)
            y_stds.append(np.nan)

    y_means = np.array(y_means, dtype=np.float32)
    y_stds = np.array(y_stds, dtype=np.float32)
    sample_counts = np.array(sample_counts, dtype=np.float32)

    # =============== 平滑曲线 (三次样条插值) ===============
    valid_mask = ~np.isnan(y_means)
    if valid_mask.sum() > 2:  # 至少需要若干点进行插值
        spl = make_interp_spline(bin_centers[valid_mask], y_means[valid_mask], k=3)
        x_smooth = np.linspace(np.min(bin_centers[valid_mask]), np.max(bin_centers[valid_mask]), 500)
        y_smooth = spl(x_smooth)
    else:
        # 数据点不足以插值
        x_smooth = bin_centers
        y_smooth = y_means

    # 构造布林带
    upper_band = y_means + std_factor * y_stds
    lower_band = y_means - std_factor * y_stds
    if valid_mask.sum() > 2:
        spl_upper = make_interp_spline(bin_centers[valid_mask], upper_band[valid_mask], k=3)
        spl_lower = make_interp_spline(bin_centers[valid_mask], lower_band[valid_mask], k=3)
        upper_band_smooth = spl_upper(x_smooth)
        lower_band_smooth = spl_lower(x_smooth)
    else:
        upper_band_smooth = upper_band
        lower_band_smooth = lower_band

    # =============== SMA计算 ===============
    # 将y_means转换为pandas.Series以方便rolling
    y_series = pd.Series(y_means)
    sma_short = y_series.rolling(window=sma_short_bins, min_periods=1).mean()
    sma_long  = y_series.rolling(window=sma_long_bins, min_periods=1).mean()

    # =============== MACD 计算 ===============
    def calculate_macd(values, short_p, long_p, signal_p):
        vals = pd.Series(values, dtype=np.float32)
        ema_short = vals.ewm(span=short_p, adjust=False).mean()
        ema_long  = vals.ewm(span=long_p, adjust=False).mean()
        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=signal_p, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    macd_line, signal_line, histogram = calculate_macd(y_means, macd_short_period, macd_long_period, macd_signal_period)

    # =============== 动态阈值划分风险（绝对偏离；与 SM1 S1-13 对齐） ===============
    # 阈值 τ：仍按当前实现，用 MACD 直方图的加权均值的绝对值
    valid_hist_mask = ~np.isnan(histogram)
    weighted_sum   = np.nansum(histogram[valid_hist_mask] * sample_counts[valid_hist_mask])
    total_weights  = np.nansum(sample_counts[valid_hist_mask])
    hist_mean      = (weighted_sum / total_weights) if total_weights > 0 else 0.0
    threshold      = abs(hist_mean)  # τ >= 0
    
    # 用“绝对偏离”分级，保证与 Risk_Value = |MACD - Signal| 单调一致
    delta        = macd_line - signal_line
    abs_delta    = np.abs(delta)
    
    risk_levels = []
    for ad, m, s in zip(abs_delta, macd_line, signal_line):
        if np.isnan(ad) or np.isnan(m) or np.isnan(s):
            risk_levels.append('No Data')
        elif ad >= (k_factor * threshold):
            risk_levels.append('High Risk')      # 3
        elif ad > threshold:
            risk_levels.append('Medium Risk')    # 2
        else:
            risk_levels.append('Low Risk')       # 1


    # =============== 将结果汇总为DataFrame ===============
    risk_df = pd.DataFrame({
        'bin_center': bin_centers,
        'Mean': y_means,
        'Std': y_stds,
        'Sample_Count': sample_counts,
        'SMA_Short': sma_short,
        'SMA_Long': sma_long,
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': histogram,
        'Risk': risk_levels
    })

    # =============== 计算 Risk = |MACD - Signal| ===============
    risk_df['Risk_Value'] = np.abs(risk_df['MACD'] - risk_df['Signal'])

    # =============== 映射 Risk 等级到整数代码 ===============
    # 定义风险等级与整数代码的映射
    risk_mapping = {
        'Low Risk': 1,
        'Medium Risk': 2,
        'High Risk': 3,
        'No Data': 0
    }

    # 创建 Risk_Level 列
    risk_df['Risk_Level'] = risk_df['Risk'].map(risk_mapping)

    # =============== 将 Risk_Value 映射回栅格数据 ===============
    # 创建一个空的 Risk_Value 栅格
    rasRisk_value = np.full(rasX_values.shape, -9999, dtype=np.float32)  # -9999 作为 nodata

    # 使用 numpy.digitize 确定每个栅格点的 bin
    bin_indices = np.digitize(rasX_flat, bins) - 1  # digitize 返回的是 bin 索引从1开始

    # 确保 bin_indices 在有效范围内
    bin_indices[bin_indices < 0] = 0
    bin_indices[bin_indices >= num_bins] = num_bins - 1

    # 获取每个栅格点的 Risk_Value
    rasRisk_value_flat = risk_df['Risk_Value'].values[bin_indices]

    # 将 Risk_Value 填充回栅格
    rasRisk_value[mask] = rasRisk_value_flat

    # =============== 将 Risk_Level 映射回栅格数据 ===============
    # 创建一个数组来存储 Risk_Level 的值
    rasRisk_level = np.full(rasX_values.shape, 0, dtype=np.int8)  # 0 为 No Data

    # 获取每个栅格点的 Risk_Level
    rasRisk_level_flat = risk_df['Risk_Level'].values[bin_indices]

    # 将 Risk_Level 填充回栅格
    rasRisk_level[mask] = rasRisk_level_flat

    # =============== 保存 Risk_Value 栅格为 .tif 文件 ===============
    # 使用 rasterio 写入 Risk_Value 栅格文件
    with rasterio.open(rasX_path) as src:
        meta_value = src.meta.copy()

    meta_value.update({
        'dtype': 'float32',
        'count': 1,
        'nodata': -9999,
        'compress': 'lzw'
    })

    with rasterio.open(output_risk_value_tif_path, 'w', **meta_value) as dst:
        dst.write(rasRisk_value, 1)

    # =============== 保存 Risk_Level 栅格为 .tif 文件 ===============
    # 使用 rasterio 写入 Risk_Level 栅格文件
    with rasterio.open(rasX_path) as src:
        meta_level = src.meta.copy()

    meta_level.update({
        'dtype': 'int8',
        'count': 1,
        'nodata': 0,  # 0 表示 No Data
        'compress': 'lzw'
    })

    with rasterio.open(output_risk_level_tif_path, 'w', **meta_level) as dst:
        dst.write(rasRisk_level, 1)

    # =============== 绘图 ===============

    # --- 版式参数校验（允许同时包含 'large' 与 'small'）---
    _layout = layout if 'layout' in locals() else ["large"]
    if _layout is None:
        _layout = ["large"]
    if isinstance(_layout, str):
        _layout = [_layout]

    if not isinstance(_layout, (list, tuple)):
        raise ValueError(
            "参数 layout 必须是列表或元组，例如 ['large'] 或 ['small']。\n"
            "The 'layout' parameter must be a list or tuple, e.g., ['large'] or ['small']."
        )
    valid_flags = {'small', 'large'}
    unknown = [x for x in _layout if x not in valid_flags]
    if unknown:
        raise ValueError(
            f"layout 含未识别选项：{unknown}。可选：'small' 或 'large'。\n"
            f"Unrecognized layout option(s): {unknown}. Allowed values are 'small' or 'large'."
        )
    # 固定渲染顺序，避免输出顺序不稳定
    modes = [m for m in ('large', 'small') if m in _layout]
    if not modes:
        modes = ['large']

    # 逐一渲染所需版式；注意：函数将返回“最后一张”图对象 fig（保持原有返回签名）
    for _mode in modes:
        is_small = (_mode == 'small')
        is_large = not is_small

        # ——版式尺寸与字体——
        if is_small:
            # A4 竖向宽度 ~8.27 in；小尺寸取一半宽度，便于两图并排拼接
            A4_WIDTH_IN = 8.27
            fig_w = A4_WIDTH_IN / 2.0       # ~4.135 in
            fig_h = fig_w * 0.62            # ~2.56 in，紧凑纵横比
            label_fs = 8
            tick_fs  = 7
            legend_fs = 7
            show_title = False               # 小尺寸去标题
            rotate_ticks = True              # 刻度竖排
        else:
            # 保持原始 1080p 风格
            fig_w, fig_h = 19.2, 10.8
            label_fs = 14
            tick_fs  = None                  # 沿用 matplotlib 默认
            legend_fs = 12
            show_title = True
            rotate_ticks = False

        # --- 建图 ---
        fig, ax1 = plt.subplots(figsize=(fig_w, fig_h))  # 尺寸随版式变化

        # 平滑均值曲线
        ax1.plot(x_smooth, y_smooth, label="Mean", color='blue', lw=2)

        # 布林带
        if len(x_smooth) == len(upper_band_smooth):
            ax1.fill_between(
                x_smooth, lower_band_smooth, upper_band_smooth,
                color='gray', alpha=0.3, label=f"Mean ±{std_factor}σ"
            )

        # SMA
        ax1.plot(
            bin_centers, sma_short,
            label=f"SMA {sma_short_bins}-Bin", color='orange', lw=2, linestyle='--'
        )
        ax1.plot(
            bin_centers, sma_long,
            label=f"SMA {sma_long_bins}-Bin", color='purple', lw=2, linestyle='--'
        )

        # 标题（小尺寸不显示）
        if show_title:
            ax1.set_title(
                f"Advanced Risk Detector\nX vs Y with {num_bins} Bins\n"
                f"(std_factor={std_factor}, k_factor={k_factor})",
                fontsize=max(label_fs, 16)
            )

        # 坐标轴标签
        ax1.set_xlabel("X (explanatory variable)", fontsize=label_fs)
        ax1.set_ylabel("Y (dependent variable)", fontsize=label_fs)

        # small：减少主刻度数量（更简洁）
        if is_small:
            ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))   # X ≤ 5 主刻度
            ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))   # 左 Y ≤ 4 主刻度

        # 左侧图例（仅 large 在轴内显示）
        if is_large:
            ax1.legend(loc='upper left', fontsize=legend_fs)

        # 第二个y轴，绘制 MACD
        ax2 = ax1.twinx()
        ax2.plot(bin_centers, macd_line,   label="MACD",   color='green', lw=2)
        ax2.plot(bin_centers, signal_line, label="Signal", color='red',   lw=2)
        bar_width = (bins[1] - bins[0]) * 0.6 if len(bins) > 1 else 0.01
        ax2.bar(bin_centers, histogram, width=bar_width, alpha=0.3, color='gray', label="MACD Histogram")
        ax2.set_ylabel("MACD", fontsize=label_fs)

        # ——Y 轴自动科学计数法（左右轴都开启；当量级足够小/大时自动切换）
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)


        # 右轴少刻度（仅 small）
        if is_small:
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
        # 右侧图例（仅 large 在轴内显示）
        if is_large:
            ax2.legend(loc='upper right', fontsize=legend_fs)

        # 风险区间着色
        for i in range(num_bins):
            x_bin_center = bin_centers[i]
            risk = risk_levels[i]
            left_edge  = x_bin_center - (bins[1] - bins[0]) / 2
            right_edge = x_bin_center + (bins[1] - bins[0]) / 2
            if risk == 'High Risk':
                ax1.axvspan(left_edge, right_edge, color='red',   alpha=0.1)
            elif risk == 'Medium Risk':
                ax1.axvspan(left_edge, right_edge, color='yellow', alpha=0.1)
            elif risk == 'Low Risk':
                ax1.axvspan(left_edge, right_edge, color='green',  alpha=0.1)
            else:
                pass  # 'No Data'

        # 风险图例
        risk_patches = [
            Patch(facecolor='red',    edgecolor='red',    alpha=0.1, label='High Risk'),
            Patch(facecolor='yellow', edgecolor='yellow', alpha=0.1, label='Medium Risk'),
            Patch(facecolor='green',  edgecolor='green',  alpha=0.1, label='Low Risk')
        ]
        current_handles, current_labels = ax1.get_legend_handles_labels()
        if is_large:
            ax1.legend(handles=current_handles + risk_patches, loc='upper left', fontsize=legend_fs)

        # ——small：刻度竖排 & 统一刻度字号 + 底部双行图例——
        if is_small:
            if tick_fs is not None:
                ax1.tick_params(axis='both', labelsize=tick_fs)
                ax2.tick_params(axis='y',    labelsize=tick_fs)
            for t in ax1.get_xticklabels():
                #t.set_rotation(90)
                t.set_verticalalignment('center')
                t.set_horizontalalignment('center')
            for t in ax1.get_yticklabels():
                t.set_rotation(90)
                t.set_verticalalignment('center')
                t.set_horizontalalignment('center')
            for t in ax2.get_yticklabels():
                t.set_rotation(90)
                t.set_verticalalignment('center')
                t.set_horizontalalignment('center')

            # ——small：底部双行图例（在 x 轴标签的下一行）
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            all_handles = h1 + h2 + risk_patches
            all_labels  = l1 + l2 + [p.get_label() for p in risk_patches]

            # 为底部图例预留空间（rect 的下边距增大）
            fig.tight_layout(rect=[0, 0.36, 1, 1])
            fig.legend(
                all_handles, all_labels,
                loc='lower center', bbox_to_anchor=(0.5, 0.04),
                ncol=4, fontsize=legend_fs, frameon=True
            )

        else:
            # large：正常紧凑布局
            plt.tight_layout()

    # 注意：若 layout 同时包含 ["large", "small"]，本函数将返回最后一张图对象（例如 "small"）
    return risk_df, fig








# ================= 使用示例 =================
if __name__ == "__main__":
    # 定义输出路径
    outpath = "G:/Yangtze vulnerability/myana/output_ngd_opencl/out adv"
    os.makedirs(outpath, exist_ok=True)  # 确保输出目录存在
    
    #set output
    output_risk_level_tif = os.path.join(outpath, "Risk_Level.tif")
    output_risk_value_tif = os.path.join(outpath, "Risk_Value.tif")
    out_csv = os.path.join(outpath, "advanced_risk_results.csv")

    # 栅格示例路径（请替换为您实际的文件路径）
    rasX = r"C:\Users\yuan wang\OneDrive\geodect\XY without reclass resample\Aspect.tif"
    rasY = r"C:\Users\yuan wang\OneDrive\geodect\XY without reclass resample\EVI_mytest2.tif"
    
    # 调用函数，使用默认参数
    risk_results, fig = advanced_risk_detector(
        rasX,
        rasY,
        output_risk_level_tif,
        output_risk_value_tif
    )
    
    # 如果需要自定义参数，比如k_factor=2, num_bins=50：
    # risk_results, fig = advanced_risk_detector(rasX, rasY, output_risk_level_tif, output_risk_value_tif, num_bins=50, k_factor=2.0)

    # 保存为1080p高清PNG和SVG矢量图
    fig.savefig(os.path.join(outpath, "advanced_risk_1080p.png"), dpi=100)  # 19.2 x 10.8 inches * 100 dpi = 1920x1080
    fig.savefig(os.path.join(outpath, "advanced_risk.svg"), format="svg")
    risk_results.to_csv(out_csv, index=False)
    
    # 显示图形
    plt.show()
    
    # risk_results DataFrame 包含每个bin的相关统计信息与风险分类，可自行保存
    # risk_results.to_csv(os.path.join(outpath, "advanced_risk_results.csv"), index=False)
