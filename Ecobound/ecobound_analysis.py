import numpy as np
import rasterio
import matplotlib.pyplot as plt

def extract_raster(raster_path):
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
    return arr, nodata, transform, crs

def bin_stats(X, Y, C=100):
    bins = np.linspace(np.min(X), np.max(X), C + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    counts = np.zeros(C, dtype=int)
    Y_means = np.full(C, np.nan, dtype=float)
    for i in range(C):
        mask = (X >= bins[i]) & (X < bins[i + 1])
        counts[i] = mask.sum()
        if counts[i] > 0:
            Y_means[i] = np.mean(Y[mask])
    return centers, counts, Y_means

def compute_entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def entropy_gain_scan(R, counts, B=30):
    valid = (~np.isnan(R)) & (counts > 0)
    R_valid = R[valid]
    w_valid = counts[valid]
    C = len(R_valid)

    Rn = np.minimum(np.floor((R_valid - R_valid.min()) / (R_valid.ptp() + 1e-12) * B).astype(int), B - 1)

    nAb = np.zeros(B)
    for i in range(len(Rn)):
        nAb[Rn[i]] += w_valid[i]
    PAb = nAb / np.sum(nAb)
    Hall = compute_entropy(PAb)

    max_delta_H = -np.inf
    best_k = -1

    for k in range(1, C):
        nLb = np.zeros(B)
        nRb = np.zeros(B)
        for i in range(C):
            if i < k:
                nLb[Rn[i]] += w_valid[i]
            else:
                nRb[Rn[i]] += w_valid[i]
        PLb = nLb / np.sum(nLb) if np.sum(nLb) > 0 else np.zeros(B)
        PRb = nRb / np.sum(nRb) if np.sum(nRb) > 0 else np.zeros(B)
        HL = compute_entropy(PLb)
        HR = compute_entropy(PRb)
        delta_H = HL + HR - Hall
        if delta_H > max_delta_H:
            max_delta_H = delta_H
            best_k = k
    return best_k

def variance_reduction_threshold(X, Y, centers, k):
    T = centers[k]
    mask1 = X < T
    mask2 = ~mask1
    Y1, Y2 = Y[mask1], Y[mask2]
    n1, n2 = len(Y1), len(Y2)
    N = n1 + n2
    var_tot = np.var(Y, ddof=1)
    vr = 1 - (n1 * np.var(Y1, ddof=1) + n2 * np.var(Y2, ddof=1)) / (N * var_tot + 1e-12)
    return vr, T

def permutation_test_vr(rasX_path, rasY_path, best_k, C1=100, repeat=1000):
    """
    ΔV置换检验：用户只需输入 raster 路径 + best_k。
    自动完成栅格读取、掩膜、标准化、bin。
    """
    # 1. 提取并掩膜
    X_arr, ndX, _, _ = extract_raster(rasX_path)
    Y_arr, ndY, _, _ = extract_raster(rasY_path)
    mask = (X_arr != ndX) & (Y_arr != ndY) & ~np.isnan(X_arr) & ~np.isnan(Y_arr)
    X_flat = X_arr[mask]
    Y_flat = Y_arr[mask]
    Y_flat = (Y_flat - np.mean(Y_flat)) / (np.std(Y_flat) + 1e-12)

    # 2. 分箱统计
    centers, counts, Y_means = bin_stats(X_flat, Y_flat, C=C1)

    # 3. 获取观测 VR
    obs_vr, _ = variance_reduction_threshold(X_flat, Y_flat, centers, best_k)

    # 4. 置换检验
    vr_dist = []
    for i in range(repeat):
        Y_perm = np.random.permutation(Y_flat)
        vr_perm, _ = variance_reduction_threshold(X_flat, Y_perm, centers, best_k)
        vr_dist.append(vr_perm)
        if (i + 1) % 10 == 0 or i == repeat - 1:
            print(f"置换进度：{i + 1} / {repeat}", end="\r")

    vr_dist = np.array(vr_dist)
    p_val = np.sum(vr_dist >= obs_vr) / repeat
    print()
    return p_val, vr_dist



def run_ecobound_analysis(rasX_path, rasY_path, C1=30, B_bins=50):
    X_arr, ndX, _, _ = extract_raster(rasX_path)
    Y_arr, ndY, _, _ = extract_raster(rasY_path)
    mask = (X_arr != ndX) & (Y_arr != ndY) & ~np.isnan(X_arr) & ~np.isnan(Y_arr)
    rasX_flat = X_arr[mask]
    rasY_flat = Y_arr[mask]

    Y_mean = np.mean(rasY_flat)
    Y_std = np.std(rasY_flat)
    rasY_flat = (rasY_flat - Y_mean) / (Y_std + 1e-12)

    centers, counts, Y_means = bin_stats(rasX_flat, rasY_flat, C=C1)
    best_k = entropy_gain_scan(Y_means, counts, B=B_bins)
    vr_best, T_entropy = variance_reduction_threshold(rasX_flat, rasY_flat, centers, best_k)

    print(f"首要生态阈值 (Primary ecological threshold) (熵增扫描) T_entropy = {T_entropy:.6f}")
    print(f"对应方差削减率 (Variance reduction rate) VR = {vr_best:.4f}")
    print(f"最佳分箱索引 (Best bin index) k* = {best_k} / {C1-1}")
    print("有效像元数 (Valid pixels)：", mask.sum())
    print("X 范围 (Range of X)：", rasX_flat.min(), "→", rasX_flat.max())
    print("Y standlized 范围 (Range of standardized Y)：", rasY_flat.min(), "→", rasY_flat.max())

    plt.plot(centers, Y_means, marker='o')
    plt.axvline(T_entropy, color='r', linestyle='--', label=f'T = {T_entropy:.2f}')
    plt.title(f"Response curve (VR = {vr_best:.4f})")
    plt.xlabel("X")
    plt.ylabel("Y mean (standardized)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.show()

    return T_entropy, vr_best, best_k

# --- 类定义区 ---
class EcoBoundAnalyzer:

    def plot(self, save_path=None, show=False, dpi=300):
        """
        绘制 Y~X 响应曲线图，支持保存 SVG 或 JPG。
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(self.centers, self.Y_means, marker='o', lw=2)
        plt.axvline(self.T_entropy, color='red', linestyle='--', label=f"T = {self.T_entropy:.2f}")
        plt.xlabel("X value")
        plt.ylabel("Y (standardized)")
        plt.title("EcoBound Response Curve")
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=dpi)
        if show:
            plt.show()
        #plt.close()
    def __init__(self, rasX_path, rasY_path):
        self.rasX_path = rasX_path
        self.rasY_path = rasY_path

        # 加载并掩膜
        self.X_arr, self.ndX, _, _ = extract_raster(rasX_path)
        self.Y_arr, self.ndY, _, _ = extract_raster(rasY_path)

        self.mask = (self.X_arr != self.ndX) & (self.Y_arr != self.ndY) & \
                    ~np.isnan(self.X_arr) & ~np.isnan(self.Y_arr)

        self.X_flat = self.X_arr[self.mask]
        self.Y_flat_raw = self.Y_arr[self.mask]
        self.Y_flat = (self.Y_flat_raw - np.mean(self.Y_flat_raw)) / (np.std(self.Y_flat_raw) + 1e-12)

    def run_ecobound(self, C1=100, B_bins=30):
        self.centers, self.counts, self.Y_means = bin_stats(self.X_flat, self.Y_flat, C=C1)
        self.best_k = entropy_gain_scan(self.Y_means, self.counts, B=B_bins)
        self.vr_best, self.T_entropy = variance_reduction_threshold(self.X_flat, self.Y_flat, self.centers, self.best_k)

        print(f"首要生态阈值 (Primary ecological threshold) (熵增扫描) T_entropy = {self.T_entropy:.6f}")
        print(f"对应方差削减率 (Variance reduction rate) VR = {self.vr_best:.4f}")
        print(f"最佳分箱索引 (Best bin index) k* = {self.best_k} / {C1-1}")
        print("有效像元数 (Valid pixels)：", self.mask.sum())
        print("X 范围 (Range of X)：", self.X_flat.min(), "→", self.X_flat.max())
        print("Y standlized 范围 (Range of standardized Y)：", self.Y_flat.min(), "→", self.Y_flat.max())

        # 绘图
        plt.plot(self.centers, self.Y_means, marker='o')
        plt.axvline(self.T_entropy, color='r', linestyle='--', label=f'T = {self.T_entropy:.2f}')
        plt.title(f"Response curve (VR = {self.vr_best:.4f})")
        plt.xlabel("X")
        plt.ylabel("Y mean (standardized)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        #plt.show()

        return self.T_entropy, self.vr_best, self.best_k

    def run_permutation_test(self, repeat=1000):
        print(f"正在进行 ΔV 置换检验 (Permutation test, repeats = {repeat}) ...")
        obs_vr = self.vr_best
        vr_dist = []

        for i in range(repeat):
            Y_perm = np.random.permutation(self.Y_flat)
            vr_perm, _ = variance_reduction_threshold(self.X_flat, Y_perm, self.centers, self.best_k)
            vr_dist.append(vr_perm)
            if (i + 1) % 10 == 0 or i == repeat - 1:
                print(f"置换进度：{i + 1} / {repeat}", end="\r")

        vr_dist = np.array(vr_dist)
        p_val = np.sum(vr_dist >= obs_vr) / repeat
        print()
        
        if p_val == 0:
            print(f"ΔV 检验 p 值 (Permutation p-value): < {1/repeat:.4f} (N={repeat})")
        else:
            print(f"ΔV 检验 p 值 (Permutation p-value): {p_val:.4f} (N={repeat})")
        return p_val, vr_dist



if __name__ == "__main__":

    # 用户参数
    rasX_path = r"G:\SJY LEN\SJY Geodector\input_resample\temp.tif"
    rasX_path = r"G:\SJY LEN\SJY Geodector\input_resample\pre.tif"
    rasX_path = r"G:\SJY LEN\SJY Geodector\input_resample\demresample.tif"
    rasY_path = r"G:\SJY LEN\SJY Geodector\input_resample\Y7Resa.tif"

    C1 = 100     # 第一次分箱数
    B_bins = 30  # 第二次熵扫描分箱数

   
    # 创建类
    analyzer = EcoBoundAnalyzer(rasX_path, rasY_path)
    
    # 主流程
    T, VR, k = analyzer.run_ecobound(C1=100, B_bins=30)
    
    # 可选 ΔV 检验（无冗余计算）
    repeat = 100 #置换检验次数
    p_val, dist = analyzer.run_permutation_test(repeat)    
    
    
    
