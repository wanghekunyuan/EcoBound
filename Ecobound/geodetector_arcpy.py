#(V 3.20 updated)
import os
import arcpy
from arcpy.sa import *
import numpy as np
import scipy.stats as stats
import pandas as pd
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Geodetector:
    def __init__(self, raster_list, y_path, mode='qmax', alpha=0.05):
        self.raster_list = raster_list
        self.y_path = y_path
        self.mode = mode  # 过滤模式参数 ('qmax' 或 'all')
        self.alpha = alpha  # 显著性水平
        self.y_raster = arcpy.Raster(y_path)
        self.y_array, self.y_mask = self.convert_raster_to_numpy(self.y_raster, raster_name='Y_Raster')
        self.q_results = []
        self.filtered_q_results = pd.DataFrame()  # 初始化为空的 DataFrame
        self.interaction_results = []
        self.risk_results = []
        self.ecological_results = []
        self.q_cache = {}  # Q 结果缓存
        self.raster_statistics_cache = {}  # 栅格统计缓存
        self.overall_variance, self.total_count = self.calculate_overall_statistics()  # 计算总体统计数据
        
        # V 3.20 updated log: add this line
        self.q_sample_stats = {}  # per-X sample stats: {cache_key: (var_y_given_X, N_X_intersect_Y)}

    def convert_raster_to_numpy(self, raster, raster_name=None):
        """
        将栅格转换为 NumPy 数组，并处理 NoData 值。

        Parameters:
            raster (arcpy.Raster): 要转换的栅格对象。
            raster_name (str): 栅格名称，用于日志记录。

        Returns:
            tuple: (NumPy 数组, 掩蔽数组)
        """
        desc = arcpy.Describe(raster)
        pixel_type = desc.pixelType
        if pixel_type.startswith('F'):
            nodata_value = np.nan
        else:
            nodata = raster.noDataValue
            if nodata is not None:
                nodata_value = nodata
            else:
                logging.warning(f"栅格 {raster_name or 'Unknown'} 没有定义 NoData 值，假设所有数据有效。")
                nodata_value = None

        try:
            array = arcpy.RasterToNumPyArray(raster, nodata_to_value=nodata_value).astype(float)
        except Exception as e:
            logging.error(f"无法将栅格 {raster_name or 'Unknown'} 转换为 NumPy 数组: {e}")
            return None, None

        if np.isnan(nodata_value):
            mask = ~np.isnan(array)
        elif nodata_value is not None:
            mask = array != nodata_value
            array[array == nodata_value] = np.nan  # 将 NoData 值设置为 NaN 以便后续处理
        else:
            mask = np.ones_like(array, dtype=bool)

        return array, mask

    def process_raster_data(self, x_raster, x_raster_path, cache_key):
        """
        处理栅格数据，提取唯一值及对应的 Y 值。

        Parameters:
            x_raster (arcpy.Raster): 自变量栅格。
            x_raster_path (str): 栅格路径，用于日志记录。
            cache_key (str): 缓存键，用于命名交互栅格。

        Returns:
            tuple: (唯一值列表, Y 值数组, X 值数组)
        """
        if x_raster_path is not None:
            raster_name = os.path.basename(x_raster_path)
        else:
            raster_name = cache_key  # 使用 cache_key 作为 raster_name

        logging.info(f"Processing raster: {raster_name}")
        array_x, mask_x = self.convert_raster_to_numpy(x_raster, raster_name=raster_name)
        if array_x is None:
            return None, None, None

        # 使用 x_raster 的掩膜来筛选 y_array
        y_values = self.y_array[mask_x]
        x_values = array_x[mask_x]

        # 获取唯一类别
        unique_values = np.unique(x_values)
        unique_values = unique_values[~np.isnan(unique_values)]
        unique_values = sorted(unique_values)
        num_strata = len(unique_values)

        if num_strata == 0:
            logging.warning(f"栅格 {raster_name} 没有有效的唯一值。")
            return None, None, None

        return unique_values, y_values, x_values

    def get_raster_statistics(self, x_raster_path=None, x_raster=None, cache_key=None):
        if cache_key is None:
            cache_key = x_raster_path
        if cache_key in self.raster_statistics_cache:
            return self.raster_statistics_cache[cache_key]

        arcpy.env.overwriteOutput = True

        if x_raster is None:
            try:
                x_raster = arcpy.Raster(x_raster_path)
            except Exception as e:
                logging.error(f"无法加载栅格 {x_raster_path}: {e}")
                return None, None, None

        unique_values, y_values, x_values = self.process_raster_data(x_raster, x_raster_path, cache_key)

        if unique_values is None:
            logging.warning(f"栅格 {x_raster_path} 缺少有效数据，跳过。")
            return None, None, None

        # 缓存结果
        self.raster_statistics_cache[cache_key] = (unique_values, y_values, x_values)
        return unique_values, y_values, x_values

    def process_strata(self, unique_values, y_values, x_values):
        variances = []
        means = []
        pixel_counts = []

        for value in unique_values:
            # 创建类别掩码
            mask = x_values == value
            masked_y = y_values[mask]

            # 检查是否全为 NaN
            if np.all(np.isnan(masked_y)):
                logging.warning(f"类别值 {value} 对应的 Y 值全为 NoData，跳过。")
                continue

            # 计算均值和标准差
            mean = np.nanmean(masked_y)
            std_dev = np.nanstd(masked_y, ddof=1)  # ddof=1 表示使用样本标准差

            # 如果标准差为 NaN，表示只有一个有效值，将其设为 0
            if np.isnan(std_dev):
                std_dev = 0.0

            # 计算方差
            variance = std_dev ** 2

            # 计算有效像素数
            pixel_count = np.count_nonzero(~np.isnan(masked_y))

            # 将结果添加到列表中
            variances.append(variance)
            means.append(mean)
            pixel_counts.append(pixel_count)

            # 记录每个分层的详细信息
            logging.debug(f"分层值: {value}, 均值: {mean}, 方差: {variance}, 像素数: {pixel_count}")

        # 检查是否有有效的值
        if not variances or not means or not pixel_counts:
            logging.warning("此层级缺少有效数据，返回 None。")
            return None, None, None

        return variances, means, pixel_counts

    def calculate_overall_statistics(self):
        if self.y_array is None:
            logging.error("Y 栅格数据未正确加载。")
            return 0, 0

        y_std_dev = np.nanstd(self.y_array, ddof=1)
        if np.isnan(y_std_dev):
            y_variance = 0.0
        else:
            y_variance = y_std_dev ** 2

        y_count = np.count_nonzero(~np.isnan(self.y_array))

        return y_variance, y_count

    def calculate_q_statistic(self, variances, pixel_counts, total_count, overall_variance):
        """
        计算 Q 统计量，分母采用“当前样本集”的 N 与 Var(Y)：
            q = 1 - SSW / (N * Var(Y | current sample))
        """
        # 组内平方和
        SSW = sum(N_h * var_h for N_h, var_h in zip(pixel_counts, variances))

        denom = total_count * overall_variance
        if denom == 0 or np.isnan(denom):
            return 0  # 避免除以零/无效
        q = 1 - SSW / denom
        return q


    def calculate_q(self, x_path=None, x_raster=None, cache_key=None):
        if cache_key is None:
            cache_key = x_path

        if cache_key in self.q_cache:
            logging.info(f"使用缓存的 Q 值: {cache_key}")
            return self.q_cache[cache_key]

        if x_raster is None and x_path is not None:
            try:
                x_raster = arcpy.Raster(x_path)
            except Exception as e:
                logging.error(f"无法加载栅格 {x_path}: {e}")
                return None

        unique_values, y_values, x_values = self.get_raster_statistics(
            x_raster_path=x_path, x_raster=x_raster, cache_key=cache_key
        )
        if unique_values is None:
            logging.warning(f"栅格 {x_path} 缺少有效数据，跳过。")
            return None

        # --- 核心升级：基于“当前样本 X∩Y”计算 Var(Y) 与 N ---
        y_std = np.nanstd(y_values, ddof=1)
        y_var_sample = 0.0 if np.isnan(y_std) else (y_std ** 2)
        N_sample = np.count_nonzero(~np.isnan(y_values))

        variances, means, pixel_counts = self.process_strata(unique_values, y_values, x_values)
        if variances is None or means is None or pixel_counts is None:
            logging.warning(f"栅格 {x_path} 的层级缺少有效数据，跳过。")
            return None

        q = self.calculate_q_statistic(
            variances, pixel_counts,
            total_count=N_sample,
            overall_variance=y_var_sample
        )

        result = (q, unique_values, means, variances, pixel_counts)
        self.q_cache[cache_key] = result

        # 记下该 X 的样本口径 Var/N，供后续输出和显著性计算使用
        self.q_sample_stats[cache_key] = (y_var_sample, N_sample)

        logging.info(f"Calculated Q for {cache_key}: Q={q}")
        return result


    def calculate_q_for_list(self):
        """串行计算所有栅格的 Q 值，无需过滤。"""
        logging.info("开始串行计算 Q 统计量...")

        for x_path in self.raster_list:
            logging.info(f"正在处理 {x_path}...")
            result = self.calculate_q(x_path=x_path)
            if result is None:
                logging.warning(f"栅格 {x_path} 的 Q 统计量计算失败，跳过。")
                continue

            q, unique_values, means, variances, pixel_counts = result

            # 使用“当前样本口径”的 N 与 Var(Y)（来自第3步缓存）
            y_var_sample, N_sample = self.q_sample_stats.get(
                x_path, (self.overall_variance, self.total_count)
            )

            # 显著性也基于本次样本大小 N_sample（其余保持现状）
            lambda_value, f_critical, is_significant = self.calculate_q_significance(
                q, N=N_sample
            )

            self.q_results.append({
                'Raster': x_path,
                'Q_statistic': q,
                'Lambda': lambda_value,
                'F_critical': f_critical,
                'Significant': is_significant,
                'Unique_values': unique_values,
                'Means': means,
                'Variances': variances,
                'Pixel_counts': pixel_counts,
                # >>> 输出口径升级：样本口径 <<<
                'Overall_variance': y_var_sample,
                'Total_count': N_sample
            })

            logging.info(f"栅格: {x_path} - Q 统计量: {q}, 显著: {is_significant}")

        logging.info("完成 Q 统计量的串行计算。")


    def filter_q_results(self):
        """根据模式过滤 Q 结果，仅在 qmax 模式下。"""
        if not self.q_results:
            logging.warning("Q 结果为空，无法进行过滤。")
            self.filtered_q_results = pd.DataFrame()
            return

        q_results_df = pd.DataFrame(self.q_results)
        if 'Significant' not in q_results_df.columns:
            logging.error("'Significant' 列不存在于 Q 结果中。")
            self.filtered_q_results = pd.DataFrame()
            return

        if self.mode == 'qmax':
            q_results_df['Base_X'] = q_results_df['Raster'].apply(lambda x: os.path.basename(x).split('_')[0])
            filtered = q_results_df[q_results_df['Significant'] == True].groupby('Base_X').apply(lambda df: df.loc[df['Q_statistic'].idxmax()]).reset_index(drop=True)
            self.filtered_q_results = filtered
            logging.info(f"在 qmax 模式下，过滤后得到 {len(self.filtered_q_results)} 个结果。")
        else:
            filtered = q_results_df[q_results_df['Significant'] == True]
            self.filtered_q_results = filtered
            logging.info(f"在 'all' 模式下，过滤后得到 {len(self.filtered_q_results)} 个结果。")

    def calculate_q_significance(self, q, alpha=None, N=None, L=None):
        """
        计算 q 的显著性。为尽量少改动：
        - 若传入 N，则使用该 N（样本口径）；否则退回旧逻辑 self.total_count。
        - L 默认仍按旧逻辑使用 len(self.raster_list)（不在此次补丁范围内改动）。
        """
        if alpha is None:
            alpha = self.alpha
        if N is None:
            N = self.total_count
        if L is None:
            L = len(self.raster_list)

        if (1 - q) == 0:
            lambda_value = np.inf
        else:
            lambda_value = q * (N - 1) / (1 - q)

        try:
            f_critical = stats.f.ppf(1 - alpha, L - 1, N - L)
        except Exception as e:
            logging.error(f"计算 F 临界值时出错: {e}")
            f_critical = np.inf  # 避免因自由度不合法导致错误

        is_significant = lambda_value > f_critical
        return lambda_value, f_critical, is_significant


    def create_intersection_raster(self, x1_raster, x2_raster):
        try:
            intersection_raster = Combine([x1_raster, x2_raster])
            logging.info(f"Created intersection raster: {intersection_raster}")
            return intersection_raster
        except Exception as e:
            logging.error(f"创建交集栅格时出错: {e}")
            return None

    def evaluate_interaction(self, q_x1, q_x2, q_intersection):
        if q_intersection < min(q_x1, q_x2):
            interaction_type = "Nonlinear weakening"
        elif min(q_x1, q_x2) <= q_intersection <= max(q_x1, q_x2):
            interaction_type = "Single factor nonlinear weakening"
        elif (q_x1 + q_x2) > q_intersection > max(q_x1, q_x2):
            interaction_type = "Bifactor enhancement"
        elif q_intersection == q_x1 + q_x2:
            interaction_type = "Independent"
        elif q_intersection > q_x1 + q_x2:
            interaction_type = "Nonlinear enhancement"
        else:
            interaction_type = "Unknown"

        return interaction_type

    def detection_of_interaction(self):
        """进行交互检测，分析因子之间的相互影响。"""
        if self.mode == 'qmax':
            raster_list = self.filtered_q_results['Raster'].tolist()
            logging.info("使用过滤后的变量进行交互检测...")
        else:
            raster_list = [result['Raster'] for result in self.q_results]
            logging.info("使用所有变量进行交互检测...")

        for i in range(len(raster_list)):
            for j in range(i + 1, len(raster_list)):
                x1_path = raster_list[i]
                x2_path = raster_list[j]

                result1 = self.calculate_q(x_path=x1_path)
                result2 = self.calculate_q(x_path=x2_path)

                if result1 is None or result2 is None:
                    logging.warning(f"无法计算 {x1_path} 或 {x2_path} 的 Q 统计量，跳过交互检测。")
                    continue

                q_x1, _, _, _, _ = result1
                q_x2, _, _, _, _ = result2

                # 创建排序后的键以防止冲突
                sorted_paths = sorted([x1_path, x2_path])
                intersection_key = f"{os.path.basename(sorted_paths[0])}_{os.path.basename(sorted_paths[1])}"

                if intersection_key in self.q_cache:
                    result_intersection = self.q_cache[intersection_key]
                else:
                    try:
                        x1_raster = arcpy.Raster(x1_path)
                        x2_raster = arcpy.Raster(x2_path)
                        intersection_raster = self.create_intersection_raster(x1_raster, x2_raster)
                        if intersection_raster is None:
                            logging.warning(f"无法创建交集栅格 {intersection_key}，跳过。")
                            continue
                        result_intersection = self.calculate_q(x_raster=intersection_raster, cache_key=intersection_key)
                        self.q_cache[intersection_key] = result_intersection
                    except Exception as e:
                        logging.error(f"无法创建或处理交互栅格 {intersection_key}: {e}")
                        continue

                if result_intersection is None:
                    logging.warning(f"无法计算 {x1_path} 和 {x2_path} 的交集 Q 统计量，跳过。")
                    continue

                q_intersection, _, _, _, _ = result_intersection

                interaction_type = self.evaluate_interaction(q_x1, q_x2, q_intersection)

                self.interaction_results.append({
                    'X1': x1_path,
                    'X2': x2_path,
                    'q_X1': q_x1,
                    'q_X2': q_x2,
                    'q_X1_and_X2': q_intersection,
                    'Interaction': interaction_type
                })

                logging.info(f"{x1_path} 与 {x2_path} 的交互检测结果:")
                logging.info(f"q(X1) = {q_x1}")
                logging.info(f"q(X2) = {q_x2}")
                logging.info(f"q(X1 ∩ X2) = {q_intersection}")
                logging.info(f"交互类型: {interaction_type}")

    def risk_detection(self):
        """进行风险检测，分析因子对属性Y的风险影响。"""
        if self.mode == 'qmax':
            raster_list = self.filtered_q_results['Raster'].tolist()
            logging.info("使用过滤后的变量进行风险检测...")
        else:
            raster_list = [result['Raster'] for result in self.q_results]
            logging.info("使用所有变量进行风险检测...")

        for x_path in raster_list:
            result = self.calculate_q(x_path=x_path)
            if result is None:
                continue
            _, unique_values, means, variances, pixel_counts = result

            for value, mean, variance, pixel_count in zip(unique_values, means, variances, pixel_counts):
                if pixel_count <= 1:
                    continue

                self.risk_results.append({
                    'X_raster': x_path,
                    'Value': value,
                    'Mean': mean,
                    'Variance': variance,
                    'Count': pixel_count
                })

        if len(self.risk_results) >= 2:
            for i in range(len(self.risk_results)):
                for j in range(i + 1, len(self.risk_results)):
                    result1 = self.risk_results[i]
                    result2 = self.risk_results[j]

                    mean1 = result1['Mean']
                    var1 = result1['Variance']
                    n1 = result1['Count']

                    mean2 = result2['Mean']
                    var2 = result2['Variance']
                    n2 = result2['Count']

                    # 计算 t 值
                    denominator = np.sqrt(var1 / n1 + var2 / n2)
                    if denominator == 0:
                        logging.warning(f"分母为零，无法比较值 {result1['Value']} 和 {result2['Value']}，跳过。")
                        continue
                    t_value = (mean1 - mean2) / denominator

                    # 计算自由度
                    numerator = (var1 / n1 + var2 / n2) ** 2
                    denominator_df = ((var1 / n1) ** 2) / (n1 - 1) + ((var2 / n2) ** 2) / (n2 - 1)
                    if denominator_df == 0:
                        logging.warning(f"自由度计算为零，无法比较值 {result1['Value']} 和 {result2['Value']}，跳过。")
                        continue
                    df = numerator / denominator_df

                    # 计算 p 值
                    p_value = stats.t.sf(np.abs(t_value), df) * 2  # 双尾检验

                    result1[f'Comparison_with_{result2["Value"]}_p_value'] = p_value
                    result2[f'Comparison_with_{result1["Value"]}_p_value'] = p_value

                    result1[f'Comparison_with_{result2["Value"]}_significant'] = p_value < self.alpha
                    result2[f'Comparison_with_{result1["Value"]}_significant'] = p_value < self.alpha

    def ecological_detection(self):
        """进行生态检测，比较两个因子X1和X2对属性Y的空间分布影响是否有显著差异。"""
        if self.mode == 'qmax':
            raster_list = self.filtered_q_results['Raster'].tolist()
            logging.info("使用过滤后的变量进行生态检测...")
        else:
            raster_list = [result['Raster'] for result in self.q_results]
            logging.info("使用所有变量进行生态检测...")

        for i in range(len(raster_list)):
            for j in range(i + 1, len(raster_list)):
                x1_path = raster_list[i]
                x2_path = raster_list[j]

                result1 = self.calculate_q(x_path=x1_path)
                result2 = self.calculate_q(x_path=x2_path)

                if result1 is None or result2 is None:
                    logging.warning(f"无法计算 {x1_path} 或 {x2_path} 的 Q 统计量，跳过生态检测。")
                    continue

                # 正确提取 variances 和 pixel_counts
                variances1 = result1[3]
                pixel_counts1 = result1[4]
                variances2 = result2[3]
                pixel_counts2 = result2[4]

                # 获取层级数量
                unique_values1 = result1[1]
                unique_values2 = result2[1]
                num_strata1 = len(unique_values1)
                num_strata2 = len(unique_values2)

                # 计算组内平方和（SSW）
                SSW_x1 = sum(N_h * var_h for N_h, var_h in zip(pixel_counts1, variances1))
                SSW_x2 = sum(N_h * var_h for N_h, var_h in zip(pixel_counts2, variances2))

                if SSW_x2 == 0:
                    logging.warning(f"SSW_x2 为零，无法进行 F 检验，跳过 {x1_path} 和 {x2_path} 的生态检测。")
                    continue

                try:
                    # 根据标准 F 统计量公式计算
                    MSW1 = SSW_x1 / (num_strata1 - 1) if (num_strata1 - 1) != 0 else 0
                    MSW2 = SSW_x2 / (num_strata2 - 1) if (num_strata2 - 1) != 0 else 0

                    # 避免除以零
                    if MSW2 == 0:
                        logging.warning(f"MSW2 为零，无法计算 F 值，跳过 {x1_path} 和 {x2_path} 的生态检测。")
                        continue

                    F = MSW1 / MSW2
                    df1 = num_strata1 - 1
                    df2 = num_strata2 - 1
                    p_value = 1 - stats.f.cdf(F, df1, df2)
                    significant = p_value < self.alpha
                except Exception as e:
                    logging.error(f"计算 F 检验时出错 ({x1_path}, {x2_path}): {e}")
                    continue

                self.ecological_results.append({
                    'X1_Raster': x1_path,
                    'X2_Raster': x2_path,
                    'F_value': F,
                    'Degrees_of_Freedom_1': df1,
                    'Degrees_of_Freedom_2': df2,
                    'P_value': p_value,
                    'Significant': significant
                })

                logging.info(f"{x1_path} 与 {x2_path} 的生态检测结果:")
                logging.info(f"F 值: {F}")
                logging.info(f"自由度 1: {df1}")
                logging.info(f"自由度 2: {df2}")
                logging.info(f"P 值: {p_value}")
                logging.info(f"显著: {significant}")

    def save_q_results(self, path):
        if not self.q_results:
            logging.warning("Q 结果为空，无法保存。")
            return
        df = pd.DataFrame(self.q_results)
        df.to_csv(path, index=False, encoding='utf-8')
        logging.info(f"已保存 Q 结果到 {path}")

    def save_filtered_q_results(self, path):
        """保存过滤后的 Q 结果。"""
        if self.filtered_q_results.empty:
            logging.warning("过滤后的 Q 结果为空，无法保存。")
            return
        self.filtered_q_results.to_csv(path, index=False, encoding='utf-8')
        logging.info(f"已保存过滤后的 Q 结果到 {path}")

    def save_interaction_results(self, path):
        if not self.interaction_results:
            logging.warning("交互检测结果为空，无法保存。")
            return
        df = pd.DataFrame(self.interaction_results)
        df.to_csv(path, index=False, encoding='utf-8')
        logging.info(f"已保存交互检测结果到 {path}")

    def save_risk_results(self, path):
        if not self.risk_results:
            logging.warning("风险检测结果为空，无法保存。")
            return
        df = pd.DataFrame(self.risk_results)
        df.to_csv(path, index=False, encoding='utf-8')
        logging.info(f"已保存风险检测结果到 {path}")

    def save_ecological_results(self, path):
        if not self.ecological_results:
            logging.warning("生态检测结果为空，无法保存。")
            return
        df = pd.DataFrame(self.ecological_results)
        df.to_csv(path, index=False, encoding='utf-8')
        logging.info(f"已保存生态检测结果到 {path}")


# 使用示例
if __name__ == "__main__":
    arcpy.env.overwriteOutput = True

    def main():
        arcpy.CheckOutExtension("Spatial")

        start_time = time.time()

        raster_list = [
            r"C:\Users\yuan wang\OneDrive\geodect\X\Aspect_Reclass.tif",
            r"C:\Users\yuan wang\OneDrive\geodect\X\DEM_Reclass.tif",
            r"C:\Users\yuan wang\OneDrive\geodect\X\FVC_Reclass_11.tif",
            r"C:\Users\yuan wang\OneDrive\geodect\X\GDP_Reclass.tif",
            r"C:\Users\yuan wang\OneDrive\geodect\X\LandCover_Reclass.tif",
            r"C:\Users\yuan wang\OneDrive\geodect\X\LST_Reclass_1.tif",
            r"C:\Users\yuan wang\OneDrive\geodect\X\PopDensity_Reclass.tif",
            r"C:\Users\yuan wang\OneDrive\geodect\X\Prep_Reclass.tif",
            r"C:\Users\yuan wang\OneDrive\geodect\X\RiverDensity_Reclass.tif",
            r"C:\Users\yuan wang\OneDrive\geodect\X\Slope_Reclass.tif",
            r"C:\Users\yuan wang\OneDrive\geodect\X\Soil_Reclass.tif"
        ]

        y_path = r"C:\Users\yuan wang\OneDrive\geodect\Y\EVI_mytest2.tif"
        output_dir = r"G:\Yangtze vulnerability\myana\output_arcpy"

        geodetector = Geodetector(raster_list, y_path, mode='all', alpha=0.05)

        # 执行串行 Q 统计量计算
        geodetector.calculate_q_for_list()
        geodetector.save_q_results(os.path.join(output_dir, "q_results.csv"))

        geodetector.filter_q_results()  # 应用过滤
        geodetector.save_filtered_q_results(os.path.join(output_dir, "filtered_q_results.csv"))

        # 交互分析
        geodetector.detection_of_interaction()
        geodetector.save_interaction_results(os.path.join(output_dir, "interaction_results.csv"))

        # 风险检测
        geodetector.risk_detection()
        geodetector.save_risk_results(os.path.join(output_dir, "risk_results.csv"))

        # 生态检测
        geodetector.ecological_detection()
        geodetector.save_ecological_results(os.path.join(output_dir, "ecological_results.csv"))

        arcpy.CheckInExtension("Spatial")

        end_time = time.time()
        logging.info(f"总执行时间: {end_time - start_time} 秒")

    main()
