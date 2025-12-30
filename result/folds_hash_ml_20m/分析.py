import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # 用于实时显示进程进度条

# ===================== 适配当前目录结构的路径定义 =====================
# 当前目录（python文件所在目录）
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDS_20M_DIR = os.path.join(ROOT_DIR, "folds_hash_20m")
# 自动创建目录（若不存在）
os.makedirs(FOLDS_20M_DIR, exist_ok=True)
# 子目录路径（超参数网格、秩扫描）
SWEEPS_DIR = os.path.join(FOLDS_20M_DIR, "sweeps")
os.makedirs(SWEEPS_DIR, exist_ok=True)
PARAM_GRID_DIR = os.path.join(SWEEPS_DIR, "param_grid_nc_mf_sgd")
os.makedirs(PARAM_GRID_DIR, exist_ok=True)
RANK_SWEEP_DIR = os.path.join(SWEEPS_DIR, "rank_sweep")
os.makedirs(RANK_SWEEP_DIR, exist_ok=True)


# ===================== 1. 分析数据统计（带进程显示，修正错别字） =====================
def analyze_fold_data():
    """分析fold0-fold4等分析文件的核心统计信息，实时显示进度"""
    print("\n【进程提示】开始执行：分析数据统计")
    # 筛选所有分析文件（fold*.txt）
    fold_files = [f for f in os.listdir(FOLDS_20M_DIR) if
                  f.startswith("fold") and f.endswith(".txt")] if os.path.exists(FOLDS_20M_DIR) else []
    if not fold_files:
        print("【警告】未找到分析文件（fold*.txt），跳过该步骤")
        return None

    fold_stats = []
    # 使用tqdm包装迭代器，显示实时进度
    for fold_file in tqdm(fold_files, desc="分析文件统计进度"):
        fold_path = os.path.join(FOLDS_20M_DIR, fold_file)
        try:
            # 读取分析文件（默认格式：user_id\titem_id\trating，无表头）
            data = pd.read_csv(
                fold_path,
                sep="\t",
                header=None,
                names=["user_id", "item_id", "rating"],
                encoding="utf-8"
            )
            # 统计核心信息
            fold_info = {
                "分析名称": fold_file.split(".")[0],
                "唯一用户数": data["user_id"].nunique(),
                "唯一物品数": data["item_id"].nunique(),
                "评分总条数": len(data),
                "平均评分": round(data["rating"].mean(), 4),
                "最低评分": data["rating"].min(),
                "最高评分": data["rating"].max()
            }
            fold_stats.append(fold_info)
        except Exception as e:
            print(f"【错误】处理文件{fold_file}失败：{e}")
            continue

    # 整理为DataFrame并保存
    fold_df = pd.DataFrame(fold_stats)
    print("\n【结果提示】分析数据统计完成，核心结果如下：")
    print(fold_df.to_string(index=False))
    # 保存统计结果到folds_hash_20m目录
    save_path = os.path.join(FOLDS_20M_DIR, "fold_data_stats.csv")
    fold_df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"【保存提示】分析统计结果已保存至：{save_path}")
    return fold_df


# ===================== 2. 算法收敛分析（history文件） =====================
def analyze_history_files():
    """分析所有history_*.json文件，统计算法收敛性与最优RMSE，实时显示进度"""
    print("\n【进程提示】开始执行：算法收敛分析")
    # 筛选所有历史记录文件
    history_files = [f for f in os.listdir(FOLDS_20M_DIR) if
                     f.startswith("history_") and f.endswith(".json")] if os.path.exists(FOLDS_20M_DIR) else []
    if not history_files:
        print("【警告】未找到历史记录文件（history_*.json），跳过该步骤")
        return None

    history_results = []
    # 实时进度条
    for hist_file in tqdm(history_files, desc="历史文件解析进度"):
        # 解析文件名：history_foldX_算法名.json
        file_parts = hist_file.split("_")
        if len(file_parts) < 3:
            print(f"【警告】文件名{hist_file}格式异常，跳过")
            continue
        fold_name = file_parts[1]
        algo_name = "_".join(file_parts[2:]).replace(".json", "")
        file_path = os.path.join(FOLDS_20M_DIR, hist_file)

        try:
            # 读取JSON文件
            with open(file_path, "r", encoding="utf-8") as f:
                hist_data = json.load(f)

            # 提取训练/验证RMSE
            train_rmse_list = hist_data.get("train_rmse", [np.nan])
            val_rmse_list = hist_data.get("val_rmse", [np.nan])

            # 计算最优值（排除NaN）
            valid_train_rmse = [x for x in train_rmse_list if not np.isnan(x)]
            valid_val_rmse = [x for x in val_rmse_list if not np.isnan(x)]

            best_train_rmse = round(min(valid_train_rmse), 6) if valid_train_rmse else np.nan
            best_val_rmse = round(min(valid_val_rmse), 6) if valid_val_rmse else np.nan
            best_val_epoch = np.nan  # 初始化最优轮数

            # 解决浮点数精度问题：通过枚举索引找最小值位置（不再用index()）
            if valid_val_rmse:
                min_val = min(valid_val_rmse)
                # 遍历列表，找到第一个等于最小值的索引
                for idx, val in enumerate(valid_val_rmse):
                    # 浮点数模糊匹配（允许微小精度差异）
                    if abs(val - min_val) < 1e-9:
                        best_val_epoch = idx + 1  # 轮数从1开始计数
                        break

            # 保存结果
            history_info = {
                "分析名称": fold_name,
                "算法名称": algo_name,
                "最优训练RMSE": best_train_rmse,
                "最优验证RMSE": best_val_rmse,
                "最优验证轮数": best_val_epoch,
                "训练轮数总数": len(train_rmse_list),
                "验证RMSE全记录": val_rmse_list
            }
            history_results.append(history_info)
        except Exception as e:
            print(f"【警告】处理文件{hist_file}失败：{e}")
            continue

    # 整理结果
    hist_df = pd.DataFrame(history_results)
    # 新增：判断hist_df是否为空，避免后续报错
    if hist_df.empty:
        print("\n【警告】未获取有效历史文件数据，跳过后续打印和保存")
        return None

    print("\n【结果提示】算法收敛分析完成，核心结果如下（按最优验证RMSE排序）：")
    display_cols = ["分析名称", "算法名称", "最优验证RMSE", "最优验证轮数", "训练轮数总数"]
    # 确保列存在（避免个别数据缺失导致的列错误）
    display_cols = [col for col in display_cols if col in hist_df.columns]
    print(hist_df[display_cols].sort_values("最优验证RMSE").to_string(index=False))

    # 保存结果
    save_path = os.path.join(FOLDS_20M_DIR, "algorithm_convergence_analysis.csv")
    hist_df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"【保存提示】算法收敛结果已保存至：{save_path}")

    # 绘制核心算法收敛曲线
    plot_core_algo_convergence(hist_df)
    return hist_df


# ===================== 辅助：绘制核心算法收敛曲线 =====================
def plot_core_algo_convergence(hist_df):
    """绘制核心算法（nc_mf_sgd）在各分析的收敛曲线"""
    print("\n【进程提示】开始绘制：核心算法收敛曲线")
    # 新增：判断hist_df是否为空
    if hist_df.empty:
        print("【警告】无有效数据，跳过绘图")
        return

    target_algo = "nc_mf_sgd"
    algo_data = hist_df[hist_df["算法名称"] == target_algo]
    if algo_data.empty:
        print(f"【警告】未找到{target_algo}算法数据，跳过绘图")
        return

    # 绘制曲线
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文乱码
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    plt.figure(figsize=(12, 6))

    # 遍历各分析绘制
    for _, row in algo_data.iterrows():
        val_rmse = row["验证RMSE全记录"]
        plt.plot(range(1, len(val_rmse) + 1), val_rmse, marker=".", linewidth=1.5, label=f"分析{row['分析名称']}")

    plt.title(f"{target_algo} 算法各分析验证RMSE收敛曲线", fontsize=14)
    plt.xlabel("训练轮数（Epoch）", fontsize=12)
    plt.ylabel("验证RMSE（越小越好）", fontsize=12)
    plt.legend(loc="best", fontsize=10)
    plt.grid(alpha=0.3)

    # 保存图片
    save_path = os.path.join(FOLDS_20M_DIR, f"{target_algo}_convergence_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"【保存提示】收敛曲线已保存至：{save_path}")


# ===================== 3. 超参数网格分析（lr+reg）【修复所有错误】 =====================
def analyze_param_grid():
    """分析超参数网格（学习率+正则化）结果，绘制热力图，实时显示进度"""
    if not (SWEEPS_DIR and PARAM_GRID_DIR and os.path.exists(PARAM_GRID_DIR)):
        print("\n【警告】超参数网格目录不存在，跳过该步骤")
        return None

    print("\n【进程提示】开始执行：超参数网格分析")
    # 筛选超参数结果文件
    param_files = [f for f in os.listdir(PARAM_GRID_DIR) if f.startswith("results_") and f.endswith(".json")]
    if not param_files:
        print("【警告】未找到超参数结果文件，跳过该步骤")
        return None

    param_results = []
    # 实时进度条
    for param_file in tqdm(param_files, desc="超参数文件解析进度"):
        # 解析文件名：results_lr0.01_reg0.01.json（修复小数点截断问题）
        file_name = param_file.replace(".json", "")
        parts = file_name.split("_")
        if len(parts) < 3 or not parts[1].startswith("lr") or not parts[2].startswith("reg"):
            print(f"【警告】文件名{param_file}格式异常（非results_lrX_regY.json），跳过")
            continue

        # 正确提取lr和reg（支持小数）
        try:
            lr_str = parts[1].replace("lr", "")
            reg_str = parts[2].replace("reg", "")
            lr = float(lr_str)
            reg = float(reg_str)
        except ValueError:
            print(f"【警告】无法解析{param_file}的lr/reg参数（{lr_str}/{reg_str}），跳过")
            continue

        file_path = os.path.join(PARAM_GRID_DIR, param_file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                param_data = json.load(f)

            # 【核心修复】从summary中提取nc_mf_sgd的均值和标准差（匹配实际JSON结构）
            mean_rmse = np.nan
            std_rmse = np.nan
            if "summary" in param_data and "nc_mf_sgd" in param_data["summary"]:
                summary = param_data["summary"]["nc_mf_sgd"]
                mean_rmse = summary.get("mean", np.nan)
                std_rmse = summary.get("std", np.nan)

            # 仅保留有有效RMSE的记录
            if not np.isnan(mean_rmse):
                param_info = {
                    "学习率(lr)": lr,
                    "正则化系数(reg)": reg,
                    "平均验证RMSE": round(mean_rmse, 6),
                    "RMSE标准差": round(std_rmse, 6) if not np.isnan(std_rmse) else np.nan
                }
                param_results.append(param_info)
            else:
                print(f"【警告】文件{param_file}未找到nc_mf_sgd的mean值，跳过")
        except Exception as e:
            print(f"【错误】处理文件{param_file}失败：{e}")
            continue

    # 整理结果
    param_df = pd.DataFrame(param_results)
    if param_df.empty:
        print("【警告】未获取有效超参数数据，跳过后续操作")
        return None

    # 核心修复：自动去重（解决重复索引报错）
    print(f"【提示】超参数数据 - 去重前条数：{len(param_df)}")
    # 按“学习率(lr)”和“正则化系数(reg)”组合去重，保留第一条有效记录
    param_df = param_df.drop_duplicates(
        subset=["学习率(lr)", "正则化系数(reg)"],
        keep="first"  # 保留第一条，也可改为"last"保留最后一条
    )
    print(f"【提示】超参数数据 - 去重后条数：{len(param_df)}")

    # 生成透视表（热力图数据源）
    try:
        rmse_pivot = param_df.pivot(index="学习率(lr)", columns="正则化系数(reg)", values="平均验证RMSE")
        print("\n【结果提示】超参数网格分析完成，平均RMSE透视表如下：")
        print(rmse_pivot.round(6).to_string())
    except Exception as e:
        print(f"【警告】生成透视表失败：{e}，跳过热力图绘制")
        rmse_pivot = None

    # 保存结果
    csv_save_path = os.path.join(PARAM_GRID_DIR, "param_grid_analysis.csv")
    param_df.to_csv(csv_save_path, index=False, encoding="utf-8")
    print(f"【保存提示】超参数分析结果已保存至：{csv_save_path}")

    # 绘制热力图（仅当透视表有效时，修复na_values错误）
    if rmse_pivot is not None and not rmse_pivot.isna().all().all():
        plot_param_heatmap(rmse_pivot)
    else:
        print("【警告】透视表无有效数据，跳过热力图绘制")
    return param_df


# ===================== 辅助：绘制超参数热力图（修复na_values错误） =====================
def plot_param_heatmap(rmse_pivot):
    """绘制学习率-正则化系数RMSE热力图（移除无效参数na_values）"""
    print("\n【进程提示】开始绘制：超参数热力图")
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(10, 8))

    # 绘制热力图（移除na_values，提前处理空值）
    rmse_pivot = rmse_pivot.fillna(0)  # 处理空值（如有）
    with np.errstate(invalid='ignore'):
        sns.heatmap(
            rmse_pivot,
            annot=True,
            cmap="YlGnBu_r",  # 颜色越深RMSE越小，更直观
            fmt=".6f",
            linewidths=0.5,
            cbar_kws={"label": "平均验证RMSE（越小越好）"}
            # 移除无效参数na_values
        )
    plt.title("nc_mf_sgd 算法学习率-正则化系数 RMSE 热力图", fontsize=14)
    plt.xlabel("正则化系数(reg)", fontsize=12)
    plt.ylabel("学习率(lr)", fontsize=12)

    # 保存图片
    save_path = os.path.join(PARAM_GRID_DIR, "lr_reg_rmse_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"【保存提示】超参数热力图已保存至：{save_path}")


# ===================== 4. 秩（Rank）扫描分析【完全适配实际JSON结构】 =====================
def analyze_rank_sweep():
    """分析不同秩（Rank）下的算法性能，绘制对比曲线，实时显示进度"""
    if not (SWEEPS_DIR and RANK_SWEEP_DIR and os.path.exists(RANK_SWEEP_DIR)):
        print("\n【警告】秩扫描目录不存在，跳过该步骤")
        return None

    print("\n【进程提示】开始执行：秩（Rank）扫描分析")
    # 筛选秩扫描结果文件
    rank_files = [f for f in os.listdir(RANK_SWEEP_DIR) if f.startswith("results_rank") and f.endswith(".json")]
    if not rank_files:
        print("【警告】未找到秩扫描结果文件，跳过该步骤")
        return None

    rank_results = []
    # 实时进度条
    for rank_file in tqdm(rank_files, desc="秩文件解析进度"):
        # 解析文件名：results_rank8.json（修复小数点截断问题）
        file_name = rank_file.replace(".json", "")
        parts = file_name.split("_")
        if len(parts) < 2 or not parts[1].startswith("rank"):
            print(f"【警告】文件名{rank_file}格式异常（非results_rankX.json），跳过")
            continue

        # 正确提取rank值
        try:
            rank_str = parts[1].replace("rank", "")
            rank = int(rank_str)
        except ValueError:
            print(f"【警告】无法解析{rank_file}的秩参数（{rank_str}），跳过")
            continue

        file_path = os.path.join(RANK_SWEEP_DIR, rank_file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                rank_data = json.load(f)

            # 【核心修复】从summary中提取所有方法的mean值（匹配实际JSON结构）
            rank_info = {"秩(Rank)": rank}
            if "summary" in rank_data:
                for algo_name, algo_summary in rank_data["summary"].items():
                    # 提取该算法的均值
                    mean_rmse = algo_summary.get("mean", np.nan)
                    if not np.isnan(mean_rmse):
                        rank_info[algo_name] = round(mean_rmse, 6)

            # 仅当存在算法RMSE时，添加到结果列表
            if len(rank_info) > 1:  # 除了"秩(Rank)"还有其他列
                rank_results.append(rank_info)
        except Exception as e:
            print(f"【警告】处理文件{rank_file}失败：{e}")
            continue

    # 空数据判断：先判断是否有有效数据
    if not rank_results:
        print("【警告】未获取有效秩扫描数据，跳过后续操作")
        return None

    # 整理结果并排序
    rank_df = pd.DataFrame(rank_results)
    # 判断"秩(Rank)"列是否存在并排序
    if "秩(Rank)" in rank_df.columns:
        rank_df = rank_df.sort_values("秩(Rank)")

    print("\n【结果提示】秩扫描分析完成，各算法RMSE结果如下：")
    print(rank_df.to_string(index=False))

    # 保存结果
    csv_save_path = os.path.join(RANK_SWEEP_DIR, "rank_sweep_analysis.csv")
    rank_df.to_csv(csv_save_path, index=False, encoding="utf-8")
    print(f"【保存提示】秩扫描结果已保存至：{csv_save_path}")

    # 绘制秩-RMSE对比曲线
    plot_rank_vs_rmse(rank_df)
    return rank_df


# ===================== 辅助：绘制秩-RMSE对比曲线 =====================
def plot_rank_vs_rmse(rank_df):
    """绘制不同秩下各算法的RMSE对比曲线"""
    print("\n【进程提示】开始绘制：秩-RMSE对比曲线")
    # 空数据判断
    if rank_df.empty:
        print("【警告】无有效秩数据，跳过绘图")
        return

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(12, 6))

    # 获取所有算法列（排除秩列）
    algo_cols = [col for col in rank_df.columns if col != "秩(Rank)"]
    for algo in algo_cols:
        if algo in rank_df.columns:
            # 过滤NaN值后绘制
            valid_data = rank_df[["秩(Rank)", algo]].dropna()
            if not valid_data.empty:
                plt.plot(
                    valid_data["秩(Rank)"],
                    valid_data[algo],
                    marker="o",
                    linewidth=2,
                    label=algo
                )

    plt.title("不同秩（Rank）下各算法验证RMSE对比曲线", fontsize=14)
    plt.xlabel("秩（Rank）", fontsize=12)
    plt.ylabel("验证RMSE（越小越好）", fontsize=12)
    plt.legend(loc="best", fontsize=10)
    plt.grid(alpha=0.3)

    # 保存图片
    save_path = os.path.join(RANK_SWEEP_DIR, "rank_vs_rmse_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"【保存提示】秩-RMSE对比曲线已保存至：{save_path}")


# ===================== 主函数：统一执行所有分析 =====================
if __name__ == "__main__":
    print("=" * 60)
    print("开始执行 folds_hash_20m 全量分析（带实时进程显示）")
    print("=" * 60)

    # 依次执行所有分析模块
    fold_stats_df = analyze_fold_data()
    history_analysis_df = analyze_history_files()
    param_grid_df = analyze_param_grid()
    rank_sweep_df = analyze_rank_sweep()

    print("\n" + "=" * 60)
    print("所有分析任务执行完成！所有有效结果已保存至对应目录~")
    print("=" * 60)