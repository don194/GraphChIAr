import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import os

def read_bedpe(file_path):
    """读取BEDPE格式文件"""
    df = pd.read_csv(file_path, sep='\t', header=None)
    # 确保至少有6列数据
    if df.shape[1] < 6:
        raise ValueError(f"BEDPE文件{file_path}格式不正确，至少需要6列")
    
    # 保留前6列 (chr1, start1, end1, chr2, start2, end2)
    if df.shape[1] > 6:
        df = df.iloc[:, :6]
    
    # 设置列名
    df.columns = ['chr1', 'start1', 'end1', 'chr2', 'start2', 'end2']
    
    return df

def read_chromosome_sizes(file_path, specific_chrs=None):
    """
    读取染色体大小文件
    
    参数:
    file_path: 染色体大小文件路径
    specific_chrs: 指定的染色体列表，如果提供则只返回这些染色体的信息
    
    返回:
    DataFrame: 包含染色体名称和大小的DataFrame
    """
    chr_sizes_df = pd.read_csv(file_path, sep='\t', header=None, names=['chr', 'size'])
    
    # 如果指定了特定染色体，筛选数据
    if specific_chrs:
        # 确保染色体名称格式一致
        formatted_chrs = []
        for chrom in specific_chrs:
            if not chrom.startswith('chr') and not chrom.lower() in ['x', 'y', 'm', 'mt']:
                formatted_chrs.append(f"chr{chrom}")
            else:
                formatted_chrs.append(chrom)
                
        chr_sizes_df = chr_sizes_df[chr_sizes_df['chr'].isin(formatted_chrs)]
        
        if chr_sizes_df.empty:
            raise ValueError(f"在染色体大小文件中未找到指定的染色体: {specific_chrs}")
    
    return chr_sizes_df

def convert_to_midpoint(loops_df):
    """将loops转换为中点表示"""
    mid_df = loops_df.copy()
    mid_df['mid1'] = (mid_df['start1'] + mid_df['end1']) // 2
    mid_df['mid2'] = (mid_df['start2'] + mid_df['end2']) // 2
    return mid_df

def filter_loops_by_windows(loops_df, chr_sizes_df, window_size=2000000, step_size=2000000):
    """
    使用滑动窗口筛选loops
    
    参数:
    loops_df: 包含loops的DataFrame
    chr_sizes_df: 染色体大小的DataFrame
    window_size: 窗口大小
    step_size: 滑动步长
    
    返回:
    filtered_loops: 在滑动窗口内的loops的索引
    """
    filtered_indices = []
    
    # 转换为中点表示
    mid_df = convert_to_midpoint(loops_df)
    
    # 遍历每条染色体
    for _, row in tqdm(chr_sizes_df.iterrows(), desc="处理染色体窗口"):
        chrom, size = row['chr'], row['size']
        
        # 在每条染色体上滑动窗口
        for start in range(0, size, step_size):
            end = start + window_size
            if end > size:
                continue
                
            # 筛选当前窗口内的loops
            window_indices = mid_df[
                ((mid_df['chr1'] == chrom) & (mid_df['mid1'] >= start) & (mid_df['mid1'] < end)) &
                ((mid_df['chr2'] == chrom) & (mid_df['mid2'] >= start) & (mid_df['mid2'] < end))
            ].index.tolist()
            
            filtered_indices.extend(window_indices)
    
    # 移除重复索引
    filtered_indices = list(set(filtered_indices))
    return filtered_indices

def filter_loops_by_chromosomes(loops_df, chromosomes):
    """
    根据指定染色体筛选loops
    
    参数:
    loops_df: 包含loops的DataFrame
    chromosomes: 指定的染色体列表
    
    返回:
    filtered_loops: 在指定染色体上的loops的DataFrame
    """
    # 确保染色体名称格式一致
    formatted_chrs = []
    for chrom in chromosomes:
        if not chrom.startswith('chr') and not chrom.lower() in ['x', 'y', 'm', 'mt']:
            formatted_chrs.append(f"chr{chrom}")
        else:
            formatted_chrs.append(chrom)
    
    # 筛选在指定染色体上的loops
    mask = (loops_df['chr1'].isin(formatted_chrs)) & (loops_df['chr2'].isin(formatted_chrs))
    return loops_df[mask]

def compare_loops(reference_loops, query_loops, extending_size=0, resolution=10000):
    """
    比较两个loops集合，使用extending size参数
    
    参数:
    reference_loops: 参考loops的DataFrame，包含chr1,start1,end1,chr2,start2,end2
    query_loops: 待比较loops的DataFrame，格式同上
    extending_size: 延伸大小参数s，表示匹配时允许的偏差，单位是bin数
    resolution: 分辨率，默认10kb
    
    返回:
    recovered_count: 被恢复的参考loops数量
    total_count: 参考loops总数
    recovery_rate: 恢复率
    recovered_indices: 被恢复的参考loops的索引
    """
    
    # 转换为中点表示
    ref_mid = convert_to_midpoint(reference_loops)
    query_mid = convert_to_midpoint(query_loops)
    
    # 按染色体对进行分组
    ref_by_chrs = {}
    for idx, row in ref_mid.iterrows():
        chr_pair = (row['chr1'], row['chr2'])
        if chr_pair not in ref_by_chrs:
            ref_by_chrs[chr_pair] = []
        ref_by_chrs[chr_pair].append((idx, row['mid1'], row['mid2']))
    
    # 对每个查询loop，检查是否与参考loop匹配
    recovered_indices = set()
    
    for _, q_row in tqdm(query_mid.iterrows(), total=len(query_mid), desc="比较loops"):
        chr_pair = (q_row['chr1'], q_row['chr2'])
        
        if chr_pair not in ref_by_chrs:
            continue
        
        q_mid1 = q_row['mid1']
        q_mid2 = q_row['mid2']
        
        for ref_idx, ref_mid1, ref_mid2 in ref_by_chrs[chr_pair]:
            # 计算两个中点之间的距离（以bin为单位）
            dist1 = abs(ref_mid1 - q_mid1) / resolution
            dist2 = abs(ref_mid2 - q_mid2) / resolution
            
            # 如果两个距离都在extending_size范围内，视为匹配
            if dist1 <= extending_size and dist2 <= extending_size:
                recovered_indices.add(ref_idx)
    
    recovered_count = len(recovered_indices)
    total_count = len(reference_loops)
    recovery_rate = recovered_count / total_count if total_count > 0 else 0
    
    return recovered_count, total_count, recovery_rate, recovered_indices

def compare_loops_with_fpr(reference_loops, query_loops, extending_size=0, resolution=10000):
    """
    比较两个loops集合，计算恢复率和假阳率
    
    参数:
    reference_loops: 参考loops的DataFrame，包含chr1,start1,end1,chr2,start2,end2
    query_loops: 待比较loops的DataFrame，格式同上
    extending_size: 延伸大小参数s，表示匹配时允许的偏差，单位是bin数
    resolution: 分辨率，默认10kb
    
    返回:
    recovered_count: 被恢复的参考loops数量
    total_ref_count: 参考loops总数
    recovery_rate: 恢复率
    false_positive_count: 假阳性的查询loops数量
    total_query_count: 查询loops总数
    false_positive_rate: 假阳率
    recovered_indices: 被恢复的参考loops的索引
    matched_query_indices: 匹配到参考的查询loops的索引
    """
    
    # 转换为中点表示
    ref_mid = convert_to_midpoint(reference_loops)
    query_mid = convert_to_midpoint(query_loops)
    
    # 按染色体对进行分组
    ref_by_chrs = {}
    for idx, row in ref_mid.iterrows():
        chr_pair = (row['chr1'], row['chr2'])
        if chr_pair not in ref_by_chrs:
            ref_by_chrs[chr_pair] = []
        ref_by_chrs[chr_pair].append((idx, row['mid1'], row['mid2']))
    
    # 对每个查询loop，检查是否与参考loop匹配
    recovered_indices = set()  # 被恢复的参考loops索引
    matched_query_indices = set()  # 匹配到参考的查询loops索引
    
    for q_idx, q_row in tqdm(query_mid.iterrows(), total=len(query_mid), desc="比较loops"):
        chr_pair = (q_row['chr1'], q_row['chr2'])
        is_matched = False
        
        if chr_pair not in ref_by_chrs:
            continue
        
        q_mid1 = q_row['mid1']
        q_mid2 = q_row['mid2']
        
        for ref_idx, ref_mid1, ref_mid2 in ref_by_chrs[chr_pair]:
            # 计算两个中点之间的距离（以bin为单位）
            dist1 = abs(ref_mid1 - q_mid1) / resolution
            dist2 = abs(ref_mid2 - q_mid2) / resolution
            
            # 如果两个距离都在extending_size范围内，视为匹配
            if dist1 <= extending_size and dist2 <= extending_size:
                recovered_indices.add(ref_idx)
                is_matched = True
        
        if is_matched:
            matched_query_indices.add(q_idx)
    
    # 计算统计指标
    recovered_count = len(recovered_indices)
    total_ref_count = len(reference_loops)
    recovery_rate = recovered_count / total_ref_count if total_ref_count > 0 else 0
    
    false_positive_count = len(query_loops) - len(matched_query_indices)
    total_query_count = len(query_loops)
    false_positive_rate = false_positive_count / total_query_count if total_query_count > 0 else 0
    
    return (recovered_count, total_ref_count, recovery_rate, 
            false_positive_count, total_query_count, false_positive_rate,
            recovered_indices, matched_query_indices)

def main():
    parser = argparse.ArgumentParser(description='比较两个loops文件并计算恢复率')
    parser.add_argument('--reference', required=True, help='参考loops文件路径(BEDPE格式)')
    parser.add_argument('--query', required=True, help='查询loops文件路径(BEDPE格式)')
    parser.add_argument('--extending_size', type=int, default=0, 
                        help='延伸大小参数，表示匹配时允许的偏差bin数，默认为0')
    parser.add_argument('--resolution', type=int, default=10000, 
                        help='分辨率，默认为10kb')
    parser.add_argument('--output', help='输出恢复的loops到文件(可选)')
    parser.add_argument('--chr_sizes', help='染色体大小文件路径')
    parser.add_argument('--window_size', type=int, default=2000000,
                        help='滑动窗口大小，默认为2000000')
    parser.add_argument('--step_size', type=int, default=2000000,
                        help='滑动窗口步长，默认为2000000')
    parser.add_argument('--use_windows', action='store_true',
                        help='是否使用滑动窗口筛选loops')
    parser.add_argument('--chromosomes', nargs='+', 
                        help='指定要处理的染色体，例如: --chromosomes chr1 chr2 或 --chromosomes 1 2')
    
    args = parser.parse_args()
    
    print(f"读取参考loops文件: {args.reference}")
    ref_loops = read_bedpe(args.reference)
    print(f"读取查询loops文件: {args.query}")
    query_loops = read_bedpe(args.query)
    
    print(f"参考loops数量: {len(ref_loops)}")
    print(f"查询loops数量: {len(query_loops)}")
    
    # 如果指定了染色体，先筛选
    if args.chromosomes:
        print(f"仅处理指定染色体: {', '.join(args.chromosomes)}")
        ref_loops = filter_loops_by_chromosomes(ref_loops, args.chromosomes)
        query_loops = filter_loops_by_chromosomes(query_loops, args.chromosomes)
        print(f"筛选后的参考loops数量: {len(ref_loops)}")
        print(f"筛选后的查询loops数量: {len(query_loops)}")
    
    # 如果启用窗口筛选且提供了染色体大小文件
    if args.use_windows and args.chr_sizes:
        print(f"使用滑动窗口筛选loops (window_size={args.window_size}, step_size={args.step_size})")
        chr_sizes_df = read_chromosome_sizes(args.chr_sizes, specific_chrs=args.chromosomes)
        
        # 筛选参考loops
        filtered_indices = filter_loops_by_windows(
            ref_loops, chr_sizes_df, 
            window_size=args.window_size, 
            step_size=args.step_size
        )
        
        filtered_ref_loops = ref_loops.loc[filtered_indices]
        print(f"窗口筛选后的参考loops数量: {len(filtered_ref_loops)} (原数量: {len(ref_loops)})")
        
        # 使用筛选后的参考loops进行比较
        recovered, total, rate, indices = compare_loops(
            filtered_ref_loops, query_loops, 
            extending_size=args.extending_size,
            resolution=args.resolution
        )
    else:
        # 不使用窗口筛选，直接比较所有loops
        recovered, total, rate, indices = compare_loops(
            ref_loops, query_loops, 
            extending_size=args.extending_size,
            resolution=args.resolution
        )
    
    print(f"\n结果 (extending_size={args.extending_size}):")
    print(f"恢复的loops数量: {recovered}/{total}")
    print(f"恢复率: {rate:.2%}")
    
    if args.output:
        # 确定要保存的loops集合
        if args.use_windows and args.chr_sizes:
            output_loops = filtered_ref_loops.loc[list(indices)]
        else:
            output_loops = ref_loops.loc[list(indices)]
            
        output_loops.to_csv(args.output, sep='\t', header=False, index=False)
        print(f"已将恢复的loops保存到: {args.output}")

if __name__ == "__main__":
    main()