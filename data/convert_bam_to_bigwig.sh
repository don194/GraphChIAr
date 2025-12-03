#!/bin/bash

# 处理 BAM 文件：生成索引并转换为 BigWig
process_bam_files() {
    local bam_file=$1
    local dir=$(dirname "$bam_file")
    local cell_line=$(basename $(dirname "$dir"))  # 提取细胞系名称（上一级文件夹名称）
    local bam_base=$(basename "$bam_file" .bam)  # 提取 BAM 文件的基础名称，不含扩展名
    local bigwig_dir=$(dirname "$dir")/bigWig_files  # BigWig 文件将存放的目录
    local output_bigwig="$bigwig_dir/${bam_base}.bw"

    # 创建 BigWig 文件存放的目录
    mkdir -p "$bigwig_dir"

    # 生成 BAM 索引
    if [[ ! -f "${bam_file}.bai" ]]; then
        echo "Indexing $bam_file..."
        samtools index "$bam_file"
    else
        echo "Index for $bam_file already exists. Skipping indexing."
    fi

    # 如果 BigWig 文件已存在且其修改时间比 BAM 文件更新，则跳过转换
    if [[ -f "$output_bigwig" && "$output_bigwig" -nt "$bam_file" ]]; then
        echo "BigWig file $output_bigwig already exists and is up to date. Skipping conversion."
        return
    fi

    # 将 BAM 文件转换为 BigWig
    echo "Converting $bam_file to BigWig: $output_bigwig"
    bamCoverage -b "$bam_file" -o "$output_bigwig" --binSize 10 --normalizeUsing RPGC --effectiveGenomeSize 2913022398

    echo "Converted $bam_file to BigWig and stored in $output_bigwig"
}

# 遍历所有 bam_files 文件夹中的 BAM 文件，生成索引并转换为 BigWig
for bam_file in $(find . -type f -path "*/bam_files/*.bam"); do
    process_bam_files "$bam_file"
done

echo "BAM files indexing and conversion to BigWig completed."
