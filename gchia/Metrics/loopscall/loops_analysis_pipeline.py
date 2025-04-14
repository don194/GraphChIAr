#!/usr/bin/env python
"""
Chromatin Loops Analysis Pipeline

This script provides a comprehensive pipeline for analyzing chromatin loops:
1. Processes real HiC data and prediction results
2. Calls loops using chromatin_loop_pipline.py
3. Compares loops between real and predicted data using anchorEx.py
4. Generates visualization for loop counts and recovery rates comparison
"""

import os
import argparse
import subprocess
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import json
from tqdm import tqdm
import shutil

# Global color palette
COLOR_PALETTE = ['#6EC5E9', '#3CAEA3', '#20639B','#FF7F0E' ]
def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('loops_analysis_pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('LoopsAnalysisPipeline')

def run_loop_calling(input_file, output_dir, chromosome, resolution, is_hic=True, maxapart=2000000):
    """
    Run chromatin loop calling pipeline
    
    Parameters:
    -----------
    input_file : str
        Path to input file/directory
    output_dir : str
        Path to output directory (not used directly, kept for API consistency)
    chromosome : str
        Chromosome to analyze
    resolution : int
        Resolution in base pairs
    is_hic : bool
        Whether input is a HiC file or prediction directory
    maxapart : int
        Maximum genomic distance between two loci
        
    Returns:
    --------
    loops_file : str
        Path to the called loops file
    """
    # Determine the parent directory of the input file
    if is_hic:
        parent_dir = os.path.dirname(os.path.abspath(input_file))
    else:
        parent_dir = os.path.dirname(os.path.abspath(input_file))
    
    # Expected output path is parent_dir/{resolution}_loops
    expected_loops_dir = os.path.join(parent_dir, f"{resolution}_loops")
    
    cmd = [
        'python', 'chromatin_loop_pipline.py',
    ]
    
    if is_hic:
        cmd.extend(['--hic', input_file])
    else:
        cmd.extend(['--pred-dir', input_file])
    
    # No need to provide --output-dir as the script will use the default location
    cmd.extend([
        '--output-dir', expected_loops_dir,
        '--chromosome', chromosome,
        '--resolution', str(resolution),
        '--maxapart', str(maxapart)
    ])
    
    logging.info(f"Running loop calling with command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info("Loop calling completed successfully")
        
        # Find the loops file in the expected output directory
        loops_dir = os.path.join(expected_loops_dir, "loops")
        loops_files = glob.glob(os.path.join(loops_dir, f"*_hiccups_loops.bedpe"))
        
        if loops_files:
            return loops_files[0]
        else:
            logging.error(f"No loops file found in expected output directory: {loops_dir}")
            return None
            
    except subprocess.CalledProcessError as e:
        logging.error(f"Loop calling failed: {e.stderr.decode('utf-8') if e.stderr else str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error occurred during loop calling: {str(e)}")
        return None

def compare_loops_with_reference(reference_loops, query_loops, extending_size, resolution, chr_sizes=None):
    """
    Compare loops with reference loops and calculate recovery rate
    
    Parameters:
    -----------
    reference_loops : str
        Path to reference loops file
    query_loops : str
        Path to query loops file
    extending_size : int
        Extending size parameter for comparison
    resolution : int
        Resolution in base pairs
    chr_sizes : str
        Path to chromosome sizes file (optional)
        
    Returns:
    --------
    tuple
        (recovered_count, total_ref_count, recovery_rate)
    """
    cmd = [
        'python', 'anchorEx.py',
        '--reference', reference_loops,
        '--query', query_loops,
        '--extending_size', str(extending_size),
        '--resolution', str(resolution)
    ]
    
    if chr_sizes:
        cmd.extend(['--chr_sizes', chr_sizes, '--use_windows'])
    
    logging.info(f"Comparing loops with command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        
        # Parse results - looking for lines containing recovery information
        lines = output.split('\n')
        recovered_line = [line for line in lines if "恢复的loops数量:" in line][0]
        recovery_rate_line = [line for line in lines if "恢复率:" in line][0]
        
        # Calculate recovery rate
        recovered, total_ref = map(int, recovered_line.split(': ')[1].split('/'))
        recovery_rate = float(recovery_rate_line.split(': ')[1].strip('%')) / 100
        
        return recovered, total_ref, recovery_rate
            
    except subprocess.CalledProcessError as e:
        logging.error(f"Loops comparison failed: {e.stderr.decode('utf-8') if e.stderr else str(e)}")
        return None, None, None
    except Exception as e:
        logging.error(f"Error occurred during loops comparison: {str(e)}")
        return None, None, None


def plot_combined_counts_and_recovery(result_data, output_file, ref_loop_count):
    """
    Generate combined plot with loop counts and recovery rates
    
    Parameters:
    -----------
    result_data : dict
        Dictionary containing loop count and recovery rate information
    output_file : str
        Path to output file
    ref_loop_count : int
        The number of loops in the reference dataset
    """
    # Set global font size
    plt.rcParams.update({'font.size': 14})
    
    # Create two subplots, adjust overall dimensions
    fig, axes = plt.subplots(2, 1, figsize=(7, 10), sharex=False)
    
    # Define custom color palette
    custom_colors = COLOR_PALETTE

    # ============ Top subplot: Loop Counts chart ============
    ax = axes[0]
    
    # Set subplot to be square
    ax.set_box_aspect(1)
    
    # Extract loop counts and names
    names = []
    counts = []
    
    # Reference first
    names.append(result_data['reference']['name'])
    counts.append(result_data['reference']['loop_count'])
    
    # Then other HiC files
    for hic in result_data.get('other_hic', []):
        names.append(hic['name'])
        counts.append(hic['loop_count'])
    
    # Then prediction results
    for pred in result_data.get('predictions', []):
        names.append(pred['name'])
        counts.append(pred['loop_count'])
    
    # Create color mapping for data sources to ensure consistent colors across subplots
    color_map = {}
    color_map[result_data['reference']['name']] = custom_colors[0]
    
    for i, hic in enumerate(result_data.get('other_hic', [])):
        color_map[hic['name']] = custom_colors[(i + 1) % len(custom_colors)]
    
    for i, pred in enumerate(result_data.get('predictions', [])):
        color_map[pred['name']] = custom_colors[(i + 1 + len(result_data.get('other_hic', []))) % len(custom_colors)]
    
    # Convert counts to thousands for display only
    counts_in_thousands = [c/1000 for c in counts]
    
    # Generate bar chart using color mapping
    bar_colors = [color_map[name] for name in names]
    bars = ax.bar(names, counts_in_thousands, color=bar_colors)
    
    # Add count labels with original count values
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(counts_in_thousands),
                f'{counts[i]}', ha='center', va='bottom', fontsize=16)
    
    # Customize plot
    ax.set_ylabel('Loop Count (×1000)', fontsize=20)
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([int(i) for i in ax.get_yticks() if i >= 0])  # Only positive integer ticks
    current_ylim = ax.get_ylim()[1]
    ax.set_ylim(0, current_ylim * 1.2)
    ax.tick_params(axis='y', labelsize=16, direction='in')  # Ticks pointing inward
    
    # Add legend for the top plot
    legend_labels = [f"{names[i]}" for i in range(len(names))]
    ax.legend(bars, legend_labels, loc='upper left', bbox_to_anchor=(1.05, 1), 
              fontsize=18, frameon=False)
    
    # =========== Bottom subplot: Recovery Rates chart ===========
    ax = axes[1]
    
    # Set subplot to be square
    ax.set_box_aspect(1)
    
    # Create x-axis (extending sizes)
    extending_sizes = [0, 1, 2, 3]
    
    # Plot styles
    marker_styles = ['o', 's', '^', 'D', 'x', '*', 'p']
    line_styles = ['-', '--', '-.', ':']
    
    # Store all lines for legend and their y values for annotation placement
    all_lines = []
    all_rates = []
    all_names = []
    
    # First collect all data
    # Plot recovery rates for predictions
    for pred in result_data.get('predictions', []):
        if 'recovery_rates' not in pred:
            continue
            
        rates = [pred['recovery_rates'].get(str(s), 0) for s in extending_sizes]
        rates_percentage = [r * 100 for r in rates]  # Convert to percentage
        
        marker_idx = all_names.index(pred['name']) if pred['name'] in all_names else len(all_names)
        marker = marker_styles[marker_idx % len(marker_styles)]
        line = line_styles[marker_idx % len(line_styles)]
        color = color_map[pred['name']]  # Use previously mapped color
        
        line_obj = ax.plot(extending_sizes, rates_percentage, marker=marker, linestyle=line, 
                 linewidth=2, markersize=8, label=pred['name'], color=color)
        
        all_lines.append(line_obj[0])
        all_rates.append(rates_percentage)
        all_names.append(pred['name'])
    
    # Plot recovery rates for other HiC files
    for hic in result_data.get('other_hic', []):
        if 'recovery_rates' not in hic:
            continue
            
        rates = [hic['recovery_rates'].get(str(s), 0) for s in extending_sizes]
        rates_percentage = [r * 100 for r in rates]  # Convert to percentage
        
        marker_idx = all_names.index(hic['name']) if hic['name'] in all_names else len(all_names)
        marker = marker_styles[marker_idx % len(marker_styles)]
        line = line_styles[marker_idx % len(line_styles)]
        color = color_map[hic['name']]  # Use previously mapped color
        
        line_obj = ax.plot(extending_sizes, rates_percentage, marker=marker, linestyle=line, 
                 linewidth=2, markersize=8, label=hic['name'], color=color)
        
        all_lines.append(line_obj[0])
        all_rates.append(rates_percentage)
        all_names.append(hic['name'])
    
    # Stagger annotations to avoid overlap
    for x_idx, x in enumerate(extending_sizes):
        # Get y-values for this x position across all lines
        points_at_x = [(i, line_rates[x_idx]) for i, line_rates in enumerate(all_rates)]
        # Sort by y-value (recovery rate)
        points_at_x.sort(key=lambda p: p[1])

        # Special handling for s=0
        if x == 0:
            for i, (line_idx, y) in enumerate(points_at_x):
                line_color = all_lines[line_idx].get_color()
                if i == len(points_at_x) - 1:  # Annotate the highest value above
                    ax.text(x, y + 2, f'{y:.1f}%', ha='center', va='bottom', 
                            fontsize=16, color=line_color)
                else:  # Annotate other values below
                    ax.text(x, y - 4, f'{y:.1f}%', ha='center', va='top', 
                            fontsize=16, color=line_color)
        else:
            # For other s values, alternate annotations
            for i, (line_idx, y) in enumerate(points_at_x):
                line_color = all_lines[line_idx].get_color()
                position = (i+1) % 2  # 0 for up, 1 for down
                
                if position == 0:  # Annotate above
                    ax.text(x, y + 2, f'{y:.1f}%', ha='center', va='bottom', 
                            fontsize=16, color=line_color)
                else:  # Annotate below
                    ax.text(x, y - 4, f'{y:.1f}%', ha='center', va='top', 
                            fontsize=16, color=line_color)

    
    # Customize plot
    ax.set_xlabel('Extending Size (s)', fontsize=20)
    ax.set_ylabel('Recovery Rate (%)', fontsize=20)
    ax.set_xticks(extending_sizes)
    ax.set_xticklabels([str(s) for s in extending_sizes], fontsize=14)
    ax.tick_params(axis='both', labelsize=16, direction='in')  # Ticks pointing inward
    ax.set_ylim(0, 90)  # Set y-axis upper limit to 90%
    ax.set_xlim(-0.5, 3.5)
    # Add legend for the bottom plot
    ax.legend(all_lines, all_names, loc='upper left', bbox_to_anchor=(1.05, 1), 
              fontsize=18, frameon=False)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, right=0.75)  # Increase space between subplots for square shape
    
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Combined loop counts and recovery rates plot saved to {output_file}")



def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Comprehensive chromatin loops analysis pipeline')
    
    # Required arguments
    parser.add_argument('--reference', '-r', required=True, help='Path to reference HiC file')
    parser.add_argument('--predictions', '-p', required=True, nargs='+', 
                       help='Paths to prediction directories (can specify multiple)')
    parser.add_argument('--chromosome', '-c', required=True, help='Chromosome to analyze (e.g., chr21)')
    parser.add_argument('--resolution', type=int, default=10000, help='Resolution in base pairs (default: 10000)')
    parser.add_argument('--output-dir', required=True, help='Path to output directory for final analysis results')
    
    # Optional arguments
    parser.add_argument('--other-hic', '-o', nargs='+', help='Other HiC files for comparison (optional)')
    parser.add_argument('--maxapart', type=int, default=2000000,
                        help='Maximum genomic distance between two loci (default: 2000000)')
    parser.add_argument('--chr-sizes', help='Path to chromosome sizes file (optional)')
    parser.add_argument('--names', nargs='+', help='Names for prediction results (optional, in same order as --predictions)')
    parser.add_argument('--other-names', nargs='+', help='Names for other HiC files (optional, in same order as --other-hic)')
    parser.add_argument('--reference-name', help='Name for reference HiC file (optional, defaults to filename)')
    
    args = parser.parse_args()
    
    # Validate that names match the number of predictions if provided
    if args.names and len(args.names) != len(args.predictions):
        parser.error("Number of names must match number of prediction directories")
    
    # Validate that other-names match the number of other HiC files if provided
    if args.other_names and args.other_hic and len(args.other_names) != len(args.other_hic):
        parser.error("Number of other-names must match number of other HiC files")
        
    return args

def main():
    """Main function"""
    # Set up logging
    logger = setup_logging()
    
    # Parse arguments
    args = parse_arguments()
    
    # Create final output directory for analysis results
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否存在已有的分析结果文件
    results_file = os.path.join(output_dir, "analysis_results.json")
    
    if os.path.exists(results_file):
        logger.info(f"Found existing analysis results file: {results_file}")
        logger.info("Loading existing results instead of rerunning analysis...")
        
        try:
            # 读取现有JSON结果文件
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # 获取参考循环数量
            ref_loop_count = results['reference']['loop_count']
            
            # 直接生成可视化图表
            logger.info("Generating combined loop counts and recovery rates plot from existing data")
            combined_plot_file = os.path.join(output_dir, "combined_counts_and_recovery.svg")
            plot_combined_counts_and_recovery(results, combined_plot_file, ref_loop_count)
            
            # 打印摘要
            print("\n=== Analysis Pipeline Summary (from cached results) ===")
            print(f"Reference HiC: {args.reference} ({ref_loop_count} loops)")
            
            print("\nPrediction Results:")
            for pred in results['predictions']:
                rates_str = " ".join([f"s={s}:{float(rate):.2%}" for s, rate in pred['recovery_rates'].items()])
                print(f"  - {pred['name']}: {pred['loop_count']} loops")
                print(f"    Recovery rates: {rates_str}")
            
            if results.get('other_hic'):
                print("\nOther HiC Results:")
                for other in results['other_hic']:
                    rates_str = " ".join([f"s={s}:{float(rate):.2%}" for s, rate in other.get('recovery_rates', {}).items()])
                    print(f"  - {other['name']}: {other['loop_count']} loops")
                    print(f"    Recovery rates: {rates_str}")
            
            print(f"\nIntermediate results: in each input file's parent directory/{args.resolution}_loops")
            print(f"Final results directory: {output_dir}")
            print(f"Combined plot: {combined_plot_file}")
            
            return 0
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load or process existing results file: {str(e)}")
            logger.info("Falling back to full analysis...")
    
    # 如果没有找到JSON文件或读取失败，执行完整分析流程
    
    # Results dictionary to store all analysis data
    results = {
        'reference': {},
        'predictions': [],
        'other_hic': []
    }
    
    # Extract file basename for reference or use provided name
    if args.reference_name:
        ref_name = args.reference_name
    else:
        ref_basename = os.path.basename(args.reference).split('.')[0]
        ref_name = ref_basename
    
    results['reference']['name'] = ref_name
    
    # Process reference HiC file
    logger.info(f"Processing reference HiC file: {args.reference}")
    ref_loops_file = run_loop_calling(
        args.reference, None, args.chromosome, args.resolution, 
        is_hic=True, maxapart=args.maxapart
    )
    
    if not ref_loops_file:
        logger.error("Failed to process reference HiC file. Exiting.")
        return 1
    
    # Count loops in reference file
    with open(ref_loops_file, 'r') as f:
        ref_loop_count = sum(1 for line in f if not line.startswith('#'))
    
    results['reference']['loop_count'] = ref_loop_count
    results['reference']['loops_file'] = ref_loops_file
    logger.info(f"Reference HiC file processed. Found {ref_loop_count} loops.")
    
    # Process prediction directories
    for i, pred_dir in enumerate(args.predictions):
        # Use provided name or extract from directory path
        if args.names and i < len(args.names):
            pred_name = args.names[i]
        else:
            pred_name = os.path.basename(pred_dir)
        
        logger.info(f"Processing prediction directory: {pred_dir} (name: {pred_name})")
        
        pred_loops_file = run_loop_calling(
            pred_dir, None, args.chromosome, args.resolution, 
            is_hic=False, maxapart=args.maxapart
        )
        
        if not pred_loops_file:
            logger.error(f"Failed to process prediction directory: {pred_dir}")
            continue
        
        # Count loops in prediction file
        with open(pred_loops_file, 'r') as f:
            pred_loop_count = sum(1 for line in f if not line.startswith('#'))
            
        # Compare with reference at different extending sizes
        recovery_rates = {}
        
        for s in [0, 1, 2, 3]:
            recovered, total_ref, rate = compare_loops_with_reference(
                ref_loops_file, pred_loops_file, s, args.resolution, args.chr_sizes
            )
            
            if rate is not None:
                recovery_rates[str(s)] = rate
                logger.info(f"Prediction {pred_name}, extending_size={s}: Recovery rate = {rate:.2%}")
            
        # Store results
        pred_result = {
            'name': pred_name,
            'loops_file': pred_loops_file,
            'loop_count': pred_loop_count,
            'recovery_rates': recovery_rates,
        }
        
        results['predictions'].append(pred_result)
        logger.info(f"Prediction directory processed. Found {pred_loop_count} loops.")
    
    # Process other HiC files if provided
    if args.other_hic:
        for i, other_hic in enumerate(args.other_hic):
            # Use provided name or extract from file path
            if args.other_names and i < len(args.other_names):
                other_name = args.other_names[i]
            else:
                other_name = os.path.basename(other_hic).split('.')[0]
            
            logger.info(f"Processing other HiC file: {other_hic} (name: {other_name})")
            
            other_loops_file = run_loop_calling(
                other_hic, None, args.chromosome, args.resolution, 
                is_hic=True, maxapart=args.maxapart
            )
            
            if not other_loops_file:
                logger.error(f"Failed to process other HiC file: {other_hic}")
                continue
            
            # Count loops in other HiC file
            with open(other_loops_file, 'r') as f:
                other_loop_count = sum(1 for line in f if not line.startswith('#'))
                
            # Compare with reference at different extending sizes
            recovery_rates = {}
            
            for s in [0, 1, 2, 3]:
                recovered, total_ref, rate = compare_loops_with_reference(
                    ref_loops_file, other_loops_file, s, args.resolution, args.chr_sizes
                )
                
                if rate is not None:
                    recovery_rates[str(s)] = rate
                    logger.info(f"Other HiC {other_name}, extending_size={s}: Recovery rate = {rate:.2%}")
            
            # Store results
            other_result = {
                'name': other_name,
                'loops_file': other_loops_file,
                'loop_count': other_loop_count,
                'recovery_rates': recovery_rates,
            }
            
            results['other_hic'].append(other_result)
            logger.info(f"Other HiC file processed. Found {other_loop_count} loops.")
    
    # Generate plots and save to the specified output directory
    logger.info("Generating combined loop counts and recovery rates plot")
    combined_plot_file = os.path.join(output_dir, "combined_counts_and_recovery.svg")
    plot_combined_counts_and_recovery(results, combined_plot_file, ref_loop_count)
    
    # Save results to JSON
    results_file = os.path.join(output_dir, "analysis_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print summary
    print("\n=== Analysis Pipeline Summary ===")
    print(f"Reference HiC: {args.reference} ({ref_loop_count} loops)")
    
    print("\nPrediction Results:")
    for pred in results['predictions']:
        rates_str = " ".join([f"s={s}:{rate:.2%}" for s, rate in pred['recovery_rates'].items()])
        print(f"  - {pred['name']}: {pred['loop_count']} loops")
        print(f"    Recovery rates: {rates_str}")
    
    if args.other_hic:
        print("\nOther HiC Results:")
        for other in results['other_hic']:
            rates_str = " ".join([f"s={s}:{rate:.2%}" for s, rate in other.get('recovery_rates', {}).items()])
            print(f"  - {other['name']}: {other['loop_count']} loops")
            print(f"    Recovery rates: {rates_str}")
    
    print(f"\nIntermediate results: in each input file's parent directory/{args.resolution}_loops")
    print(f"Final results directory: {output_dir}")
    print(f"Combined plot: {combined_plot_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())