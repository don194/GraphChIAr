#!/usr/bin/env python
# filepath: /home/dh/work/gChIA/src/gchia/Results/loopscall/chromatin_loop_pipeline.py
import os
import argparse
import subprocess
import logging
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Import existing conversion modules
from hic2cool import convert_hic_to_cool, balance_matrix
# Import functions from merge2cool for handling predicted matrices
from merge2cool import (
    load_matrix, convert_to_bg2_format, create_cool_file_from_bg2, 
    process_chromosome
)

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('chromatin_loop_pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('ChromatinLoopPipeline')

def fetch_chrom_sizes(genome, output_file):
    """Fetch chromosome sizes information from UCSC"""
    try:
        cmd = ['fetchChromSizes', genome]
        with open(output_file, 'w') as f:
            subprocess.run(cmd, check=True, stdout=f)
        logging.info(f"Chromosome sizes fetched from UCSC and saved to {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to fetch chromosome sizes: {e}")
        return False

def run_hiccups(cool_file, chromosome, output_dir, resolution, peak_width=[1, 2, 4], window_width=[3, 5, 7], maxapart=5000000):
    """Run pyHICCUPS to identify chromatin loops"""
    os.makedirs(output_dir, exist_ok=True)
    chromosome = chromosome.replace('chr', '')
    output_file = os.path.join(output_dir, f"{chromosome}_hiccups_loops.bedpe")
    
    # Run pyHICCUPS algorithm
    try:
        cmd = [
            'pyHICCUPS',
            '-O', output_file,
            '-p', f"{cool_file}",
            '-C', chromosome,
            '--pw', *map(str, peak_width),
            '--ww', *map(str, window_width),
            '--only-anchors',
            '--maxapart', str(maxapart),  
            '--siglevel', '0.01',
            '--nproc', '4'  # Use 4 processes, adjustable
        ]
        
        logging.info(f"Running pyHICCUPS with command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if os.path.exists(output_file):
            logging.info(f"Successfully identified loops at {output_file}")
            return output_file
        else:
            logging.error(f"Loop calling failed, output file not created: {output_file}")
            return None
            
    except subprocess.CalledProcessError as e:
        logging.error(f"Loop calling failed: {e.stderr.decode('utf-8') if e.stderr else str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error occurred during loop calling: {str(e)}")
        return None

def visualize_loops(cool_file, loops_file, chromosome, output_dir, resolution, start=2000000, end=4000000):
    """Visualize identified chromatin loops in the 2Mb-4Mb region by default"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{chromosome}_loops_visual.png")
    
    cmd = [
        'peak-plot',
        '-O', output_file,
        '-p', f"{cool_file}",
        '-I', loops_file,
        '-C', chromosome,
        '-S', str(start),
        '-E', str(end),
        '--clr-weight-name', 'weight'
    ]
    
    try:
        logging.info(f"Visualizing loops with command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if os.path.exists(output_file):
            logging.info(f"Successfully visualized loops at {output_file}")
            return output_file
        else:
            logging.error(f"Visualization failed, output file not created")
            return None
            
    except subprocess.CalledProcessError as e:
        logging.error(f"Visualization failed: {e.stderr.decode('utf-8') if e.stderr else str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error occurred during visualization: {str(e)}")
        return None

def run_apa_analysis(cool_file, loops_file, output_dir, resolution):
    """Run Aggregate Peak Analysis (APA)"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "apa_analysis.png")
    
    try:
        cmd = [
            'apa-analysis',
            '-O', output_file,
            '-p', f"{cool_file}",
            '-I', loops_file,
            '--clr-weight-name', 'weight',
            '--vmax', '2'
        ]
        
        logging.info(f"Running APA analysis with command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if os.path.exists(output_file):
            logging.info(f"Successfully performed APA analysis at {output_file}")
            return output_file
        else:
            logging.error(f"APA analysis failed, output file not created")
            return None
            
    except subprocess.CalledProcessError as e:
        logging.error(f"APA analysis failed: {e.stderr.decode('utf-8') if e.stderr else str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error occurred during APA analysis: {str(e)}")
        return None

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Pipeline for identifying chromatin loops from Hi-C data or predicted matrices')
    
    # Input options - either HiC file or predicted matrices directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--hic', help='Path to input .hic file')
    input_group.add_argument('--pred-dir', help='Directory containing predicted matrices (.npy files)')
    
    # Required arguments
    parser.add_argument('--output-dir', '-o', default='loop_output', help='Output directory')
    parser.add_argument('--chromosome', '-c', required=True, help='Chromosome to analyze (e.g., chr21)')
    
    # Optional arguments
    parser.add_argument('--resolution', '-r', type=int, default=10000, help='Resolution in base pairs (default: 10000)')
    parser.add_argument('--genome', '-g', default='hg38', help='Genome assembly (default: hg38)')
    parser.add_argument('--chrom-sizes', help='Path to chromosome sizes file (will be generated if not provided)')
    parser.add_argument('--region-start', '-s', type=int, default=21000000, help='Start position for visualization (default: 21Mb)')
    parser.add_argument('--region-end', '-e', type=int, default=23000000, help='End position for visualization (default: 23Mb)')
    parser.add_argument('--threshold', '-t', type=float, default=0.05, 
                        help='Threshold to filter out small values for predicted matrices (default: 0.05)')
    parser.add_argument('--overlap-method', '-m', choices=['mean', 'max'], default='mean',
                        help='Method to handle overlapping regions for predicted matrices (default: mean)')
    parser.add_argument('--maxapart', type=int, default=2000000,
                        help='Maximum genomic distance between two loci (default: 2000000)')
    
    # Matrix balancing parameters
    balance_group = parser.add_argument_group('Matrix balancing parameters')
    balance_group.add_argument('--no-balance', action='store_true', help='Skip matrix balancing')
    
    
    # HiCCUPS specific parameters
    hiccups_group = parser.add_argument_group('HiCCUPS parameters')
    hiccups_group.add_argument('--peak-width', '-pw', type=int, nargs='+', default=[1, 2, 4], 
                              help='Peak width(s) for loop calling (default: 1 2 4)')
    hiccups_group.add_argument('--window-width', '-ww', type=int, nargs='+', default=[3, 5, 7], 
                              help='Window width(s) for loop calling (default: 3 5 7)')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Set up logging
    logger = setup_logging()
    
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get or generate chromosome sizes file
    if args.chrom_sizes:
        chrom_sizes_file = args.chrom_sizes
        if not os.path.exists(chrom_sizes_file):
            logger.error(f"Provided chromosome sizes file {chrom_sizes_file} does not exist")
            return 1
    else:
        chrom_sizes_file = os.path.join(args.output_dir, f"{args.genome}.chrom.sizes")
        if not os.path.exists(chrom_sizes_file):
            logger.info(f"Fetching chromosome sizes for {args.genome}")
            if not fetch_chrom_sizes(args.genome, chrom_sizes_file):
                logger.error("Failed to fetch chromosome sizes")
                return 1
    
    logger.info("Starting chromatin loop identification pipeline with pyHICCUPS")
    
    # Process input based on type
    cool_file = None
    
    if args.hic:
        # Process HiC file
        if not os.path.exists(args.hic):
            logger.error(f"Input HiC file {args.hic} does not exist")
            return 1
        
        coolname = os.path.basename(args.hic).split('.')[0]
        cool_file = os.path.join(args.output_dir, coolname + f'_res{args.resolution}.cool')
        
        # Check if the cool file already exists and is balanced
        if os.path.exists(cool_file):
            import cooler
            try:
                # Try to open the cool file and check if it has balanced weights
                clr = cooler.Cooler(cool_file)
                if 'weight' in clr.bins().columns:
                    logger.info(f"Using existing balanced cool file: {cool_file}")
                else:
                    # File exists but not balanced
                    if not args.no_balance:
                        logger.info(f"Existing cool file found but not balanced. Balancing matrix in {cool_file}")
                        balance_kwargs = {}
                        success = balance_matrix(cool_file, **balance_kwargs)
                        if not success:
                            logger.warning("Matrix balancing failed, continuing with analysis using unbalanced matrix")
                    else:
                        logger.info("Using existing unbalanced cool file and skipping balancing as requested")
            except Exception as e:
                logger.warning(f"Error checking existing cool file: {str(e)}. Will recreate the file.")
                # If error occurred during checking, recreate the file
                logger.info(f"Converting {args.hic} to {cool_file}")
                success = convert_hic_to_cool(args.hic, cool_file, args.resolution, num_proc=4)
                if not success:
                    logger.error("Conversion from hic to cool format failed")
                    return 1
                
                # Balance if needed
                if not args.no_balance:
                    logger.info(f"Balancing matrix in {cool_file}")
                    balance_kwargs = {}
                    success = balance_matrix(cool_file, **balance_kwargs)
                    if not success:
                        logger.warning("Matrix balancing failed, continuing with analysis using unbalanced matrix")
        else:
            # Cool file doesn't exist, create it
            logger.info(f"Converting {args.hic} to {cool_file}")
            success = convert_hic_to_cool(args.hic, cool_file, args.resolution, num_proc=4)
            if not success:
                logger.error("Conversion from hic to cool format failed")
                return 1
            
            # Balance if needed
            if not args.no_balance:
                logger.info(f"Balancing matrix in {cool_file}")
                balance_kwargs = {}
                success = balance_matrix(cool_file, **balance_kwargs)
                if not success:
                    logger.warning("Matrix balancing failed, continuing with analysis using unbalanced matrix")
            else:
                logger.info("Matrix balancing skipped as requested")
        
    elif args.pred_dir:
        # Process predicted matrices
        if not os.path.exists(args.pred_dir):
            logger.error(f"Input directory {args.pred_dir} does not exist")
            return 1
        
        # Process chromosome
        logger.info(f"Processing predicted matrices for {args.chromosome}")
        matrix_cool_dir = os.path.join(args.output_dir, "predicted_cool")
        os.makedirs(matrix_cool_dir, exist_ok=True)
        
        # Define expected cool file path
        expected_cool_file = os.path.join(matrix_cool_dir, f"{args.chromosome}_{args.resolution}.cool")
        
        # Check if the cool file already exists and is balanced
        if os.path.exists(expected_cool_file):
            import cooler
            try:
                # Try to open the cool file and check if it has balanced weights
                clr = cooler.Cooler(expected_cool_file)
                if 'weight' in clr.bins().columns:
                    logger.info(f"Using existing balanced cool file: {expected_cool_file}")
                    cool_file = expected_cool_file
                else:
                    # File exists but not balanced
                    if not args.no_balance:
                        logger.info(f"Existing cool file found but not balanced. Balancing matrix in {expected_cool_file}")
                        balance_kwargs = {}
                        success = balance_matrix(expected_cool_file, **balance_kwargs)
                        if not success:
                            logger.warning("Matrix balancing failed, continuing with analysis using unbalanced matrix")
                        cool_file = expected_cool_file
                    else:
                        logger.info("Using existing unbalanced cool file and skipping balancing as requested")
                        cool_file = expected_cool_file
            except Exception as e:
                logger.warning(f"Error checking existing cool file: {str(e)}. Will recreate the file.")
                # If error occurred during checking, recreate the file
                cool_file = process_chromosome(
                    args.pred_dir, args.chromosome, args.resolution, matrix_cool_dir, 
                    chrom_sizes_file, threshold=args.threshold, overlap_method=args.overlap_method
                )
                
                if not cool_file:
                    logger.error(f"Failed to process predicted matrices for {args.chromosome}")
                    return 1
                
                # Balance if needed
                if not args.no_balance:
                    logger.info(f"Balancing matrix in {cool_file}")
                    balance_kwargs = {}
                    success = balance_matrix(cool_file, **balance_kwargs)
                    if not success:
                        logger.warning("Matrix balancing failed, continuing with analysis using unbalanced matrix")
        else:
            # Cool file doesn't exist, create it
            cool_file = process_chromosome(
                args.pred_dir, args.chromosome, args.resolution, matrix_cool_dir, 
                chrom_sizes_file, threshold=args.threshold, overlap_method=args.overlap_method
            )
            
            if not cool_file:
                logger.error(f"Failed to process predicted matrices for {args.chromosome}")
                return 1
                
            # Balance if needed
            if not args.no_balance:
                logger.info(f"Balancing matrix in {cool_file}")
                balance_kwargs = {}
                success = balance_matrix(cool_file, **balance_kwargs)
                if not success:
                    logger.warning("Matrix balancing failed, continuing with analysis using unbalanced matrix")
            else:
                logger.info("Matrix balancing skipped as requested")
    
    # Run pyHICCUPS to identify chromatin loops
    logger.info(f"Identifying chromatin loops for {args.chromosome} using pyHICCUPS")
    loops_dir = os.path.join(args.output_dir, "loops")
    os.makedirs(loops_dir, exist_ok=True)
    
    # Prepare chromosome format for file naming
    chrom_for_file = args.chromosome.replace('chr', '')
    expected_loops_file = os.path.join(loops_dir, f"{chrom_for_file}_hiccups_loops.bedpe")
    
    # Check if the loops file already exists
    if os.path.exists(expected_loops_file):
        # Skip loop calling if loops file already exists
        logger.info(f"Using existing loops file: {expected_loops_file}")
        loops_file = expected_loops_file
    else:
        # If loops file doesn't exist, run HiCCUPS to identify loops
        loops_file = run_hiccups(
            cool_file, args.chromosome, loops_dir, args.resolution, 
            args.peak_width, args.window_width, args.maxapart
        )
        
        if not loops_file:
            logger.error("Loop calling with pyHICCUPS failed")
            return 1
    
    # Visualize identified loops (2Mb-4Mb range by default)
    logger.info(f"Visualizing identified loops for {args.chromosome} in the {args.region_start/1000000:.1f}Mb-{args.region_end/1000000:.1f}Mb range")
    visual_dir = os.path.join(args.output_dir, "visualization")
    visual_chrom = args.chromosome
    if args.hic:
        visual_chrom = args.chromosome.replace('chr', '')        
    visual_file = visualize_loops(cool_file, loops_file, visual_chrom, visual_dir, 
                                 args.resolution, args.region_start, args.region_end)
    
    if not visual_file:
        logger.warning("Loop visualization failed, but continuing with analysis")
    
    # Run aggregate peak analysis
    logger.info("Running aggregate peak analysis")
    apa_dir = os.path.join(args.output_dir, "apa")
    apa_file = run_apa_analysis(cool_file, loops_file, apa_dir, args.resolution)
    
    if not apa_file:
        logger.warning("APA analysis failed")
    
    # Completed
    logger.info("Chromatin loop identification pipeline completed successfully")
    logger.info(f"Results saved to {args.output_dir}")
    
    # Output summary
    print("\n=== Pipeline Summary ===")
    if args.hic:
        print(f"Input HiC file: {args.hic}")
    else:
        print(f"Input predicted matrices directory: {args.pred_dir}")
    print(f"Chromosome: {args.chromosome}")
    print(f"Resolution: {args.resolution}")
    print(f"Matrix balancing: {'No' if args.no_balance else 'Yes'}")
    
    print(f"Loop calling algorithm: pyHICCUPS")
    print(f"Peak width values: {args.peak_width}")
    print(f"Window width values: {args.window_width}")
    print(f"Maximum distance: {args.maxapart} bp")
    
    if os.path.exists(loops_file):
        # Count identified loops
        with open(loops_file, 'r') as f:
            loop_count = sum(1 for line in f if not line.startswith('#'))
        print(f"Identified loops: {loop_count}")
        print(f"Loops file: {loops_file}")
    
    if os.path.exists(visual_file):
        print(f"Visualization: {visual_file} (region: {args.region_start/1000000:.1f}Mb-{args.region_end/1000000:.1f}Mb)")
    
    if os.path.exists(apa_file):
        print(f"APA analysis: {apa_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())