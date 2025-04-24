import sys
import os
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
import logging
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.data import Data
import h5py
torch.set_float32_matmul_precision('medium')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
logging.basicConfig(level=logging.INFO)

import gchia.Model.Model as Model
from gchia.Dataset.dataset_normalization import HiCCTCFDataset

def main():
    args = init_parser()
    predict(args)

def init_parser():
    parser = argparse.ArgumentParser(description='Predict ChIA-PET interactions for specific genomic regions')
    # Input/output parameters
    parser.add_argument('--checkpoint-path', dest='checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', dest='output_dir', type=str, required=True,
                        help='Directory to save prediction results')
    parser.add_argument('--model', dest='model', default='ChIAPETMatrixPredictor', type=str,
                        help='Model architecture to use')
    
    # Data parameters
    parser.add_argument('--data-root', dest='data_root', default='data', type=str, 
                        help='Root directory for data')
    parser.add_argument('--celltype', dest='celltype', type=str, required=True,
                        help='Cell type for prediction')
    parser.add_argument('--hic-file', dest='hic_file', type=str, required=True,
                        help='Path to Hi-C file')
    parser.add_argument('--chipseq-files', dest='chipseq_files', default=[], type=str, nargs='+',
                        help='List of ChIP-seq files')
    parser.add_argument('--chia-pet-file', dest='chia_pet_file', type=str, 
                        help='Path to reference ChIA-PET file (optional for evaluation)')
    parser.add_argument('--chr-sizes-file', dest='chr_sizes_file', type=str, required=True,
                        help='Chromosome sizes file')
    parser.add_argument('--target-type', dest='target_type', default='CTCF_ChIA-PET', type=str,
                        help='Target data type')
    
    # Region parameters
    parser.add_argument('--chrom', type=str, required=True,
                        help='Chromosome to predict (e.g., chr1)')
    parser.add_argument('--start', type=int, required=True,
                        help='Start position for prediction')
    parser.add_argument('--end', type=int, required=True,
                        help='End position for prediction')
    
    # Prediction parameters
    parser.add_argument('--window_size', dest='window_size', default=2000000, type=int,
                        help='Window size for prediction')
    parser.add_argument('--step_size', dest='step_size', default=1000000, type=int,
                        help='Step size for sliding window')
    parser.add_argument('--resolution', dest='resolution', default=10000, type=int,
                        help='Resolution for prediction matrix')
    parser.add_argument('--hic_resolution', dest='hic_resolution', default=None, type=int, 
                        help='Resolution for Hi-C matrix')
    parser.add_argument('--normalize', dest='normalize', default='NONE', type=str,
                        help='Normalization method for Hi-C matrix')
    parser.add_argument('--log1p', type=str, default='True', 
                        help='Whether to apply log1p transformation (True/False)')
    parser.add_argument('--batch-size', dest='batch_size', default=4, type=int,
                        help='Batch size for inference')
    parser.add_argument('--num-workers', dest='num_workers', default=4, type=int,
                        help='Number of workers for data loading')

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    return args

def predict(args):
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    logging.info(f"Predicting ChIA-PET interactions for {args.celltype} on {args.chrom}:{args.start}-{args.end}")
    
    # Import TrainModule
    from gchia.Train.train import TrainModule
    
    # Load model directly using Lightning's method
    logging.info(f"Loading model from {args.checkpoint_path}")
    try:
        model = TrainModule.load_from_checkpoint(
            args.checkpoint_path,
            args=args,  # Pass parameters to the model
            map_location='cpu'  # Ensure initial loading to CPU
        )
        model.eval()
        logging.info("Successfully loaded model")
        
        # Extract the actual prediction model component
        model = model.model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Move model to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Process ChIP-seq features - convert feature names to file paths
    chipseq_files = []
    if args.chipseq_files:
        # Define the ChIP-seq directory based on celltype
        chipseq_dir = os.path.join(args.data_root, args.celltype, "ChIP-seq_processed")
        
        for feature_name in args.chipseq_files:
            # Construct file path using the specific naming pattern: celltype_featurename.h5
            chipseq_file = os.path.join(chipseq_dir, f"{args.celltype}_{feature_name}.h5")
            chipseq_files.append(chipseq_file)
            if os.path.exists(chipseq_file):
                logging.info(f"Found ChIP-seq file: {chipseq_file}")
            else:
                
                logging.warning(f"ChIP-seq file not found: {chipseq_file}")
    
    logging.info(f"Using {len(chipseq_files)} ChIP-seq files for prediction")
    
    # Determine the model architecture and feature dimension
    feature_dim = len(chipseq_files)
    
    # Create custom pandas DataFrame for chromosome sizes
    import pandas as pd
    chr_sizes_df = pd.DataFrame({
        'chr': [args.chrom],
        'size': [args.end + args.window_size]  # Add extra space to ensure we capture the whole region
    })
    
    # Set up parameters for the dataset
    log1p_bool = args.log1p.lower() == 'true'
    
    # Create a custom dataset for the specified region
    custom_dataset = CustomHiCDataset(
        root=args.data_root,
        hic_file=args.hic_file,
        chipseq_files=chipseq_files,  # Use the found ChIP-seq file paths
        chia_pet_file=args.chia_pet_file if hasattr(args, 'chia_pet_file') else None,
        chr_sizes_df=chr_sizes_df,
        chrom=args.chrom,
        start=args.start,
        end=args.end,
        resolution=args.resolution,
        hic_resolution=args.hic_resolution,
        window_size=args.window_size,
        step_size=args.step_size,
        log1p=log1p_bool,
        normalization=args.normalize,
        target_type=args.target_type
    )
    
    # Create DataLoader for inference
    dataloader = GraphDataLoader(
        custom_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Run predictions
    logging.info("Starting predictions...")
    predictions = []
    coordinates = []
    
    with torch.no_grad():
        total_batches = len(dataloader)
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            pred = model(batch)
            
            # Store predictions and corresponding coordinates
            for i in range(pred.size(0)):
                predictions.append(pred[i].cpu().numpy())
                coordinates.append({
                    'chrom': batch.chrom[i],
                    'start': batch.start[i].item(),
                    'end': batch.end[i].item()
                })
            
            logging.info(f"Processed batch {batch_idx+1}/{total_batches}")
    
    # Save predictions as individual .npy files (similar to test.py format)
    logging.info(f"Saving {len(predictions)} prediction matrices to {args.output_dir}")
    output_dir = os.path.join(args.output_dir, f'{args.celltype}/predictions')
    os.makedirs(output_dir, exist_ok=True)
    # Save each prediction as separate .npy file
    for i, (pred, coord) in enumerate(zip(predictions, coordinates)):
        chrom = coord['chrom']
        start = coord['start']
        end = coord['end']
        
        # Create filename with location information
        filename = f"pred_matrix_{chrom}_{int(start)}_{int(end)}.npy"
        file_path = os.path.join(output_dir, filename)
        
        # Save as numpy array
        np.save(file_path, pred)
        
        if (i + 1) % 10 == 0 or (i + 1) == len(predictions):
            logging.info(f"Saved {i+1}/{len(predictions)} prediction matrices")
    
    # Optionally evaluate if the reference ChIA-PET file was provided
    if args.chia_pet_file and os.path.exists(args.chia_pet_file):
        logging.info("Evaluating predictions against reference ChIA-PET data...")
        try:
            from scipy.stats import spearmanr, pearsonr
            
            # Calculate average correlation across all windows
            all_pearson = []
            all_spearman = []
            
            for pred, coord in zip(predictions, coordinates):
                # Get reference data for this window
                ref_data = custom_dataset._read_hic_matrix(
                    args.chia_pet_file, 
                    coord['chrom'], 
                    coord['start'], 
                    coord['end'], 
                    args.resolution
                )
                
                # Flatten matrices for correlation
                pred_flat = pred.flatten()
                ref_flat = ref_data.flatten()
                
                # Calculate correlations
                pearson_corr, _ = pearsonr(pred_flat, ref_flat)
                spearman_corr, _ = spearmanr(pred_flat, ref_flat)
                
                all_pearson.append(pearson_corr)
                all_spearman.append(spearman_corr)
            
            # Average correlations
            avg_pearson = np.mean(all_pearson)
            avg_spearman = np.mean(all_spearman)
            
            logging.info(f"Average Pearson correlation: {avg_pearson:.4f}")
            logging.info(f"Average Spearman correlation: {avg_spearman:.4f}")
            
            # Save evaluation results
            eval_results = {
                'window_pearson': all_pearson,
                'window_spearman': all_spearman,
                'average_pearson': avg_pearson,
                'average_spearman': avg_spearman
            }
            
            eval_file = os.path.join(args.output_dir, f"{args.celltype}_{args.chrom}_{args.start}_{args.end}_evaluation.npy")
            np.save(eval_file, eval_results)
            logging.info(f"Evaluation results saved to {eval_file}")
            
        except Exception as e:
            logging.error(f"Failed to evaluate predictions: {e}")
    
    logging.info("Prediction completed")

class CustomHiCDataset(HiCCTCFDataset):
    def __init__(self, root, hic_file, chipseq_files, chia_pet_file, chr_sizes_df, 
                 chrom, start, end, resolution, window_size, step_size, target_type,
                 hic_resolution=None, log1p=True, normalization='NONE'):
        """
        Custom dataset for predicting on a specific genomic region.
        
        Args:
            chr_sizes_df (pd.DataFrame): DataFrame with chromosome sizes
            chrom (str): Chromosome to predict
            start (int): Start position
            end (int): End position
            Other parameters are same as HiCCTCFDataset
        """
        # Store region-specific parameters
        self.target_chrom = chrom
        self.target_start = start
        self.target_end = end
        
        # Custom mode handling
        mode = 'custom'
        
        # Store chr_sizes_df for later use in filter_chr_sizes
        self._chr_sizes_df = chr_sizes_df
        
        # Handle chia_pet_file being None
        actual_chia_pet_file = chia_pet_file or ""  # Use empty string if None
        
        # Initialize windows list BEFORE super().__init__() call
        self.windows = []
        for window_start in range(start, end, step_size):
            window_end = window_start + window_size
            if window_end > end:
                continue
            self.windows.append((chrom, window_start, window_end))
        
        logging.info(f"Created dataset with {len(self.windows)} windows for {chrom}:{start}-{end}")
        
        # Now call parent's __init__ after windows are defined
        super().__init__(
            root=root,
            hic_file=hic_file,
            chipseq_files=chipseq_files,
            chia_pet_file=actual_chia_pet_file,
            chr_sizes=None,
            mode=mode,
            resolution=resolution,
            step_size=step_size,
            window_size=window_size,
            hic_resolution=hic_resolution,
            target_type=target_type,
            log1p=log1p,
            normalization=normalization
        )
        
        # Store the original chia_pet_file value for later use
        self.original_chia_pet_file = chia_pet_file
    
    def filter_chr_sizes(self, chr_sizes):
        """
        Override the filter_chr_sizes method to use the provided DataFrame
        instead of loading from a file.
        
        Args:
            chr_sizes: This parameter is ignored in our override
            
        Returns:
            The pre-stored chromosome sizes DataFrame
        """
        # Return the DataFrame provided in the constructor
        return self._chr_sizes_df
    
    @property
    def processed_file_names(self):
        # Now windows will be defined before this is called
        return [f"tmp_{i}.pt" for i in range(len(self.windows))]
    
    def process(self):
        # Override to avoid processing and saving files
        pass
    
    def len(self):
        return len(self.windows)
    
    def get(self, idx):
        """
        Get data for a specific window without saving to disk
        """
        chrom, start, end = self.windows[idx]
        
        # Read Hi-C matrix
        hic_matrix = self._read_hic_matrix(self.hic_file, chrom, start, end, self.hic_resolution)
        fold = self.hic_resolution // self.resolution if self.hic_resolution else 1
        
        if fold > 1:
            hic_matrix = self._interpolate_matrix(hic_matrix, fold)
        
        # Try to read target matrix if available (for evaluation)
        # Use original_chia_pet_file to check for None
        if self.original_chia_pet_file and os.path.exists(self.original_chia_pet_file):
            try:
                target_matrix = self._read_hic_matrix(self.original_chia_pet_file, chrom, start, end, self.resolution)
                target_matrix = torch.from_numpy(target_matrix).float()
                target_matrix = target_matrix.unsqueeze(0)
            except Exception:
                # If target file can't be read, create dummy target
                num_bins = self.window_size // self.resolution
                target_matrix = torch.zeros((1, num_bins, num_bins), dtype=torch.float)
        else:
            # If no target file, create dummy target
            num_bins = self.window_size // self.resolution
            target_matrix = torch.zeros((1, num_bins, num_bins), dtype=torch.float)
        
        # Convert Hi-C matrix to graph format
        edge_index, edge_attr = self._convert_to_graph(hic_matrix)
        
        # Create placeholder node features
        num_nodes = self.window_size // self.resolution
        node_features = torch.zeros((num_nodes, 1), dtype=torch.float)
        
        # Create Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=target_matrix,
            chrom=chrom,
            start=start,
            end=end
        )
        
        # Read ChIP-seq features if available
        features = []
        for chipseq_file in self.file_list:
            try:
                chipseq_signal = self._read_chipseq_profile(chipseq_file, chrom, start, end)
                chipseq_signal = torch.from_numpy(chipseq_signal).float().unsqueeze(0)
                features.append(chipseq_signal)
            except Exception as e:
                logging.warning(f"Failed to read ChIP-seq data from {chipseq_file}: {e}")
                # Create zero tensor for missing ChIP-seq data
                chipseq_signal = torch.zeros((1, end-start), dtype=torch.float)
                features.append(chipseq_signal)
        
        if features:
            features = torch.cat(features, dim=0)
            features = features.unsqueeze(0)
            data.features = features
        else:
            data.features = torch.zeros((0, (end-start)), dtype=torch.float)
        
        # Read sequence data using parent class method
        try:
            seq_emb = self._read_Sequence(chrom, start, end)
            seq = torch.from_numpy(seq_emb).float().unsqueeze(0).transpose(1, 2)
            data.seq = seq
        except Exception as e:
            logging.warning(f"Failed to read sequence data for {chrom}:{start}-{end}: {e}")
            # Create zero tensor for sequence data (assuming one-hot encoding of DNA bases)
            data.seq = torch.zeros((1, 5, end-start), dtype=torch.float)
        
        # Apply transform if specified
        return data if self.transform is None else self.transform(data)

if __name__ == '__main__':
    main()
    
    
    

# python src/gchia/Predict/predict.py \
#     --checkpoint-path /path/to/checkpoint.ckpt \
#     --output-dir /path/to/output \
#     --model ChIAPETMatrixPredictor \
#     --data-root /home/dh/work/gChIA/ProcessedData \
#     --celltype GM12878 \
#     --hic-file /path/to/hic.hic \
#     --chipseq-files ctcf h3k27ac h3k4me3 \  # 只需提供特征名称
#     --chia-pet-file /path/to/chiapet.hic \
#     --chr-sizes-file /path/to/chromosome_sizes.txt \
#     --chrom chr1 --start 0 --end 10000000
