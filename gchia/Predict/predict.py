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
    parser.add_argument('--offset', default=0, type=int, help='Genomic distance offset for inter-region prediction.')
    
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
            
            if os.path.exists(chipseq_file):
                chipseq_files.append(chipseq_file)
                logging.info(f"Found ChIP-seq file: {chipseq_file}")
            else:
                
                logging.warning(f"ChIP-seq file not found: {chipseq_file}")
    
    logging.info(f"Using {len(chipseq_files)} ChIP-seq files for prediction")
    
    # Determine the model architecture and feature dimension
    feature_dim = len(chipseq_files)
    
    # Create custom pandas DataFrame for chromosome sizes
    import pandas as pd
    effective_end = args.end + args.window_size + args.offset
    chr_sizes_df = pd.DataFrame({
        'chr': [args.chrom],
        'size': [effective_end]  # Add extra space to ensure we capture the whole region
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
        target_type=args.target_type,
        offset=args.offset
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
            num_graphs = batch.num_graphs
            for i in range(num_graphs):
                predictions.append(pred[i].cpu().numpy())
                if args.offset > 0:
                     start_A = batch.anchor_A_coords[1][i].item()
                     end_A = batch.anchor_A_coords[2][i].item()
                     start_B = batch.anchor_B_coords[1][i].item()
                     end_B = batch.anchor_B_coords[2][i].item()
                     coordinates.append({'chrom': batch.anchor_A_coords[0][i], 'start_A': start_A, 'end_A': end_A, 'start_B': start_B, 'end_B': end_B})
                else:
                    coordinates.append({'chrom': batch.chrom[i], 'start': batch.start[i].item(), 'end': batch.end[i].item()})

            logging.info(f"Processed batch {batch_idx+1}/{total_batches}")
    
    # Save predictions as individual .npy files (similar to test.py format)
    logging.info(f"Saving {len(predictions)} prediction matrices to {args.output_dir}")
    output_dir = os.path.join(args.output_dir, f'{args.celltype}/predictions')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each prediction as separate .npy file
    for pred, coord in zip(predictions, coordinates):
        if args.offset > 0:
            filename = f"pred_matrix_{coord['chrom']}_{coord['start_A']}-{coord['end_A']}_vs_{coord['start_B']}-{coord['end_B']}.npy"
        else:
            filename = f"pred_matrix_{coord['chrom']}_{int(coord['start'])}_{int(coord['end'])}.npy"
        np.save(os.path.join(output_dir, filename), pred)
    
    # Optionally evaluate if the reference ChIA-PET file was provided
    if args.chia_pet_file and os.path.exists(args.chia_pet_file):
        logging.info("Evaluating predictions...")
        from scipy.stats import spearmanr, pearsonr
        all_pearson, all_spearman = [], []
        
        # 【修改点 6】: 适配 offset 模式的评估数据读取
        for pred, coord in zip(predictions, coordinates):
            if args.offset > 0:
                 ref_data = custom_dataset._read_inter_hic_matrix(
                     args.chia_pet_file, coord['chrom'], coord['start_A'], coord['end_A'],
                     coord['chrom'], coord['start_B'], coord['end_B'], args.resolution)
            else:
                 ref_data = custom_dataset._read_hic_matrix(
                     args.chia_pet_file, coord['chrom'], coord['start'], coord['end'], args.resolution)

            if pred.shape != ref_data.shape:
                logging.warning(f"Shape mismatch! Pred: {pred.shape}, Ref: {ref_data.shape}. Skipping.")
                continue

            pearson_corr, _ = pearsonr(pred.flatten(), ref_data.flatten())
            spearman_corr, _ = spearmanr(pred.flatten(), ref_data.flatten())
            all_pearson.append(pearson_corr)

        if all_pearson:
            logging.info(f"Average Pearson correlation: {np.mean(all_pearson):.4f}")

    logging.info("Prediction completed")

class CustomHiCDataset(HiCCTCFDataset):
    def __init__(self, root, hic_file, chipseq_files, chia_pet_file, chr_sizes_df, 
                 chrom, start, end, resolution, window_size, step_size, target_type,offset=0,
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
        self.offset = offset
        # Handle chia_pet_file being None
        actual_chia_pet_file = chia_pet_file or ""  # Use empty string if None
        
        # Initialize windows list BEFORE super().__init__() call
        self.windows = []
        effective_window_span = window_size if offset == 0 else window_size + offset

        for window_start in range(start, end, step_size):
            window_end = window_start + effective_window_span
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
            normalization=normalization,
            offset=offset
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
        chrom, start, end = self.windows[idx]
        
        # 根据 offset 调用不同的数据处理逻辑
        if self.offset > 0:
            data = self._get_offset_item(chrom, start, end)
        else:
            data = self._get_standard_item(chrom, start, end)
        
        return data
    
    def _get_offset_item(self, chrom, start, end):
        num_bins = self.window_size // self.resolution
        start_A, end_A = start, start + self.window_size
        start_B, end_B = start + self.offset, start + self.offset + self.window_size

        aa_matrix = self._read_hic_matrix(self.hic_file, chrom, start_A, end_A, self.hic_resolution)
        bb_matrix = self._read_hic_matrix(self.hic_file, chrom, start_B, end_B, self.hic_resolution)
        ab_matrix = self._read_inter_hic_matrix(self.hic_file, chrom, start_A, end_A, chrom, start_B, end_B, self.hic_resolution)

        fold = self.hic_resolution // self.resolution if self.hic_resolution else 1
        aa_matrix = self._interpolate_matrix(aa_matrix, fold)
        bb_matrix = self._interpolate_matrix(bb_matrix, fold)
        ab_matrix = self._interpolate_matrix(ab_matrix, fold)
        
        edge_index_aa, edge_attr_aa = self._convert_to_graph(aa_matrix)
        edge_index_bb, edge_attr_bb = self._convert_to_graph(bb_matrix)
        edge_index_bb += num_bins

        rows_ab, cols_ab = np.nonzero(ab_matrix)
        values_ab = ab_matrix[rows_ab, cols_ab]
        cols_ab_global = cols_ab + num_bins
        edge_index_ab = torch.from_numpy(np.vstack([np.concatenate([rows_ab, cols_ab_global]), np.concatenate([cols_ab_global, rows_ab])])).long()
        edge_attr_ab = torch.from_numpy(np.concatenate([values_ab, values_ab])).float()

        edge_index = torch.cat([edge_index_aa, edge_index_bb, edge_index_ab], dim=1)
        edge_attr = torch.cat([edge_attr_aa, edge_attr_bb, edge_attr_ab], dim=0)
        
        region_id = torch.cat([torch.zeros(num_bins), torch.ones(num_bins)]).long()
        
        data = Data(
            edge_index=edge_index, edge_attr=edge_attr,
            ab_hic=torch.from_numpy(ab_matrix).float().unsqueeze(0),
            region_id=region_id,
            anchor_A_coords=(chrom, start_A, end_A),
            anchor_B_coords=(chrom, start_B, end_B)
        )
    
        features_A, seq_A = self._load_1d_features(chrom, start_A, end_A)
        features_B, seq_B = self._load_1d_features(chrom, start_B, end_B)
        
        data.features = torch.cat([features_A, features_B], dim=2)
        data.seq = torch.cat([seq_A, seq_B], dim=2)
        
        return data
        
    
    def _load_1d_features(self, chrom, start, end):
        features = []
        for chipseq_file in self.file_list:
            chipseq_signal = self._read_chipseq_profile(chipseq_file, chrom, start, end)
            features.append(torch.from_numpy(chipseq_signal).float().unsqueeze(0))
        
        if features:
            features = torch.cat(features, dim=0).unsqueeze(0)
        else:
            features = torch.zeros((1, 0, end-start), dtype=torch.float)
            
        seq_emb = self._read_Sequence(chrom, start, end)
        seq = torch.from_numpy(seq_emb).float().unsqueeze(0).transpose(1, 2)
        
        return features, seq

    def _get_standard_item(self, chrom, start, end):
        hic_matrix = self._read_hic_matrix(self.hic_file, chrom, start, end, self.hic_resolution)
        fold = self.hic_resolution // self.resolution if self.hic_resolution else 1
        if fold > 1:
            hic_matrix = self._interpolate_matrix(hic_matrix, fold)
        
        edge_index, edge_attr = self._convert_to_graph(hic_matrix)
        
        num_nodes = self.window_size // self.resolution
        node_features = torch.zeros((num_nodes, 1), dtype=torch.float)
        
        data = Data(
            x=node_features, edge_index=edge_index, edge_attr=edge_attr,
            chrom=chrom, start=start, end=end
        )
        
        features, seq = self._load_1d_features(chrom, start, end)
        data.features = features
        data.seq = seq
        
        return data

if __name__ == '__main__':
    main()
    
    
    

# python src/gchia/Predict/predict.py \
#     --checkpoint-path /path/to/checkpoint.ckpt \
#     --output-dir /path/to/output \
#     --model ChIAPETMatrixPredictor \
#     --data-root /home/dh/work/gChIA/ProcessedData \
#     --celltype GM12878 \
#     --hic-file /path/to/hic.hic \
#     --chipseq-files ctcf h3k27ac h3k4me3 \  
#     --chia-pet-file /path/to/chiapet.hic \
#     --chr-sizes-file /path/to/chromosome_sizes.txt \
#     --chrom chr1 --start 0 --end 10000000
