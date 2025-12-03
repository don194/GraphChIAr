import torch
import pytorch_lightning as pl
import os
import sys
import logging
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
from gchia.Train.train import TrainModule
import argparse
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def separate_pred_matrix(pred_matrix, batch, offset_val = 0):
    """
    Separate the predicted matrix by individual graphs
    
    Args:
        pred_matrix: Matrix predicted by the model
        batch: Data object containing batch information
        
    Returns:
        list of tuple: Each element contains (prediction matrix, chrom, start, end) for a single graph
    """
    separated_data = []
    num_graphs = batch.num_graphs

    for i in range(num_graphs):
        graph_matrix = pred_matrix[i]
        
        if offset_val > 0:
            coords_A = batch.anchor_A_coords[i]  
            coords_B = batch.anchor_B_coords[i]  
            coords = {
                'chrom': coords_A[0],
                'start_A': coords_A[1],  
                'end_A': coords_A[2],
                'start_B': coords_B[1],
                'end_B': coords_B[2]
            }
        else:
            coords = {
                'chrom': batch.chrom[i],
                'start': batch.start[i].item(),
                'end': batch.end[i].item()
            }
        
        separated_data.append((graph_matrix, coords))
    return separated_data

def load_model_and_predict(checkpoint_path, args):
    """
    Load trained model and make predictions on test set
    
    Args:
        checkpoint_path: Path to model checkpoint
        args: Configuration parameters matching those used during training
        
    Returns:
        list of list of tuple: Prediction matrices and location info for each graph in each batch
    """
    # Handle chipseq-files parameter for prediction
    if not hasattr(args, 'chipseq_files'):
        args.chipseq_files = []
    
    # Calculate feature dimension based on ChIP-seq files
    feature_dim = 0
    if args.chipseq_files:
        feature_dim += len(args.chipseq_files)
    # Add CTCF file for backward compatibility if specified
    elif hasattr(args, 'ctcf_file') and args.ctcf_file:
        feature_dim += 1
        
    logger.info(f"Using feature dimension: {feature_dim}")
    
    # Handle backward compatibility for CTCF file
    if hasattr(args, 'ctcf_file') and args.ctcf_file:
        # Check for processed CTCF file
        if not args.ctcf_file.endswith('.h5'):
            ctcf_processed_dir = f"{args.data_root}/{args.celltype}/ChIP-seq_processed"
            if os.path.exists(ctcf_processed_dir):
                ctcf_basename = os.path.basename(args.ctcf_file).replace('.bw', '.h5')
                processed_ctcf = os.path.join(ctcf_processed_dir, ctcf_basename)
                if os.path.exists(processed_ctcf):
                    logger.info(f"Using processed CTCF file: {processed_ctcf}")
                    args.ctcf_file = processed_ctcf
    
    # 1. Load model
    model = TrainModule.load_from_checkpoint(
        checkpoint_path,
        args=args,
        strict=False
    )
    model.eval()
    
    # 2. Get test data loader
    test_loader = model.get_dataloader(args, mode='test')
    
    # 3. Make predictions on test set
    all_batch_predictions = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    logger.info(f"Starting predictions using device: {device}")
    logger.info(f"Total batches to process: {len(test_loader)}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            pred_matrix = model(batch)
            
            # Separate prediction matrices by graph and get location info
            separated_data = separate_pred_matrix(pred_matrix, batch,args.offset)
            
            # Move matrices to CPU
            separated_data_cpu = [(matrix.cpu(), coords) for matrix, coords in separated_data]
            
            all_batch_predictions.append(separated_data_cpu)
            
            if batch_idx % 10 == 0:  # Print progress every 10 batches
                logger.info(f"Processed batch {batch_idx}/{len(test_loader)}")

    logger.info(f"Completed predictions for {len(test_loader)} batches")
    return all_batch_predictions


def save_predictions(all_batch_predictions, save_dir, offset_val = 0):
    """
    Save prediction results with chromosome information in filenames
    
    Args:
        all_batch_predictions: Prediction results and location info from all batches
        save_dir: Directory to save the results
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving predictions to directory: {save_dir}")
    
    # Count total predictions
    total_predictions = sum(len(batch) for batch in all_batch_predictions)
    saved_count = 0
    
    # Flatten predictions from all batches
    for batch_predictions in all_batch_predictions:
        for matrix, coords in batch_predictions:
            matrix_np = matrix.numpy()
            
            
            if offset_val > 0:
                filename = f"pred_matrix_{coords['chrom']}_{coords['start_A']}-{coords['end_A']}_vs_{coords['start_B']}-{coords['end_B']}.npy"
            else:
                filename = f'pred_matrix_{coords["chrom"]}_{int(coords["start"])}_{int(coords["end"])}.npy'

            save_path = os.path.join(save_dir, filename)
            np.save(save_path, matrix_np)
    
    logger.info("Successfully saved prediction matrices")


def main():
    parser = argparse.ArgumentParser(description='Predict ChIA-PET interactions using trained model')
    # Add required parameters matching those used during training
    parser.add_argument('--offset', dest='offset', default=0, type=int,
                        help='Offset value for prediction windows')
    parser.add_argument('--data-root', required=True, type=str,
                        help='Root directory for processed data')
    parser.add_argument('--celltype', required=True, type=str,
                        help='Cell type to predict on')
    parser.add_argument('--hic-file', required=True, type=str,
                        help='Path to Hi-C file')
    parser.add_argument('--chia-pet-file', required=True, type=str,
                        help='Path to ChIA-PET file')
    parser.add_argument('--chr-sizes-file', required=True, type=str,
                        help='Path to chromosome sizes file')
    parser.add_argument('--target-type', dest='target_type', default='CTCF_ChIA-PET', type=str,
                        help='Target type for ChIA-PET file')
    parser.add_argument('--ctcf-file', default=None, type=str,
                        help='Path to CTCF BigWig or H5 file (for backward compatibility)')
    parser.add_argument('--chipseq-files', dest='chipseq_files', default=[], type=str, nargs='+',
                        help='List of ChIP-seq feature names or file paths')
    parser.add_argument('--model', default='ChIAPETMatrixPredictor', type=str,
                        help='Model architecture to use')
    parser.add_argument('--checkpoint-path', required=True, type=str,
                        help='Path to the model checkpoint')
    parser.add_argument('--save-dir', required=True, type=str,
                        help='Directory to save prediction results')
    # Dataloader parameters
    parser.add_argument('--batch_size', dest='dataloader_batch_size', default=16, type=int,
                        help='Batch size for prediction')
    parser.add_argument('--num_workers', dest='dataloader_num_workers', default=4, type=int,
                        help='Number of workers for dataloader')
    
    # dataset parameters
    parser.add_argument('--step_size', dest='step_size', default=500000, type=int,
                        help='Step size for sliding window')
    parser.add_argument('--window_size', dest='window_size', default=2000000, type=int,
                        help='Window size for Hi-C matrix')
    parser.add_argument('--resolution', dest='resolution', default=10000, type=int,
                        help='Resolution for pre matrix')
    parser.add_argument('--hic_resolution', dest='hic_resolution', default=None, type=int,
                        help='Resolution for Hi-C matrix')
    
    parser.add_argument('--normalize', dest='normalize', default='NONE', type=str,
                        help='Normalization method for Hi-C matrix')
    parser.add_argument('--log1p', type=str, default='True', help='log1p transform the Hi-C matrix (True/False)')
    parser.add_argument('--interpolate', dest='interpolate', default='False', type=str,
                        help='Whether to interpolate Hi-C and ChIA-PET matrices to 5x resolution (True/False)')

                        
    args = parser.parse_args()
    
    # Display configuration
    logger.info("Prediction configuration:")
    logger.info(f"  Cell type: {args.celltype}")
    logger.info(f"  Hi-C file: {args.hic_file}")
    if args.ctcf_file:
        logger.info(f"  CTCF file: {args.ctcf_file}")
    if args.chipseq_files:
        logger.info(f"  ChIP-seq files: {args.chipseq_files}")
    logger.info(f"  ChIA-PET file: {args.chia_pet_file}")
    logger.info(f"  Resolution: {args.resolution}")
    logger.info(f"  Window size: {args.window_size}")
    logger.info(f"  Step size: {args.step_size}")
    logger.info(f"  Normalization: {args.normalize}")
    logger.info(f"  Log1p transformation: {args.log1p}")
    logger.info(f"  Checkpoint: {args.checkpoint_path}")
    logger.info(f"  Save directory: {args.save_dir}")
    
    # Load model and make predictions
    all_batch_predictions = load_model_and_predict(args.checkpoint_path, args)
    
    # Save prediction results
    save_predictions(all_batch_predictions, args.save_dir, args.offset)
    
    logger.info("Prediction completed successfully")
    
if __name__ == '__main__':
    main()