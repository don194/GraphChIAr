import sys
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import logging
import os
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
torch.set_float32_matmul_precision('medium')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
logging.basicConfig(level=logging.INFO)

import gchia.Model.Model as Model 
from gchia.Dataset.dataset_normalization import HiCCTCFDataset

logging.getLogger('numexpr').setLevel(logging.WARNING)

def main():
    args = init_parser()
    init_training(args)

def init_parser():
    parser = argparse.ArgumentParser(description='Train a model to predict ChIA-PET interactions')
    # Data and Run Directories
    parser.add_argument('--seed', dest='run_seed', default=42, type=int, 
                        help='Seed for training')
    parser.add_argument('--save_path', dest='save_path', default='checkpoints', type=str, 
                        help='Path to save model checkpoints')
    # Model Parameters
    parser.add_argument('--model', dest='model', default='ChIAPETMatrixPredictor', type=str,
                        help='Model to train')
    # Data directories
    parser.add_argument('--data-root', dest='data_root', default='data', type=str, 
                        help='Root directory for data', required=True)
    parser.add_argument('--celltype', dest='celltype', default='GM12878', type=str, 
                        help='Cell type to train on', required=True)
    parser.add_argument('--hic-file', dest='hic_file', default=None, type=str,
                        help='Hi-C file')
    parser.add_argument('--chia-pet-file', dest='chia_pet_file', default=None, type=str,
                        help='ChIA-PET file')
    parser.add_argument('--target-type', dest='target_type', default='CTCF_ChIA-PET', type=str,
                        help='Target type for ChIA-PET file')
    parser.add_argument('--ctcf-file', dest='ctcf_file', default=None, type=str,
                        help='CTCF BigWig or H5 file')
    parser.add_argument('--chr-sizes-file', dest='chr_sizes_file', default=None, type=str,
                        help='Chromosome sizes file')
    # Training Parameters
    parser.add_argument('--lr', dest='trainer_lr', default=2e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--patience', dest='trainer_patience', default=80, type=int,
                        help='Early stopping patience')
    parser.add_argument('--max_epochs', dest='trainer_max_epochs', default=500, type=int,
                        help='Maximum number of epochs to train')
    parser.add_argument('--save_top_k', dest='trainer_save_top_k', default=20, type=int,
                        help='Number of best models to save')
    # Dataset Parameters
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
    # Add chipseq-files argument to parser
    parser.add_argument('--chipseq-files', dest='chipseq_files', default=[], type=str, nargs='+',
                        help='List of ChIP-seq feature names or file paths')


    # Dataloader Parameters
    parser.add_argument('--batch_size', dest='dataloader_batch_size', default=64, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', dest='dataloader_num_workers', default=8, type=int,
                        help='Number of workers for dataloader')
    
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    return args

def init_training(args):
    # Early stopping callback
    early_stop_callback = callbacks.EarlyStopping(
        monitor='avg_val_loss',
        patience=args.trainer_patience,
        verbose=True,
        mode='min'
    )
    # Checkpoints
    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor='avg_val_loss',
        dirpath=f'{args.save_path}/model-{args.celltype}',
        save_top_k=args.trainer_save_top_k,
        mode='min'
    )
    
    # LR monitor
    lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')
    
    # Logger
    csv_logger = pl.loggers.CSVLogger(f'{args.save_path}/logs')
    all_loggers = csv_logger
    
    # assign seed
    pl.seed_everything(args.run_seed, workers=True)
    
    pl_module = TrainModule(args)
    pl_trainer = pl.Trainer(accelerator="gpu",
                            devices=1,
                            max_epochs=args.trainer_max_epochs,
                            callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
                            logger=all_loggers,
                            log_every_n_steps=1,
                            deterministic=True
                        )
    trainloader = pl_module.get_dataloader(args, 'train')
    valloader = pl_module.get_dataloader(args, 'val')
    testloader = pl_module.get_dataloader(args, 'test')
    pl_trainer.fit(pl_module, trainloader, valloader)
    
class TrainModule(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = self.get_model(args)
        self.save_hyperparameters()

    def forward(self,x):
        return self.model(x)
        
    def _calculate_loss(self, pred_matrix, batch):
        """
        Calculate MSE loss between predicted matrix and ground truth matrix after converting COO to dense format.
        
        Args:
            pred_matrix: Predicted matrix from the model
            batch: Batch object containing ground truth matrix in COO format
        
        Returns:
            torch.Tensor: Average loss across all graphs in the batch
        """
        batch_matrix = batch.batch
        total_loss = 0
        loss_fn = torch.nn.MSELoss()

        for graph_idx in torch.unique(batch_matrix):
            # Get the predicted matrix for current graph
            graph_pred = pred_matrix[graph_idx]
            
            # Get the ground truth sparse matrix for current graph and convert to dense
            graph_y = batch.y[graph_idx]
            n = graph_pred.size(0)  # matrix size
            
            # Create upper triangular mask
            triu_mask = torch.triu(torch.ones(n, n), diagonal=1).bool().to(graph_pred.device)
            
            # Apply mask to both predicted and ground truth matrices
            graph_pred_triu = graph_pred[triu_mask].to(graph_pred.device)
            dense_y_triu = graph_y[triu_mask].to(graph_pred.device)
            
            # Calculate loss for upper triangular part only
            loss = loss_fn(graph_pred_triu, dense_y_triu)
            total_loss += loss
            
        num_graphs = len(torch.unique(batch_matrix))
        return total_loss / num_graphs
    
    def training_step(self, batch, batch_idx):
        pred_matrix = self(batch)
        
        loss= self._calculate_loss(pred_matrix, batch)
        
        self.log('train_loss', loss, batch_size=batch.size(0), prog_bar=True)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        pred_matrix = self(batch)
        
        loss= self._calculate_loss(pred_matrix, batch)
        
        self.log_dict({'val_loss': loss}, batch_size=batch.size(0), prog_bar=True)
        return {'val_loss': loss}
    
    def test_step(self, batch, batch_idx):
        pred_matrix = self(batch)
        loss= self._calculate_loss(pred_matrix, batch)
        
        self.log_dict({'test_loss': loss}, batch_size=batch.size(0), prog_bar=True)
        return {'test_loss': loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss, prog_bar=True)
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr = self.args.trainer_lr,
                                     weight_decay = 0)
       
        # import pl_bolts
        # scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=self.args.trainer_max_epochs)

        warmup_epochs = 10
        max_epochs = self.args.trainer_max_epochs
        
        def lambda_lr(epoch):
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs))
            else:
                return 0.5 * (1.0 + np.cos(np.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
        
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'avg_val_loss',
            'strict': True,
            'name': 'WarmupCosineAnnealing',
        }
        return {'optimizer' : optimizer, 'lr_scheduler' : scheduler_config}
    
    def get_dataset(self, args, mode):
        """
        Create and return dataset for the specified mode
        
        Args:
            args: Command line arguments
            mode: Dataset mode ('train', 'val', 'test')
            
        Returns:
            HiCCTCFDataset: Dataset for the specified mode
        """
        
        log1p_bool = args.log1p.lower() == 'true'
        
        # Use the processed data root
        celltype_root = f'{args.data_root}/{args.celltype}'
        step = args.step_size
        if mode == 'test':
            if(args.window_size >= 500000):
                step = args.window_size // 4
        else :
            step = args.step_size
        # Process multiple ChIP-seq files
        chipseq_list = []
        
        # Process chipseq-files argument if provided
        if hasattr(args, 'chipseq_files') and args.chipseq_files:
            for chipseq_name in args.chipseq_files:
                # Check if it's a full path or just a feature name
                if os.path.isfile(chipseq_name):
                    # It's a full path
                    chipseq_list.append(chipseq_name)
                    logging.info(f"Added ChIP-seq file from path: {chipseq_name}")
                else:
                    # It's a feature name, look for processed h5 file
                    # Fix directory name to match script's expectation (CTCFChIP-seq_processed)
                    chipseq_h5 = os.path.join(args.data_root, args.celltype, 
                                             "ChIP-seq_processed", 
                                             f"{args.celltype}_{chipseq_name}.h5")
                    
                    if os.path.exists(chipseq_h5):
                        chipseq_list.append(chipseq_h5)
                        logging.info(f"Added ChIP-seq file: {chipseq_h5}")
                    else:
                        logging.warning(f"Could not find h5 file for {chipseq_name}, skipping.")

        # For backward compatibility, add CTCF file if provided and not already in the list
        if hasattr(args, 'ctcf_file') and args.ctcf_file and not any('ctcf' in os.path.basename(f).lower() for f in chipseq_list):
            ctcf_file = args.ctcf_file
            if not ctcf_file.endswith('.h5') and os.path.exists(f"{args.data_root}/{args.celltype}/ChIP-seq_processed"):
                # Try to find the corresponding processed file
                ctcf_basename = os.path.basename(ctcf_file).replace('.bw', '.h5')
                processed_ctcf = os.path.join(args.data_root, args.celltype, "ChIP-seq_processed", ctcf_basename)
                if os.path.exists(processed_ctcf):
                    logging.info(f"Using processed CTCF file: {processed_ctcf}")
                    ctcf_file = processed_ctcf
            
            chipseq_list.append(ctcf_file)
            logging.info(f"Added CTCF file: {ctcf_file}")
        
        logging.info(f"Using ChIP-seq files: {chipseq_list}")
        # Create directories if they don't exist
        os.makedirs(celltype_root, exist_ok=True)
        
        # Initialize dataset with proper parameters
        dataset_args = {
            'root': celltype_root,
            'hic_file': args.hic_file,
            'chipseq_files': chipseq_list,  # Pass the ChIP-seq files list
            'chia_pet_file': args.chia_pet_file,
            'chr_sizes': args.chr_sizes_file,
            'mode': mode,
            'step_size': step,
            'window_size': args.window_size,
            'resolution': args.resolution,
            'normalization': args.normalize,
            'log1p': log1p_bool,
            'target_type': args.target_type
            
        }
        
        # Add hic_resolution if specified
        if hasattr(args, 'hic_resolution') and args.hic_resolution is not None:
            dataset_args['hic_resolution'] = args.hic_resolution
        
        dataset = HiCCTCFDataset(**dataset_args)
        
        # Record length for printing validation image
        if mode == 'val':
            self.val_length = len(dataset) / args.dataloader_batch_size
            logging.info(f'Validation dataset length: {len(dataset)}, batches: {self.val_length}')

        return dataset
    
    def get_dataloader(self, args, mode):
        dataset = self.get_dataset(args, mode)
        # shuffle = False
        if mode == 'train':
            shuffle = True
        else:
            shuffle = False
        batch_size = args.dataloader_batch_size
        num_workers = args.dataloader_num_workers
        dataloader = GraphDataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=num_workers)
        return dataloader
    
    def get_model(self, args):
        model_name = args.model
        ModelClass = getattr(Model, model_name)
        
        # Calculate feature dimension based on number of ChIP-seq files
        feature_dim = 0  # Default feature dimension
        
        # Count chipseq files if available
        if hasattr(args, 'chipseq_files') and args.chipseq_files:
            feature_dim += len(args.chipseq_files)
            print(f"Feature dim: {feature_dim}")
        # Add CTCF file for backward compatibility if it's not already counted in chipseq_files
        elif hasattr(args, 'ctcf_file') and args.ctcf_file:
            feature_dim += 1
        
        # Create model with appropriate feature dimension
        model = ModelClass(
            resolution=args.resolution, 
            window_size=args.window_size,
            feature_dim=feature_dim  # Pass the calculated feature dimension
        )
        return model
        
if __name__ == '__main__':
    main()



