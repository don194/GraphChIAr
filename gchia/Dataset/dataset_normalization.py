import concurrent
import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
import hicstraw
import cooler  # Add cooler import
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
import pyBigWig
from sklearn.preprocessing import StandardScaler
import os
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import h5py
from scipy.interpolate import interp2d
from pathlib import Path  
class HiCCTCFDataset(Dataset):

    def __init__(self, root, hic_file, chipseq_files, chia_pet_file, chr_sizes,
                 mode = 'train' ,
                 resolution = 10000,
                 step_size = 2000000 ,
                 window_size = 2000000,
                 offset = 0,
                 hic_resolution = None,
                 target_type = 'CTCF_ChIA-PET',
                 interpolate = False,
                 log1p = True, transform = None, pre_transform = None, normalization='NONE',n_processes=None):
        """
        Initializes the dataset normalization class.

        Args:
            root (str): Root directory where the dataset is stored.
            hic_file (str): Path to the Hi-C file.
            ctcf_file (str): Path to the CTCF file.
            chia_pet_file (str): Path to the ChIA-PET file.
            chr_sizes (str): Path to the chromosome sizes file.
            mode (str, optional): Mode of operation, either 'train' or 'test'. Defaults to 'train'.
            resolution (int, optional): Resolution of the data. Defaults to 10000.
            step_size (int, optional): Step size for the sliding window. Defaults to 2000000.
            window_size (int, optional): Size of the window. Defaults to 2000000.
            offset (int, optional): Offset for the sliding window. Defaults to 0.
            transform (callable, optional): A function/transform that takes in a sample and returns a transformed version. Defaults to None.
            pre_transform (callable, optional): A function/transform that takes in a sample before any other processing. Defaults to None.
            normalization (str, optional): Type of normalization to apply. Defaults to 'NONE'.

        Attributes:
            hic_file (str): Path to the Hi-C file.
            ctcf_file (str): Path to the CTCF file.
            chia_pet_file (str): Path to the ChIA-PET file.
            resolution (int): Resolution of the data.
            step_size (int): Step size for the sliding window.
            window_size (int): Size of the window.
            offset (int): Offset for the sliding window.
            normalization (str): Type of normalization to apply.
            mode (str): Mode of operation, either 'train' or 'test'.
            chr_sizes (DataFrame): Filtered chromosome sizes.
            processed_files (list): List of processed files.
        """
        self.hic_file = hic_file
        self.file_list = chipseq_files
        self.chia_pet_file = chia_pet_file
        self.resolution = resolution
        self.hic_resolution = hic_resolution if hic_resolution is not None else resolution
        self.step_size = step_size
        self.window_size = window_size
        self.offset = offset
        self.normalization = normalization
        self.target_type = target_type
        self.log1p = log1p
        print(f'Using log1p transformation: {log1p}')
        self.mode = mode
        self.chr_sizes = self.filter_chr_sizes(chr_sizes)
        self.processed_files = []
        self.interpolate = interpolate
        # Detect file types
        self.hic_file_type = os.path.splitext(hic_file)[1].lower().replace('.', '')
        self.chia_file_type = os.path.splitext(chia_pet_file)[1].lower().replace('.', '')
        
        # Update processed directory name with file types
        if hic_resolution is not None:
            self.processed_dir_name = f'processed_{resolution}_{window_size}_{normalization}_log1p_{log1p}_hicres_{hic_resolution}_hictype_{self.hic_file_type}_chiatype_{self.chia_file_type}'
        else:
            self.processed_dir_name = f'processed_{resolution}_{window_size}_{normalization}_log1p_{log1p}_hictype_{self.hic_file_type}_chiatype_{self.chia_file_type}'
        
        # Update processed directory name with interpolate option
        if self.interpolate:
            self.processed_dir_name += '_interpolate_5x'
        if offset > 0:
            self.processed_dir_name += f'_offset_{offset}'
        self.n_processes = n_processes if n_processes is not None else max(1, cpu_count() // 2)
        repo_root = Path(__file__).resolve().parents[3]  # …/GraphChIAr
        self.seq_h5 = str(repo_root / 'ReferenceGenome' / 'hg38' / 'hg38.h5')
        super().__init__(root, transform, pre_transform)
    
    
    
    def filter_chr_sizes(self, chr_sizes):
        """
        Load and filter the chromosome sizes based on the mode.
        """
        # Load chr_sizes file into a DataFrame
        chr_sizes_df = pd.read_csv(chr_sizes, sep='\t', header=None, names=['chr', 'size'])
        
        if self.mode == 'train':
            # Exclude chromosomes 5 and 10
            chr_sizes_df = chr_sizes_df[~chr_sizes_df['chr'].isin(['chr5', 'chr10', 'chrY'])]
        elif self.mode == 'val':
            # Only include chromosome 5
            chr_sizes_df = chr_sizes_df[chr_sizes_df['chr'] == 'chr5']
        elif self.mode == 'test':
            # Only include chromosome 10
            chr_sizes_df = chr_sizes_df[chr_sizes_df['chr'] == 'chr10']
        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Choose from 'train', 'val', or 'test'.")
        
        return chr_sizes_df
    
    @property    
    def raw_file_names(self):
        return [self.hic_file, self.ctcf_file, self.chia_pet_file]
    
    @property
    def processed_file_names(self):
        if not self.processed_files:
            for _, row in self.chr_sizes.iterrows():
                chrom, size = row['chr'], row['size']
                end_limit = size
                if self.offset > 0:
                    end_limit = size - self.offset
                for start in range(0, size, self.step_size):
                    # Calculate the end position of the window
                    end = start + self.window_size
                    if end > end_limit:
                        continue
                    self.processed_files.append(f'data_{chrom}_{start}_{end}.pt')
        return self.processed_files
    @property
    def processed_dir(self):
        dir = os.path.join(self.root, self.target_type)
        return os.path.join(dir, self.processed_dir_name)
    
    def _read_inter_hic_matrix(self, hic_file, chrom, start1, end1, start2, end2, res):
        """
        Reads an inter-regional Hi-C interaction matrix.
        Supports both .hic and .cool/.mcool files.
        """
        try:
            file_ext = os.path.splitext(hic_file)[1].lower()
            num_bins1 = (end1 - start1) // res
            num_bins2 = (end2 - start2) // res

            if file_ext in ['.cool', '.mcool']:
                import cooler
                if file_ext == '.mcool':
                    clr = cooler.Cooler(f'{hic_file}::/resolutions/{res}')
                else:
                    clr = cooler.Cooler(hic_file)
                
                chr_name = chrom
                if chr_name not in clr.chromnames:
                    alt_chr_name = chr_name[3:] if chr_name.startswith('chr') else f'chr{chr_name}'
                    if alt_chr_name in clr.chromnames:
                        chr_name = alt_chr_name
                    else:
                        print(f"Warning: {chrom} not found in cooler file.")
                        return np.zeros((num_bins1, num_bins2), dtype=np.float32)

                region1 = (chr_name, start1, end1)
                region2 = (chr_name, start2, end2)
                matrix = clr.matrix(balance=self.normalization != 'NONE').fetch(region1, region2)
                matrix = np.nan_to_num(matrix, nan=0.0)
            else:
                chrom_num = chrom.replace('chr', '')
                loc1 = f'{chrom_num}:{start1}:{end1}'
                loc2 = f'{chrom_num}:{start2}:{end2}'
                result = hicstraw.straw('observed', self.normalization, hic_file, loc1, loc2, 'BP', res)
                matrix = np.zeros((num_bins1, num_bins2), dtype=np.float32)
                
                for entry in result:
                    bin_x = (entry.binX - start1) // res
                    bin_y = (entry.binY - start2) // res
                    if 0 <= bin_x < num_bins1 and 0 <= bin_y < num_bins2:
                        matrix[bin_x, bin_y] = entry.counts
            
            if self.log1p:
                matrix = np.log1p(matrix)
                
            return matrix

        except Exception as e:
            print(f'Error reading inter-Hi-C matrix for {chrom}:{start1}-{end1} vs {start2}-{end2}: {str(e)}')
            return np.zeros((num_bins1, num_bins2), dtype=np.float32)

    
    def _read_hic_matrix(self, hic_file, chrom, start, end, res):
        """
        Reads a Hi-C interaction matrix for a specified genomic region from a Hi-C file.
        Supports both .hic files (using hicstraw) and .cool/.mcool files (using cooler).
        
        Args:
            hic_file (str): Path to the Hi-C file (.hic, .cool, or .mcool).
            chrom (str): Chromosome name.
            start (int): Start position of the genomic region.
            end (int): End position of the genomic region.
            res (int): Resolution of the Hi-C data.
            
        Returns:
            np.ndarray: A matrix representing the Hi-C interaction matrix for the specified region.
        """
        try:
            # Determine file format based on extension
            file_ext = os.path.splitext(hic_file)[1].lower()
            num_bins = (end - start) // res
            
            if file_ext in ['.cool', '.mcool']:
                # Use cooler for .cool and .mcool files
                import cooler
                
                # For mcool files, specify resolution
                if file_ext == '.mcool':
                    clr = cooler.Cooler(f'{hic_file}::/resolutions/{res}')
                else:
                    clr = cooler.Cooler(hic_file)
                    
                # Handle chromosome name format (some files use 'chr1', others use '1')
                chr_name = chrom
                if chr_name not in clr.chromnames:
                    if chr_name.startswith('chr'):
                        alt_chr_name = chr_name[3:]  # Remove 'chr' prefix
                    else:
                        alt_chr_name = f'chr{chr_name}'  # Add 'chr' prefix
                    
                    if alt_chr_name in clr.chromnames:
                        chr_name = alt_chr_name
                    else:
                        print(f"Warning: {chrom} not found in cooler file. Available chromosomes: {clr.chromnames}")
                        return np.zeros((num_bins, num_bins), dtype=np.float32)
                    
                # Fetch the matrix for the specified region
                matrix = clr.matrix(balance=self.normalization != 'NONE').fetch((chr_name, start, end))
                
                # Handle NaN values
                matrix = np.nan_to_num(matrix, nan=0.0)
                
            else:  # Default to hicstraw for .hic files
                # Construct the Hi-C location string
                chrom = chrom.replace('chr', '')
                loc = f'{chrom}:{start}:{end}'
                # Extract Hi-C interaction values
                result = hicstraw.straw('observed', self.normalization, hic_file, loc, loc, 'BP', res)
                
                # Initialize matrix
                matrix = np.zeros((num_bins, num_bins), dtype=np.float32)
                
                for entry in result:
                    bin_x = (entry.binX - start) // res
                    bin_y = (entry.binY - start) // res
                    
                    if 0 <= bin_x < num_bins and 0 <= bin_y < num_bins:
                        matrix[bin_x, bin_y] = entry.counts
                        matrix[bin_y, bin_x] = entry.counts
            
            # Apply log1p transformation if needed
            if self.log1p:
                matrix = np.log1p(matrix)
                
            return matrix
            
        except Exception as e:
            print(f'Error reading Hi-C matrix for {chrom}:{start}-{end}: {str(e)}')
            num_bins = (end - start) // self.resolution
            return np.zeros((num_bins, num_bins), dtype=np.float32)
            

    def _interpolate_matrix(self, matrix, fold):
        """
        Performs bilinear interpolation to upscale the Hi-C matrix to match the target resolution.
        :param matrix: Input Hi-C matrix
        :param fold: Upscaling factor (e.g., if target resolution is 1000bp and Hi-C is 5000bp, fold=5)
        :return: Interpolated matrix
        """
        if fold == 1:
            return matrix  # No need for interpolation if resolutions match
        #print(f'Interpolating matrix with fold {fold}...')
        id_size = matrix.shape[0]  # Assume the input is a square matrix
        fc_ = 1 / fold  # Step size for interpolation

        # Create an interpolation function using bilinear interpolation
        f = interp2d(np.arange(id_size), np.arange(id_size), matrix, kind='linear')

        # Generate new coordinates for the interpolated matrix
        new_co = np.linspace(-0.5 + fc_ / 2, id_size - 0.5 - fc_ / 2, id_size * fold)

        # Compute the upsampled matrix
        new_matrix = f(new_co, new_co)
        return new_matrix        
    def _read_chipseq_profile(self, chipseq_file, chrom, start, end):
        """
        Reads and processes the CTCF profile from a BigWig file for a specified genomic region.
        Args:
            ctcf_file (str): Path to the BigWig file containing CTCF signal data.
            chrom (str): Chromosome name.
            start (int): Start position of the genomic region.
            end (int): End position of the genomic region.
        Returns:
            np.ndarray: A 2D array where each row represents the mean CTCF signal for bins within a Hi-C resolution window.
        """
        bw = h5py.File(chipseq_file, 'r')
        try:
            # Extract the CTCF signal values for the specified region
            signals = bw[chrom][start:end]
            # signals = np.array(signals)
            # signals = np.nan_to_num(signals, nan=0.0)
            signals = np.log1p(signals)
            return signals
        
        finally:
            bw.close()
            
    def _convert_to_graph(self, hic_matrix):
        """
        Simplified and synchronized graph conversion function.
        Uses numpy operations first, then converts to torch tensors once.
        """
        try:
            rows, cols = np.nonzero(hic_matrix)
            values = hic_matrix[rows, cols]
            
            mask = rows <= cols
            rows = rows[mask]
            cols = cols[mask]
            values = values[mask]
            
            edge_index = torch.from_numpy(
                np.concatenate([
                    np.stack([rows, cols]),
                    np.stack([cols, rows])
                ], axis=1)
            )
            
            edge_attr = torch.from_numpy(
                np.concatenate([values, values])
            ).float()
            
            return edge_index, edge_attr
                
        except Exception as e:
            print(f'Error in _convert_to_graph: {str(e)}')
            raise e
    
    def _read_Sequence(self, chrom, start, end):
        with h5py.File(self.seq_h5, 'r') as f:
            seq = f[chrom][start:end]
        seq_emb = np.zeros((len(seq), 5))
        seq_emb[np.arange(len(seq)), seq] = 1
        return seq_emb


    def _process_window(self, params):
        """
        Process a single genomic window. This function will be called by each worker process.
        
        Args:
            params (tuple): Contains (chrom, start, end, output_file)
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        chrom, start, end, out_file = params
        
        try:
            # Skip if the file already exists
            if os.path.exists(out_file):
                return True

            # Read the Hi-C matrix and CTCF profile
            print(f'Processing {chrom}:{start}-{end}...with offset {self.offset}')
            if self.offset == 0:
                hic_matrix = self._read_hic_matrix(self.hic_file, chrom, start, end, self.hic_resolution)
                fold = self.hic_resolution // self.resolution
                # print(f'hic matrix shape before: {hic_matrix.shape}')
                hic_matrix = self._interpolate_matrix(hic_matrix, fold)
                # print(f'hic matrix shape: {hic_matrix.shape}')
                if self.interpolate:
                    target_resolution = self.resolution * 5 # interpolate to 5x
                else :
                    target_resolution = self.resolution
                
                target_matrix = self._read_hic_matrix(self.chia_pet_file, chrom, start, end, target_resolution)
                # print(f'target matrix shape: {target_matrix.shape}')
                if self.interpolate:
                    target_matrix = self._interpolate_matrix(target_matrix, 5)
                    # print(f'target matrix shape after interpolation: {target_matrix.shape}')
                target_matrix = torch.from_numpy(target_matrix).float()
                target_matrix = target_matrix.unsqueeze(0)
                num_nodes = self.window_size // self.resolution
                # Initialize placeholder node features  
                node_features = torch.zeros((num_nodes, 1), dtype=torch.float)
                # print('converting to graph')
                # if hic_matrix.nnz >= 0 and target_matrix.nnz >= 0:
                    # print('converting to graph...')
                edge_index, edge_attr = self._convert_to_graph(hic_matrix)
                data = Data(
                    x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=target_matrix,
                    chrom=chrom,
                    start=start,
                    end=end
                )
            else:
                num_nodes_per_anchor = self.window_size // self.resolution
                start_A, end_A = start, start + self.window_size
                start_B, end_B = start + self.offset, start + self.offset + self.window_size
                aa_matrix = self._read_hic_matrix(self.hic_file, chrom, start_A, end_A, self.hic_resolution)
                bb_matrix = self._read_hic_matrix(self.hic_file, chrom, start_B, end_B, self.hic_resolution)
                ab_matrix = self._read_inter_hic_matrix(self.hic_file, chrom, start_A, end_A, start_B, end_B, self.hic_resolution)
                target_matrix = self._read_inter_hic_matrix(self.chia_pet_file, chrom, start_A, end_A, start_B, end_B, self.resolution)
                fold = self.hic_resolution // self.resolution
                aa_matrix = self._interpolate_matrix(aa_matrix, fold)
                bb_matrix = self._interpolate_matrix(bb_matrix, fold)
                ab_matrix = self._interpolate_matrix(ab_matrix, fold)
                edge_index_aa, edge_attr_aa = self._convert_to_graph(aa_matrix)
                edge_index_bb, edge_attr_bb = self._convert_to_graph(bb_matrix)
                edge_index_bb += num_nodes_per_anchor
                edge_index_ab = torch.from_numpy(
                    np.concatenate([
                        np.stack([np.nonzero(ab_matrix)[0], np.nonzero(ab_matrix)[1] + num_nodes_per_anchor]),
                        np.stack([np.nonzero(ab_matrix)[1] + num_nodes_per_anchor, np.nonzero(ab_matrix)[0]])
                    ], axis=1)
                )
                edge_attr_ab = torch.from_numpy(
                    np.concatenate([ab_matrix[np.nonzero(ab_matrix)], ab_matrix[np.nonzero(ab_matrix)]])
                ).float()
                edge_index = torch.cat([edge_index_aa, edge_index_bb, edge_index_ab], dim=1)
                edge_attr = torch.cat([edge_attr_aa, edge_attr_bb, edge_attr_ab], dim=0)
                region_id = torch.cat([torch.zeros(num_nodes_per_anchor), torch.ones(num_nodes_per_anchor)]).long()
                data = Data(edge_index=edge_index, edge_attr=edge_attr,
                            y=torch.from_numpy(target_matrix).float().unsqueeze(0),
                            ab_hic=torch.from_numpy(ab_matrix).float().unsqueeze(0),
                            region_id=region_id,
                            anchor_A_coords=(chrom, start_A, end_A),
                            anchor_B_coords=(chrom, start_B, end_B))

            torch.save(data, out_file)
            print(f'Saved {chrom}:{start}-{end}')
            return True

        except Exception as e:
            print(f'Error processing {chrom}:{start}-{end}: {str(e)}')
            return False

    def process(self):
        """
        Process implementation with improved synchronization.
        Processes chromosomes sequentially but windows in parallel.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Set multiprocessing start method
        if self.n_processes > 1:
            import torch.multiprocessing as mp
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                pass
        
        start_time = time.time()
        print(f'Start time: {time.strftime("%H:%M:%S", time.localtime(start_time))}')
        
        # Process each chromosome
        for _, row in self.chr_sizes.iterrows():
            chrom, size = row['chr'], row['size']
            print(f'Processing {chrom}...')
            
            # Prepare windows for this chromosome
            windows = []
            for start in range(0, size, self.step_size):
                end = start + self.window_size
                if end > size:
                    continue
                out_file = os.path.join(self.processed_dir, f'data_{chrom}_{start}_{end}.pt')
                windows.append((chrom, start, end, out_file))
            
            if self.n_processes > 1:
                # Process windows in parallel
                with mp.Pool(processes=self.n_processes) as pool:
                    # Use a timeout to prevent hanging
                    results = pool.map_async(self._process_window, windows)
                    try:
                        # Wait for all processes with a timeout
                        results.get(timeout=3600)  # 1 hour timeout
                    except mp.TimeoutError:
                        print(f"Timeout processing chromosome {chrom}")
                        pool.terminate()
                    except Exception as e:
                        print(f"Error processing chromosome {chrom}: {str(e)}")
                        pool.terminate()
                    finally:
                        pool.close()
                        pool.join()
            else:
                # Process windows sequentially
                for window in windows:
                    self._process_window(window)
        
        end_time = time.time()
        print(f'End time: {time.strftime("%H:%M:%S", time.localtime(end_time))}')
        print(f'Total processing time: {end_time - start_time:.2f} seconds')

                
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        filename = self.processed_file_names[idx]
        file_path = os.path.join(self.processed_dir, filename)
        data = torch.load(file_path)

        if hasattr(data, 'region_id'): # Offset mode
            (chrom_A, start_A, end_A) = data.anchor_A_coords
            (chrom_B, start_B, end_B) = data.anchor_B_coords

            def load_features(file, chrom, start, end):
                return self._read_chipseq_profile(file, chrom, start, end)
            def load_sequence(chrom, start, end):
                return self._read_Sequence(chrom, start, end)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Load features and sequence for anchor A
                chip_futures_A = {executor.submit(load_features, f, chrom_A, start_A, end_A): f for f in self.file_list}
                seq_future_A = executor.submit(load_sequence, chrom_A, start_A, end_A)
                
                # Load features and sequence for anchor B
                chip_futures_B = {executor.submit(load_features, f, chrom_B, start_B, end_B): f for f in self.file_list}
                seq_future_B = executor.submit(load_sequence, chrom_B, start_B, end_B)

                features_A = [future.result() for future in chip_futures_A]
                features_B = [future.result() for future in chip_futures_B]
                seq_A = seq_future_A.result()
                seq_B = seq_future_B.result()

            features = np.concatenate([np.stack(features_A, axis=0), np.stack(features_B, axis=0)], axis=1)
            seq = np.concatenate([seq_A, seq_B], axis=0)
            
        else: # Standard mode
            chrom, start, end = data.chrom, data.start, data.end
            
            def load_features(file):
                return self._read_chipseq_profile(file, chrom, start, end)
            def load_sequence():
                return self._read_Sequence(chrom, start, end)
                
            with concurrent.futures.ThreadPoolExecutor() as executor:
                chip_futures = {executor.submit(load_features, f): f for f in self.file_list}
                seq_future = executor.submit(load_sequence)
                
                features = np.stack([future.result() for future in chip_futures], axis=0)
                seq = seq_future.result()

        data.features = torch.from_numpy(features).float().unsqueeze(0)
        data.seq = torch.from_numpy(seq).float().transpose(0, 1).unsqueeze(0)

        return data