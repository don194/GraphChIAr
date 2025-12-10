# GraphChIAr

## GraphChIAr: Genome-wide super-resolution reconstruction of protein-mediated remote chromatin interactions by augmenting Hi-C interaction maps with multiple ChIP-seq profiles  

GraphChIAr is a deep learning framework that predicts ChIA-PET interactions by integrating Hi-C, ChIP-seq, and genomic sequence data. This model uses advanced neural network architectures to capture the complex spatial relationships in chromatin structure.

## Project Structure

```
GraphChIAr/
├── data/                   # Data preprocessing and download scripts
│   ├── downloadepi.sh      # Script to download epigenetic data
│   ├── convert_bam_to...   # Tools for processing BAM to BigWig
│   └── GM12878/            # Cell-line specific data (Hi-C, ChIA-PET)
├── gchia/                  # Main source code package
│   ├── Config/             # Configuration files (e.g., plotting configs)
│   ├── Dataset/            # Data loading and normalization logic
│   ├── Figure/             # Jupyter notebooks for generating publication figures
│   ├── Metrics/            # Evaluation metrics (SCC, Loss visualization)
│   ├── Model/              # Neural network architectures (Difformer, Blocks)
│   ├── Predict/            # Inference scripts
│   └── Train/              # Training loops and testing scripts
├── ReferenceGenome/        # Genome reference files (hg38 chrom sizes)
├── train_and_evaluate.sh   # All-in-one execution script
└── README.md               # Project documentation

```

## Data Requirements

To run GraphChIAr, you need to prepare the following data:

### Data Directory Structure

```
data/
  ├── [cell_type]/
  │    ├── Hi-C/
  │    │    └── [cell_type].(hic|cool|mcool)
  │    ├── [target_type]/
  │    │    └── [cell_type].(hic|cool|mcool)
  │    └── bigWig_files/
  │         └── [cell_type]_[chipseq_feature].bw
  └── ReferenceGenome/
       └── hg38/
            ├── hg38.chrom.sizes
            └── hg38.h5
```

### Required Data Files

1. **Hi-C data**: 3D genome contact maps in .hic, .cool, or .mcool format.
2. **ChIA-PET or Micro-C data**: Target interaction data in .hic, .cool, or .mcool format.
3. **ChIP-seq data**: Protein binding profiles in .bw (BigWig) format for features like CTCF, H3K4me3, H3K27ac, etc.
4. **Reference Genome**: Human genome chromosome sizes file (hg38.chrom.sizes). Genomic sequence data in HDF5 format (hg38.h5). This file contains the one-hot encoded DNA sequences required by the model.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GraphChIAr.git
cd GraphChIAr

# Create and activate conda environment with Python 3.9
conda create -n GraphChIA python=3.9 -y
conda activate GraphChIA

# Install PyTorch 2.4 with CUDA 12.4 support
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# Install PyTorch Geometric
conda install pyg -c pyg

# Install PyTorch Lightning
pip install pytorch-lightning==1.9.0

# Install Hi-C and genomic data processing packages
pip install hic-straw cooler pyBigWig h5py

# Install other dependencies
pip install numpy==1.26.4 pandas matplotlib scikit-learn==1.3.0 scipy==1.13
```

## Usage

### 1. Training and Evaluation

You can train the model and evaluate it on test cell types using the provided shell script:

```
./train_and_evaluate.sh --train-celltype GM12878 \
  --test-celltypes "K562 IMR90" \
  --target-type "CTCF_ChIA-PET" \
  --chipseq "ctcf" \
  --resolution 10000 \
  --window_size 2000000 \
  --step_size 500000 \
  --model GraphChIAr --normalize NONE --log1p true \
  --batch_size 32 \
  --patience 10 \
  --max_epochs 80 \
  --worker_num 4 \
  --hic-format hic \
  --target-format hic
```

### 2. Prediction with Pre-trained Model

To make predictions on specific genomic regions using a trained checkpoint:

```
python -m gchia.Predict.predict \
  --checkpoint-path /path/to/checkpoint.ckpt \
  --output-dir /path/to/output/directory \
  --model GraphChIAr \
  --data-root /path/to/ProcessedData \
  --celltype GM12878 \
  --hic-file /path/to/GM12878.hic \
  --chipseq-files ctcf \
  --chr-sizes-file ReferenceGenome/hg38/hg38.chrom.sizes \
  --hic_resolution 5000 \
  --resolution 10000 \
  --offset 10000 \
  --step_size 500000 \
  --window_size 2000000 \
  --chrom chr1 --start 0 --end 10000000 \
  --log1p true
```

## Parameters

### Data & Input Parameters

|   |   |   |
|---|---|---|
|**Parameter**|**Default**|**Description**|
|`--train-celltype`|`GM12878`|The cell line used for training the model.|
|`--test-celltypes`|`GM12878 K562 IMR90`|Space-separated list of cell lines used for testing/evaluation.|
|`--target-type`|`CTCF_ChIA-PET`|The target experimental data type (Ground Truth).|
|`--chipseq`|`ctcf`|Space-separated list of ChIP-seq features (e.g., ctcf H3K4me3).|
|`--hic-format`|`hic`|File format for input Hi-C data (`hic`, `cool`, `mcool`).|
|`--target-format`|`hic`|File format for target data (`hic`, `cool`, `mcool`).|
|`--data-root`|`./data`|Root directory containing the processed data.|

### Model & Geometry Parameters

|   |   |   |
|---|---|---|
|**Parameter**|**Default**|**Description**|
|`--resolution`|`10000`|The resolution (bin size) for the target interaction matrix.|
|`--hic_resolution`|`None`|Optional input Hi-C resolution if different from target resolution.|
|`--offset`|`0`|Genomic distance offset from diagonal to exclude or start sampling (in bp).|
|`--window_size`|`2000000`|Size of the genomic window (in base pairs) for each sample.|
|`--step_size`|`500000`|Step size (stride) for the sliding window mechanism.|
|`--model`|`GraphChIAr`|The model architecture to use.|
|`--normalize`|`NONE`|Normalization method applied to the interaction matrix.|
|`--log1p`|`true`|Whether to apply log1p transformation to the input data.|

### Training Hyperparameters

|   |   |   |
|---|---|---|
|**Parameter**|**Default**|**Description**|
|`--batch_size`|`32`|Number of samples per batch during training.|
|`--max_epochs`|`40`|Maximum number of training epochs.|
|`--lr`|`0.0001`|Initial learning rate for the optimizer.|
|`--patience`|`8`|Number of epochs to wait for improvement before early stopping.|
|`--worker_num`|`4`|Number of worker processes for data loading.|
|`--seed`|`42`|Random seed for reproducibility.|

### Prediction Specific Parameters

|   |   |   |
|---|---|---|
|**Parameter**|**Required**|**Description**|
|`--checkpoint-path`|**Yes**|Path to the saved model checkpoint (`.ckpt`).|
|`--output-dir`|**Yes**|Directory to save the predicted matrices and metrics.|
|`--chrom`|No|Specific chromosome to predict (e.g., `chr1`).|
|`--start`|No|Start coordinate for prediction range.|
|`--end`|No|End coordinate for prediction range.|


## Output Structure

```
results/[experiment_name]/
  ├── parameters.txt                   # Configuration parameters
  ├── model-[cell_type]/               # Model checkpoints
  └── [train_cell]/                    # predictions
       ├── predictions/                # Predicted matrices
       └── metrics/                    # Evaluation metrics
```

## Example Workflow

1. **Prepare data** according to the required directory structure
2. **Train a model** using the train_and_evaluate.sh script
3. **Make predictions** on new cell types using the trained model

## Citation

If you use GraphChIAr in your research, please cite:
```

```


