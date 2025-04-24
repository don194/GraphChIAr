# GraphChIAr

## Multi-resolution regression model for ChIA-PET prediction through integrated Hi-C, ChIP-seq, and sequence data

GraphChIAr is a deep learning framework that predicts ChIA-PET interactions by integrating Hi-C, ChIP-seq, and genomic sequence data. This model uses advanced neural network architectures to capture the complex spatial relationships in chromatin structure.

## Project Structure

```
GraphChIAr/
├── data/                       # Directory for input data
├── gchia/                      # Main source code package
│   ├── Dataset/                # Dataset preparation and processing
│   │   └── dataset_normalization.py
│   ├── Figure/                 # Output figures and visualizations
│   │   └── 1/, 2/, 3/, 4/, 5/ 
│   ├── Metrics/                # Evaluation metrics implementation
│   │   ├── loopscall/          # Loop calling analysis
│   │   ├── lossplot.py         # Training loss visualization
│   │   ├── visualize.py        # Prediction visualization
│   │   ├── SCC.py              # Stratum-adjusted correlation coefficient
│   │   └── SCCplot.py          # SCC visualization
│   ├── Model/                  # Model architectures
│   ├── Predict/                # Scripts for making predictions
│   └── Train/                  # Scripts for model training
├── train_and_evaluate.sh       # Scripts for model training and evaluation
├── predict_pip.sh              # Prediction-only pipeline script
├── ProcessedData/              # Directory for processed data
└── ReferenceGenome/            # Reference genome files
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
            └── hg38.chrom.sizes
```

### Required Data Files

1. **Hi-C data**: 3D genome contact maps in .hic, .cool, or .mcool format.
2. **ChIA-PET or Micro-C data**: Target interaction data in .hic, .cool, or .mcool format.
3. **ChIP-seq data**: Protein binding profiles in .bw (BigWig) format for features like CTCF, H3K4me3, H3K27ac, etc.
4. **Reference Genome**: Human genome chromosome sizes file (hg38.chrom.sizes).

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GraphChIAr.git
cd GraphChIAr

# Create and activate conda environment (recommended)
conda create -n graphchiar python=3.8
conda activate graphchiar

# Install dependencies
pip install torch
pip install torch-geometric
pip install pytorch-lightning
pip install h5py cooler hic-straw
pip install numpy pandas matplotlib scikit-learn
pip install pyBigWig
```

## Usage

### Training and Prediction Pipeline

To train a model and make predictions on test cell types:

```bash
./train_and_evaluate.sh --train-celltype GM12878 --test-celltypes "K562 IMR90" \
  --target-type "CTCF_ChIA-PET" --chipseq "ctcf H3K4me3 H3K27ac" \
  --resolution 10000 --window_size 2000000 --step_size 500000 \
  --model GraphChIAr --normalize NONE --log1p true \
  --batch_size 32 --patience 8 --max_epochs 40 --worker_num 4 \
  --hic-format hic --target-format hic
```

### Prediction with Pre-trained Model

To make predictions using a pre-trained model checkpoint:

```bash
python predict.py \
  --checkpoint-path /path/to/checkpoint.ckpt \
  --output-dir /path/to/output/directory \
  --model GraphChIAr_efeaturesq_high \
  --data-root /path/to/ProcessedData \
  --celltype GM12878 \
  --hic-file /path/to/Hi-C \
  --chipseq-files ctcf \
  --chr-sizes-file /path/to/ReferenceGenome/hg38/hg38.chrom.sizes \
  --hic_resolution 5000 \
  --resolution 1000 \
  --step_size 500000 \
  --window_size 500000 \
  --chrom chr1 --start 0 --end 10000000 \
  --log1p true
```

## Parameters

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--step_size` | Step size for sliding window | 500000 |
| `--window_size` | Window size for Hi-C matrix | 2000000 |
| `--resolution` | Resolution for Hi-C matrix | 10000 |
| `--hic_resolution` | Optional higher resolution for Hi-C matrix | None |
| `--model` | Model architecture | GraphChIAr |
| `--normalize` | Normalization method for Hi-C matrix | NONE |
| `--log1p` | Apply log1p transformation | true |
| `--batch_size` | Batch size | 32 |
| `--worker_num` | Number of data loader workers | 4 |
| `--train-celltype` | Cell type for training | GM12878 |
| `--test-celltypes` | Space-separated list of cell types for testing | GM12878 K562 IMR90 |
| `--target-type` | Target data type | CTCF_ChIA-PET |
| `--chipseq` | Space-separated list of ChIP-seq features | ctcf |
| `--hic-format` | Format of Hi-C files (hic, cool, mcool) | hic |
| `--target-format` | Format of target files | Same as hic-format |

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--lr` | Learning rate | 0.0001 |
| `--patience` | Early stopping patience | 8 |
| `--max_epochs` | Maximum training epochs | 40 |

### Prediction Parameter

| Parameter | Description | Required |
|-----------|-------------|----------|
| `--model-path` | Path to pre-trained model checkpoint | Yes |

## Output Structure

```
results/[experiment_name]/
  ├── parameters.txt                   # Configuration parameters
  ├── model-[cell_type]/               # Model checkpoints
  └── [train_cell]to[test_cell]/       # Cross-cell predictions
       ├── predictions/                # Predicted matrices
       └── metrics/                    # Evaluation metrics
```

## Metrics and Evaluation

The model is evaluated using multiple metrics including:
- Stratum-adjusted correlation coefficient (SCC)
- Pearson and Spearman correlations
- Mean squared error (MSE)
- Loop detection accuracy (via the loopscall module)

Visualization tools in the Figure directory help interpret results and compare predictions with ground truth data.

## Example Workflow

1. **Prepare data** according to the required directory structure
2. **Train a model** using the train_and_evaluate.sh script
3. **Make predictions** on new cell types using the trained model
4. **Evaluate results** using the metrics in the Metrics/ directory
5. **Visualize predictions** using tools in the Figure/ directory

## Citation

If you use GraphChIAr in your research, please cite:
```

```


