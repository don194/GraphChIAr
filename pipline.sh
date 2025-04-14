#!/bin/bash

# Function to show usage
show_usage() {
    echo "Usage: $0 [options]"
    echo "Example: $0 --step_size 500000 --window_size 2000000 --resolution 10000 --normalize NONE --batch_size 32 --patience 8 --vis_samples 10 --log1p true --worker_num 4 --train-celltype GM12878 --test-celltypes \"K562 IMR90\" --target-type \"chia-pet\" --chipseq \"ctcf H3K4me3 H3K27ac\""
    echo ""
    echo "Options:"
    echo "  --step_size    Step size for sliding window (default: 500000)"
    echo "  --window_size  Window size for Hi-C matrix (default: 2000000)"
    echo "  --resolution   Resolution for Hi-C matrix (default: 10000)"
    echo "  --hic_resolution Resolution for Hi-C matrix (default: None)"
    echo "  --model        Model (default: ChIAPETMatrixPredictor)"
    echo "  --normalize    Normalization method for Hi-C matrix (default: NONE)"
    echo "  --batch_size   Batch size for training (default: 32)"
    echo "  --patience     Early stopping patience (default: 8)"
    echo "  --max_epochs   Maximum number of training epochs (default: 40)"
    echo "  --log1p        Apply log1p transformation to data (default: true)"
    echo "  --worker_num   Number of worker processes for data loading (default: 4)"
    echo "  --train-celltype Cell type to use for training (default: GM12878)"
    echo "  --test-celltypes Space-separated list of cell types to use for testing (default: GM12878 K562 IMR90)"
    echo "  --target-type   Target data type: 'chia-pet' or 'micro-c' (default: chia-pet)"
    echo "  --chipseq      Space-separated list of ChIP-seq features to use (default: ctcf)"
    echo "  --hic-format   Format of Hi-C files to use: 'hic', 'cool', or 'mcool' (default: hic)"
    echo "  --target-format Format of target files to use: 'hic', 'cool', or 'mcool' (default: same as hic-format)"
    exit 1
}

# Default values
STEP_SIZE=500000
WINDOW_SIZE=2000000
RESOLUTION=10000
NORMALIZE="NONE"
MODEL="GraphChIAr"
BATCH_SIZE=32
PATIENCE=8
MAX_EPOCHS=40
LEARNING_RATE=0.0001
LOG1P="true"
WORKER_NUM=4  
TRAIN_CELLTYPE="GM12878"  # Default training cell line
TEST_CELLTYPES=("GM12878" "K562" "IMR90")  # Default test cell lines
TARGET_TYPE="CTCF_ChIA-PET"  # Default target type
CHIPSEQ=("ctcf")     # Default ChIP-seq features
HIC_FORMAT="hic"     # Default Hi-C file format
TARGET_FORMAT=""     # By default, use same format as HIC_FORMAT

# Parse named parameters
while [[ $# -gt 0 ]]; do
    case $1 in
        --step_size)
            STEP_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --window_size)
            WINDOW_SIZE="$2"
            shift 2
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --hic_resolution)
            HIC_RESOLUTION="$2"
            shift 2
            ;;
        --normalize)
            NORMALIZE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --max_epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --log1p)
            LOG1P="$2"
            shift 2
            ;;
        --worker_num)  
            WORKER_NUM="$2"
            shift 2
            ;;
        --train-celltype)
            TRAIN_CELLTYPE="$2"
            shift 2
            ;;
        --test-celltypes)
            # Parse space-separated list into array
            IFS=' ' read -r -a TEST_CELLTYPES <<< "$2"
            shift 2
            ;;
        --target-type)
            TARGET_TYPE="$2"
            shift 2
            ;;
        --chipseq)
            # Parse space-separated list into array
            IFS=' ' read -r -a CHIPSEQ <<< "$2"
            shift 2
            ;;
        --hic-format)
            HIC_FORMAT="$2"
            shift 2
            ;;
        --target-format)
            TARGET_FORMAT="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            show_usage
            ;;
    esac
done

# If target format isn't specified, use the same as Hi-C format
if [ -z "${TARGET_FORMAT}" ]; then
    TARGET_FORMAT="${HIC_FORMAT}"
fi

# Check that specified formats are valid
if [[ ! "${HIC_FORMAT}" =~ ^(hic|cool|mcool)$ ]]; then
    echo "Error: Invalid Hi-C format '${HIC_FORMAT}'. Must be one of: hic, cool, mcool"
    exit 1
fi

if [[ ! "${TARGET_FORMAT}" =~ ^(hic|cool|mcool)$ ]]; then
    echo "Error: Invalid target format '${TARGET_FORMAT}'. Must be one of: hic, cool, mcool"
    exit 1
fi

# Generate experiment name using date, cell type, resolution and log1p setting
CURRENT_DATE=$(date +%y%m%d)
EXPERIMENT_NAME="${CURRENT_DATE}_${TRAIN_CELLTYPE}_${RESOLUTION}_${NORMALIZE}_${MODEL}_${TARGET_TYPE}_log1p_${LOG1P}_hictype_${HIC_FORMAT}_chiatype_${TARGET_FORMAT}"
if [ -n "${HIC_RESOLUTION}" ]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_hicres_${HIC_RESOLUTION}"
fi

# Set base directories
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="${BASE_DIR}/data"
PROCESSED_DATA_DIR="${BASE_DIR}/ProcessedData"
RESULTS_DIR="${BASE_DIR}/results/${EXPERIMENT_NAME}"
SRC_DIR="${BASE_DIR}/gchia"

# Create necessary directories
echo "Creating results directory: ${RESULTS_DIR}"
mkdir -p ${RESULTS_DIR}

# Save parameters to txt file
PARAM_FILE="${RESULTS_DIR}/parameters.txt"
echo "Experiment Parameters" > ${PARAM_FILE}
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')" >> ${PARAM_FILE}
echo "Experiment Name: ${EXPERIMENT_NAME}" >> ${PARAM_FILE}
echo "Train Cell Type: ${TRAIN_CELLTYPE}" >> ${PARAM_FILE}
echo "Test Cell Types: ${TEST_CELLTYPES[*]}" >> ${PARAM_FILE}
echo "Target Type: ${TARGET_TYPE}" >> ${PARAM_FILE}
echo "ChIP-seq Features: ${CHIPSEQ[*]}" >> ${PARAM_FILE}
echo "Step Size: ${STEP_SIZE}" >> ${PARAM_FILE}
echo "Window Size: ${WINDOW_SIZE}" >> ${PARAM_FILE}
echo "Resolution: ${RESOLUTION}" >> ${PARAM_FILE}
echo "Normalization: ${NORMALIZE}" >> ${PARAM_FILE}
echo "Log1p Transformation: ${LOG1P}" >> ${PARAM_FILE}
echo "Model: ${MODEL}" >> ${PARAM_FILE}
echo "Batch Size: ${BATCH_SIZE}" >> ${PARAM_FILE}
echo "Worker Number: ${WORKER_NUM}" >> ${PARAM_FILE}  
echo "Patience: ${PATIENCE}" >> ${PARAM_FILE}
echo "Max Epochs: ${MAX_EPOCHS}" >> ${PARAM_FILE}
echo "Hi-C File Format: ${HIC_FORMAT}" >> ${PARAM_FILE}
echo "Target File Format: ${TARGET_FORMAT}" >> ${PARAM_FILE}
echo "" >> ${PARAM_FILE}
echo "Directory Structure:" >> ${PARAM_FILE}
echo "Base Directory: ${BASE_DIR}" >> ${PARAM_FILE}
echo "Data Directory: ${DATA_DIR}" >> ${PARAM_FILE}
echo "Processed Data Directory: ${PROCESSED_DATA_DIR}" >> ${PARAM_FILE}
echo "Results Directory: ${RESULTS_DIR}" >> ${PARAM_FILE}
echo "Source Directory: ${SRC_DIR}" >> ${PARAM_FILE}
echo "" >> ${PARAM_FILE}
echo "Command Line Arguments:" >> ${PARAM_FILE}
echo "$0 $@" >> ${PARAM_FILE}

# Echo configuration
echo "Running pipeline with the following parameters:"
echo "Experiment name: ${EXPERIMENT_NAME}"
echo "Train cell type: ${TRAIN_CELLTYPE}"
echo "Test cell types: ${TEST_CELLTYPES[*]}"
echo "Target type: ${TARGET_TYPE}"
echo "ChIP-seq features: ${CHIPSEQ[*]}"
echo "Step size: ${STEP_SIZE}"
echo "Window size: ${WINDOW_SIZE}"
echo "Resolution: ${RESOLUTION}"
echo "Normalization: ${NORMALIZE}"
echo "Log1p Transformation: ${LOG1P}"
echo "Model: ${MODEL}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Worker Number: ${WORKER_NUM}" 
echo "Patience: ${PATIENCE}"
echo "Max Epochs: ${MAX_EPOCHS}"
echo "Parameters saved to: ${PARAM_FILE}"
echo "-----------------------------------"

# Step 0: Data Preprocessing
echo "Starting data preprocessing..."
echo "" >> ${PARAM_FILE}
echo "Data preprocessing started: $(date '+%Y-%m-%d %H:%M:%S')" >> ${PARAM_FILE}

# Helper function to process ChIP-seq files for a cell type
process_chipseq_for_celltype() {
    local cell_type=$1
    local chipseq_features=("${!2}")
    local -n result=$3 
    
    echo "Processing ChIP-seq data for ${cell_type}..."
    local cell_processed_dir="${PROCESSED_DATA_DIR}/${cell_type}"
    local chipseq_processed_dir="${cell_processed_dir}/ChIP-seq_processed"
    mkdir -p "${chipseq_processed_dir}"
    
    # 初始化空结果
    result=""
    
    for chipseq_feature in "${chipseq_features[@]}"; do
        local chipseq_bw_file="${DATA_DIR}/${cell_type}/bigWig_files/${cell_type}_${chipseq_feature}.bw"
        local chipseq_h5_file="${chipseq_processed_dir}/${cell_type}_${chipseq_feature}.h5"
        
        # Check if BigWig file exists
        if [ ! -f "${chipseq_bw_file}" ]; then
            echo "Warning: BigWig file not found for ${chipseq_feature}: ${chipseq_bw_file}"
            continue
        fi
        
        # Check if H5 file already exists
        if [ -f "${chipseq_h5_file}" ]; then
            echo "Found preprocessed ChIP-seq file for ${chipseq_feature}: ${chipseq_h5_file}"
        else
            echo "Processing ChIP-seq data for ${cell_type}_${chipseq_feature}..."
            # Process BigWig to H5
            python "${SRC_DIR}/Dataset/processchipseq.py" \
                --input-file "${chipseq_bw_file}" \
                --output-dir "${chipseq_processed_dir}" \
                --chrom-sizes "${DATA_DIR}/ReferenceGenome/hg38/hg38.chrom.sizes"
            
            if [ -f "${chipseq_h5_file}" ]; then
                echo "Successfully processed ${chipseq_bw_file} to ${chipseq_h5_file}"
            else
                echo "Warning: Failed to process ${chipseq_bw_file}"
                continue
            fi
        fi
        
        # Add to result
        result="${result} ${chipseq_feature}"
    done
}

# Helper function to find HiC and target files
find_data_files() {
    local cell_type=$1
    local target_type=$2
    local files_array_name=$3
    
    # Find Hi-C file based on specified format
    local hic_file=$(find "${DATA_DIR}/${cell_type}/Hi-C" -name "*.${HIC_FORMAT}" -type f | head -n 1)
    if [ -z "${hic_file}" ]; then
        echo "Error: No Hi-C file with format .${HIC_FORMAT} found for ${cell_type}"
        return 1
    fi
    
    # Find target file based on target type and specified format
    local target_file=""
    target_file=$(find "${DATA_DIR}/${cell_type}/${target_type}" -name "*.${TARGET_FORMAT}" -type f | head -n 1)
    if [ -z "${target_file}" ]; then
        echo "Error: No ${target_type} file with format .${TARGET_FORMAT} found for ${cell_type}"
        return 1
    fi
    
    # Set values in the provided array name
    eval "${files_array_name}=('${hic_file}' '${target_file}')"
    return 0
}

# Process ChIP-seq files for training cell type
TRAIN_CHIPSEQ_CMD_ARG=""
process_chipseq_for_celltype "${TRAIN_CELLTYPE}" CHIPSEQ[@] TRAIN_CHIPSEQ_CMD_ARG
echo "ChIP-seq command argument for training: ${TRAIN_CHIPSEQ_CMD_ARG}"

# Find Hi-C and target files for training cell type
declare -a TRAIN_FILES
if ! find_data_files "${TRAIN_CELLTYPE}" "${TARGET_TYPE}" TRAIN_FILES; then
    echo "Error finding data files for training cell type ${TRAIN_CELLTYPE}"
    exit 1
fi

TRAIN_HIC_FILE="${TRAIN_FILES[0]}"
TRAIN_TARGET_FILE="${TRAIN_FILES[1]}"
echo "Using Hi-C file for ${TRAIN_CELLTYPE}: ${TRAIN_HIC_FILE}"
echo "Using ${TARGET_TYPE} file for ${TRAIN_CELLTYPE}: ${TRAIN_TARGET_FILE}"

# Convert log1p string to boolean flag for Python scripts
LOG1P_FLAG=""
if [ "${LOG1P}" = "true" ]; then
    LOG1P_FLAG="--log1p True"
else
    LOG1P_FLAG="--log1p False"
fi

# Add hic_resolution flag if set
HIC_RES_FLAG=""
if [ -n "${HIC_RESOLUTION}" ]; then
    HIC_RES_FLAG="--hic_resolution ${HIC_RESOLUTION}"
fi

echo "Data preprocessing completed: $(date '+%Y-%m-%d %H:%M:%S')" >> ${PARAM_FILE}
# Step 1: Training
echo "Starting model training..."
echo "" >> ${PARAM_FILE}
echo "Training started: $(date '+%Y-%m-%d %H:%M:%S')" >> ${PARAM_FILE}

python ${SRC_DIR}/Train/train.py \
    --seed 42 \
    --save_path "${RESULTS_DIR}" \
    --model "${MODEL}" \
    --data-root "${PROCESSED_DATA_DIR}" \
    --celltype "${TRAIN_CELLTYPE}" \
    --hic-file "${TRAIN_HIC_FILE}" \
    --chia-pet-file "${TRAIN_TARGET_FILE}" \
    --chipseq-files ${TRAIN_CHIPSEQ_CMD_ARG} \
    --chr-sizes-file "${DATA_DIR}/ReferenceGenome/hg38/hg38.chrom.sizes" \
    --target-type "${TARGET_TYPE}" \
    --patience ${PATIENCE} \
    --max_epochs ${MAX_EPOCHS} \
    --save_top_k 60 \
    --lr ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${WORKER_NUM} \
    --step_size ${STEP_SIZE} \
    --window_size ${WINDOW_SIZE} \
    --resolution ${RESOLUTION} \
    --normalize ${NORMALIZE} \
    ${LOG1P_FLAG} \
    ${HIC_RES_FLAG}

echo "Training completed: $(date '+%Y-%m-%d %H:%M:%S')" >> ${PARAM_FILE}

# Get the latest checkpoint file
CHECKPOINT_FILE=$(ls -t ${RESULTS_DIR}/model-${TRAIN_CELLTYPE}/epoch*.ckpt 2>/dev/null | head -n1)
if [ -z "$CHECKPOINT_FILE" ]; then
    echo "Error: No checkpoint file found in ${RESULTS_DIR}/model-${TRAIN_CELLTYPE}/"
    exit 1
fi
echo "Using checkpoint file: ${CHECKPOINT_FILE}" | tee -a ${PARAM_FILE}

# Step 2: Predictions
echo "Starting predictions..."
echo "" >> ${PARAM_FILE}
echo "Predictions started: $(date '+%Y-%m-%d %H:%M:%S')" >> ${PARAM_FILE}

# Process and predict for each test cell type
for TEST_CELL in "${TEST_CELLTYPES[@]}"; do
    echo "Processing test cell type: ${TEST_CELL}"
    
    # Process ChIP-seq for this test cell type (using same features as training)
    TEST_CHIPSEQ_CMD_ARG=""
    process_chipseq_for_celltype "${TEST_CELL}" CHIPSEQ[@] TEST_CHIPSEQ_CMD_ARG
    
    # Find data files for this cell type
    declare -a TEST_FILES
    if ! find_data_files "${TEST_CELL}" "${TARGET_TYPE}" TEST_FILES; then
        echo "Warning: Could not find data files for ${TEST_CELL}, skipping"
        continue
    fi
    
    TEST_HIC_FILE="${TEST_FILES[0]}"
    TEST_TARGET_FILE="${TEST_FILES[1]}"
    echo "Using Hi-C file for ${TEST_CELL}: ${TEST_HIC_FILE}"
    echo "Using ${TARGET_TYPE} file for ${TEST_CELL}: ${TEST_TARGET_FILE}"
    
    # Create prediction directory
    PREDICTION_DIR="${RESULTS_DIR}"
    if [ "${TEST_CELL}" = "${TRAIN_CELLTYPE}" ]; then
        PREDICTION_DIR="${RESULTS_DIR}/${TEST_CELL}"
    else
        PREDICTION_DIR="${RESULTS_DIR}/${TRAIN_CELLTYPE}to${TEST_CELL}"
    fi
    mkdir -p "${PREDICTION_DIR}/predictions"
    
    # Modify the predict command with updated parameters
    echo "Predicting ${TEST_CELL}..."
    python ${SRC_DIR}/Predict/predict.py \
        --data-root "${PROCESSED_DATA_DIR}" \
        --celltype "${TEST_CELL}" \
        --hic-file "${TEST_HIC_FILE}" \
        --chia-pet-file "${TEST_TARGET_FILE}" \
        --chipseq-files ${TEST_CHIPSEQ_CMD_ARG} \
        --chr-sizes-file "${DATA_DIR}/ReferenceGenome/hg38/hg38.chrom.sizes" \
        --model "${MODEL}" \
        --target-type "${TARGET_TYPE}" \
        --checkpoint-path "${CHECKPOINT_FILE}" \
        --save-dir "${PREDICTION_DIR}/predictions" \
        --batch_size ${BATCH_SIZE} \
        --num_workers ${WORKER_NUM} \
        --step_size ${STEP_SIZE} \
        --window_size ${WINDOW_SIZE} \
        --resolution ${RESOLUTION} \
        --normalize ${NORMALIZE} \
        ${LOG1P_FLAG} \
        ${HIC_RES_FLAG}
done

echo "Predictions completed: $(date '+%Y-%m-%d %H:%M:%S')" >> ${PARAM_FILE}

# Step 3: Metrics Calculation
echo "Starting metrics calculation..."
echo "" >> ${PARAM_FILE}
echo "Metrics calculation started: $(date '+%Y-%m-%d %H:%M:%S')" >> ${PARAM_FILE}

for TEST_CELL in "${TEST_CELLTYPES[@]}"; do
    echo "Calculating metrics for ${TEST_CELL}..."
    OUTPUT_DIR="${RESULTS_DIR}"
    if [ "${TEST_CELL}" = "${TRAIN_CELLTYPE}" ]; then
        OUTPUT_DIR="${RESULTS_DIR}/${TEST_CELL}"
    else
        OUTPUT_DIR="${RESULTS_DIR}/${TRAIN_CELLTYPE}to${TEST_CELL}"
    fi

    # Check if prediction directory exists
    if [ ! -d "${OUTPUT_DIR}/predictions" ]; then
        echo "Warning: No predictions found for ${TEST_CELL}, skipping metrics calculation"
        continue
    fi
    
    # Find target file based on target type and specified format
    TARGET_FILE=""
    target_file=$(find "${DATA_DIR}/${TEST_CELL}/${TARGET_TYPE}" -name "*.${TARGET_FORMAT}" -type f | head -n 1)
    
    if [ -z "${target_file}" ]; then
        echo "Warning: No target file with format .${TARGET_FORMAT} found for ${TEST_CELL}, skipping metrics calculation"
        continue
    fi
    
    TARGET_FILE="${target_file}"
    
    mkdir -p "${OUTPUT_DIR}/metrics"
    python ${SRC_DIR}/Metrics/SCC.py \
        --matrix_directory "${OUTPUT_DIR}/predictions" \
        --real_hic_filepath "${TARGET_FILE}" \
        --chromosome chr10 \
        --output_path "${OUTPUT_DIR}/metrics" \
        --max_distance ${WINDOW_SIZE} \
        --resolution ${RESOLUTION} \
        --norm ${NORMALIZE} \
        ${LOG1P_FLAG}
done

echo "Metrics calculation completed: $(date '+%Y-%m-%d %H:%M:%S')" >> ${PARAM_FILE}

# Add final completion time to parameters file
echo "" >> ${PARAM_FILE}
echo "Pipeline completed: $(date '+%Y-%m-%d %H:%M:%S')" >> ${PARAM_FILE}

echo "Pipeline completed successfully!"