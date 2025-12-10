#!/bin/bash

# Extract the first md5sum value from an ENCODE file JSON
get_first_md5_from_json() {
    local url=$1

    # Use curl to fetch the JSON payload and jq to pull the md5sum
    md5_value=$(curl -s "$url" | jq -r '.md5sum')

    if [[ -n "$md5_value" ]]; then
        echo "$md5_value"
    else
        echo "MD5 not found for $url"
    fi
}

# Define the cell lines and their associated download links
declare -A download_links=(
    # GM12878
    ["GM12878/ctcf_chip-seq/ENCFF430XCG"]="https://www.encodeproject.org/files/ENCFF430XCG/@@download/ENCFF430XCG.bam"
    ["GM12878/ctcf_chip-seq/ENCFF794BPW"]="https://www.encodeproject.org/files/ENCFF794BPW/@@download/ENCFF794BPW.bam"
    ["GM12878/H3K4me3_chip-seq/ENCFF540GUY"]="https://www.encodeproject.org/files/ENCFF540GUY/@@download/ENCFF540GUY.bam"
    ["GM12878/H3K4me3_chip-seq/ENCFF938DOA"]="https://www.encodeproject.org/files/ENCFF938DOA/@@download/ENCFF938DOA.bam"
    ["GM12878/H3K27me3_chip-seq/ENCFF265UBT"]="https://www.encodeproject.org/files/ENCFF265UBT/@@download/ENCFF265UBT.bam"
    ["GM12878/H3K27me3_chip-seq/ENCFF824VSE"]="https://www.encodeproject.org/files/ENCFF824VSE/@@download/ENCFF824VSE.bam"
    ["GM12878/H3K27ac_chip-seq/ENCFF269GKF"]="https://www.encodeproject.org/files/ENCFF269GKF/@@download/ENCFF269GKF.bam"
    ["GM12878/H3K27ac_chip-seq/ENCFF201OHW"]="https://www.encodeproject.org/files/ENCFF201OHW/@@download/ENCFF201OHW.bam"
    ["GM12878/RAD21_chip-seq/ENCFF545ETC"]="https://www.encodeproject.org/files/ENCFF545ETC/@@download/ENCFF545ETC.bam"
    ["GM12878/RAD21_chip-seq/ENCFF310ZIS"]="https://www.encodeproject.org/files/ENCFF310ZIS/@@download/ENCFF310ZIS.bam"
    ["GM12878/DNase-seq/ENCFF593WBR"]="https://www.encodeproject.org/files/ENCFF593WBR/@@download/ENCFF593WBR.bam"
    ["GM12878/DNase-seq/ENCFF658WKQ"]="https://www.encodeproject.org/files/ENCFF658WKQ/@@download/ENCFF658WKQ.bam"
    ["GM12878/POLR2A/ENCFF418NDO"]="https://www.encodeproject.org/files/ENCFF418NDO/@@download/ENCFF418NDO.bam"
    ["GM12878/POLR2A/ENCFF886CYK"]="https://www.encodeproject.org/files/ENCFF886CYK/@@download/ENCFF886CYK.bam"
    ["GM12878/SMC3/ENCFF622ERY"]="https://www.encodeproject.org/files/ENCFF622ERY/@@download/ENCFF622ERY.bam"
    ["GM12878/SMC3/ENCFF302PYC"]="https://www.encodeproject.org/files/ENCFF302PYC/@@download/ENCFF302PYC.bam"

    # IMR90
    # ["IMR90/ctcf_chip-seq/ENCFF584CBK"]="https://www.encodeproject.org/files/ENCFF584CBK/@@download/ENCFF584CBK.bam"
    # ["IMR90/ctcf_chip-seq/ENCFF160QTH"]="https://www.encodeproject.org/files/ENCFF160QTH/@@download/ENCFF160QTH.bam"
    # ["IMR90/H3K4me3_chip-seq/ENCFF355WLE"]="https://www.encodeproject.org/files/ENCFF355WLE/@@download/ENCFF355WLE.bam"
    # ["IMR90/H3K4me3_chip-seq/ENCFF021CLP"]="https://www.encodeproject.org/files/ENCFF021CLP/@@download/ENCFF021CLP.bam"
    # ["IMR90/H3K27me3_chip-seq/ENCFF547BFL"]="https://www.encodeproject.org/files/ENCFF547BFL/@@download/ENCFF547BFL.bam"
    # ["IMR90/H3K27me3_chip-seq/ENCFF084CZB"]="https://www.encodeproject.org/files/ENCFF084CZB/@@download/ENCFF084CZB.bam"
    # ["IMR90/H3K27ac_chip-seq/ENCFF064XTS"]="https://www.encodeproject.org/files/ENCFF064XTS/@@download/ENCFF064XTS.bam"
    # ["IMR90/H3K27ac_chip-seq/ENCFF827KCG"]="https://www.encodeproject.org/files/ENCFF827KCG/@@download/ENCFF827KCG.bam"
    # ["IMR90/RAD21_chip-seq/ENCFF918IXN"]="https://www.encodeproject.org/files/ENCFF918IXN/@@download/ENCFF918IXN.bam"
    # ["IMR90/RAD21_chip-seq/ENCFF824JOL"]="https://www.encodeproject.org/files/ENCFF824JOL/@@download/ENCFF824JOL.bam"
    # ["IMR90/DNase-seq/ENCFF264LKY"]="https://www.encodeproject.org/files/ENCFF264LKY/@@download/ENCFF264LKY.bam"

    # # H1-hESC
    # ["H1-hESC/ctcf_chip-seq/ENCFF767YLK"]="https://www.encodeproject.org/files/ENCFF767YLK/@@download/ENCFF767YLK.bam"
    # ["H1-hESC/H3K4me3_chip-seq/ENCFF638NIJ"]="https://www.encodeproject.org/files/ENCFF638NIJ/@@download/ENCFF638NIJ.bam"
    # ["H1-hESC/H3K4me3_chip-seq/ENCFF940GKD"]="https://www.encodeproject.org/files/ENCFF940GKD/@@download/ENCFF940GKD.bam"
    # ["H1-hESC/H3K27me3_chip-seq/ENCFF495SRZ"]="https://www.encodeproject.org/files/ENCFF495SRZ/@@download/ENCFF495SRZ.bam"
    # ["H1-hESC/H3K27me3_chip-seq/ENCFF915QOK"]="https://www.encodeproject.org/files/ENCFF915QOK/@@download/ENCFF915QOK.bam"
    # ["H1-hESC/H3K27ac_chip-seq/ENCFF324DHZ"]="https://www.encodeproject.org/files/ENCFF324DHZ/@@download/ENCFF324DHZ.bam"
    # ["H1-hESC/H3K27ac_chip-seq/ENCFF262JMM"]="https://www.encodeproject.org/files/ENCFF262JMM/@@download/ENCFF262JMM.bam"
    # ["H1-hESC/RAD21_chip-seq/ENCFF200HXK"]="https://www.encodeproject.org/files/ENCFF200HXK/@@download/ENCFF200HXK.bam"
    # ["H1-hESC/RAD21_chip-seq/ENCFF100EES"]="https://www.encodeproject.org/files/ENCFF100EES/@@download/ENCFF100EES.bam"
    # ["H1-hESC/DNase-seq/ENCFF175QBB"]="https://www.encodeproject.org/files/ENCFF175QBB/@@download/ENCFF175QBB.bam"

    # # K562
    # ["K562/ctcf_chip-seq/ENCFF111JKR"]="https://www.encodeproject.org/files/ENCFF111JKR/@@download/ENCFF111JKR.bam"
    # ["K562/ctcf_chip-seq/ENCFF945CJG"]="https://www.encodeproject.org/files/ENCFF945CJG/@@download/ENCFF945CJG.bam"
    # ["K562/ctcf_chip-seq/ENCFF706PLK"]="https://www.encodeproject.org/files/ENCFF706PLK/@@download/ENCFF706PLK.bam"
    # ["K562/H3K4me3_chip-seq/ENCFF181ANT"]="https://www.encodeproject.org/files/ENCFF181ANT/@@download/ENCFF181ANT.bam"
    # ["K562/H3K4me3_chip-seq/ENCFF747HEB"]="https://www.encodeproject.org/files/ENCFF747HEB/@@download/ENCFF747HEB.bam"
    # ["K562/H3K27me3_chip-seq/ENCFF652TXG"]="https://www.encodeproject.org/files/ENCFF652TXG/@@download/ENCFF652TXG.bam"
    # ["K562/H3K27me3_chip-seq/ENCFF508LLH"]="https://www.encodeproject.org/files/ENCFF508LLH/@@download/ENCFF508LLH.bam"
    # ["K562/H3K27ac_chip-seq/ENCFF600THN"]="https://www.encodeproject.org/files/ENCFF600THN/@@download/ENCFF600THN.bam"
    # ["K562/H3K27ac_chip-seq/ENCFF704LGA"]="https://www.encodeproject.org/files/ENCFF704LGA/@@download/ENCFF704LGA.bam"
    # ["K562/H3K27ac_chip-seq/ENCFF232RQF"]="https://www.encodeproject.org/files/ENCFF232RQF/@@download/ENCFF232RQF.bam"
    # ["K562/RAD21_chip-seq/ENCFF636DAM"]="https://www.encodeproject.org/files/ENCFF636DAM/@@download/ENCFF636DAM.bam"
    # ["K562/RAD21_chip-seq/ENCFF099DUX"]="https://www.encodeproject.org/files/ENCFF099DUX/@@download/ENCFF099DUX.bam"
    # ["K562/DNase-seq/ENCFF533TKL"]="https://www.encodeproject.org/files/ENCFF533TKL/@@download/ENCFF533TKL.bam"
    # ["K562/POLR2A/ENCFF380XSC"]="https://www.encodeproject.org/files/ENCFF380XSC/@@download/ENCFF380XSC.bam"
    # ["K562/POLR2A/ENCFF717URX"]="https://www.encodeproject.org/files/ENCFF717URX/@@download/ENCFF717URX.bam"

    # # HCT116
    # ["HCT116/ctcf_chip-seq/ENCFF109VAD"]="https://www.encodeproject.org/files/ENCFF109VAD/@@download/ENCFF109VAD.bam"
    # ["HCT116/ctcf_chip-seq/ENCFF689CXB"]="https://www.encodeproject.org/files/ENCFF689CXB/@@download/ENCFF689CXB.bam"
    # ["HCT116/H3K4me3_chip-seq/ENCFF013LEY"]="https://www.encodeproject.org/files/ENCFF013LEY/@@download/ENCFF013LEY.bam"
    # ["HCT116/H3K4me3_chip-seq/ENCFF178XNO"]="https://www.encodeproject.org/files/ENCFF178XNO/@@download/ENCFF178XNO.bam"
    # ["HCT116/H3K27me3_chip-seq/ENCFF896UDP"]="https://www.encodeproject.org/files/ENCFF896UDP/@@download/ENCFF896UDP.bam"
    # ["HCT116/H3K27ac_chip-seq/ENCFF900PXN"]="https://www.encodeproject.org/files/ENCFF900PXN/@@download/ENCFF900PXN.bam"
    # ["HCT116/H3K27ac_chip-seq/ENCFF398DWU"]="https://www.encodeproject.org/files/ENCFF398DWU/@@download/ENCFF398DWU.bam"
    # ["HCT116/RAD21_chip-seq/ENCFF973DXN"]="https://www.encodeproject.org/files/ENCFF973DXN/@@download/ENCFF973DXN.bam"
    # ["HCT116/RAD21_chip-seq/ENCFF761EMD"]="https://www.encodeproject.org/files/ENCFF761EMD/@@download/ENCFF761EMD.bam"
    # ["HCT116/DNase-seq/ENCFF955RHM"]="https://www.encodeproject.org/files/ENCFF955RHM/@@download/ENCFF955RHM.bam"
    # ["HCT116/DNase-seq/ENCFF958BEI"]="https://www.encodeproject.org/files/ENCFF958BEI/@@download/ENCFF958BEI.bam"
)

# Initialize the MD5 lookup table
declare -A md5_values=()

# Track downloaded BAM targets and merged outputs
declare -a target_files=()
declare -a merged_files=()

# Retrieve and store the MD5 value for each file
for path in "${!download_links[@]}"; do
    file_id=$(basename "$path")
    json_url="https://www.encodeproject.org/files/${file_id}/?format=json"
    md5=$(get_first_md5_from_json "$json_url")
    
    # Report the MD5 value we grabbed for visibility
    echo "Retrieved MD5 for $path: $md5"
    
    md5_values["$path"]="$md5"
done

# Validate file integrity via MD5
check_md5() {
    local file=$1
    local expected_md5=$2

    # Compute the local MD5 for comparison
    local actual_md5=$(md5sum "$file" | awk '{print $1}')

    if [[ "$actual_md5" == "$expected_md5" ]]; then
        echo "$file MD5 matches."
        return 0
    else
        echo "Warning: $file MD5 does not match."
        return 1
    fi
}

# Use samtools to verify BAM integrity
check_bam_integrity() {
    local bam_file=$1

    if samtools quickcheck "$bam_file"; then
        echo "$bam_file is complete and valid."
        return 0
    else
        echo "Warning: $bam_file is corrupt or incomplete."
        return 1
    fi
}

# Download helper that validates folders, resumes downloads, and skips valid files
download_file() {
    local dir=$(dirname "$1")
    local file_name=$(basename "$url")
    local url=$2

    mkdir -p "$dir"

    # Look up the expected MD5 checksum
    local expected_md5=${md5_values["$1"]}

    # Keep downloading until the file passes MD5 validation
    while true; do
        # If the file already exists, confirm its MD5 first
        if [[ -f "$dir/$file_name" ]]; then
            if check_md5 "$dir/$file_name" "$expected_md5"; then
                echo "$file_name already exists and is valid. Skipping download."
                # Track the valid file so we can process it later
                target_files+=("$dir/$file_name")
                break
            else
                echo "$file_name is invalid. Re-downloading..."
                rm "$dir/$file_name"
            fi
        fi

        # Download the file (quiet mode to avoid overwhelming logs)
        wget -c -P "$dir" "$url" || echo "Failed to download: $file_name" >> download_errors.log

        # Inspect the downloaded file
        if [[ -f "$dir/$file_name" ]]; then
            echo "File $file_name found. Checking MD5..."
            if check_md5 "$dir/$file_name" "$expected_md5"; then
                echo "$file_name download complete and valid."
            # Register the verified artifact
                target_files+=("$dir/$file_name")
                break
            else
                echo "Removing invalid file: $file_name"
                rm "$dir/$file_name"
            fi
        fi
    done
}

# Merge all target .bam files within the specified folder
merge_target_bams_in_folder() {
    local dir=$1
    local cell_line=$(basename $(dirname "$dir"))  # Parent directory gives the cell line
    local epi=$(basename "$dir" | cut -d'_' -f1)  # Folder prefix stands for epigenomic mark
    local output_bam="$dir/${cell_line}_${epi}.bam"  # Construct final BAM filename

    # Locate all target .bam files inside this folder
    local folder_target_bams=()
    for target_file in "${target_files[@]}"; do
        if [[ "$(dirname "$target_file")" == "$dir" && "$target_file" == *.bam ]]; then
            folder_target_bams+=("$target_file")
        fi
    done

    if [[ "${#folder_target_bams[@]}" -gt 1 ]]; then
        echo "Merging target BAM files in $dir into $output_bam..."
        samtools merge -f "$output_bam" "${folder_target_bams[@]}"
        echo "Merged target BAM files into $output_bam"
        merged_files+=("$output_bam")
    elif [[ "${#folder_target_bams[@]}" -eq 1 ]]; then
        echo "Only one target BAM file found in $dir. Copying to $output_bam..."
        cp "${folder_target_bams[0]}" "$output_bam"
        merged_files+=("$output_bam")
    else
        echo "No target BAM files found in $dir. Skipping merge."
    fi
}

# Convert a merged BAM file into BigWig format
convert_merged_bam_to_bigwig() {
    local bam_file=$1
    local cell_line=$2
    local output_dir="./${cell_line}/bigWig_files"

    # Ensure the BigWig output directory exists
    mkdir -p "$output_dir"

    # Index the BAM before conversion
    echo "Indexing $bam_file..."
    samtools index "$bam_file"

    # Produce the BigWig track
    local bigwig_file="$output_dir/$(basename ${bam_file%.bam}).bw"
    echo "Converting $bam_file to $bigwig_file..."
    bamCoverage -b "$bam_file" -o "$bigwig_file" --binSize 10 --normalizeUsing RPGC --effectiveGenomeSize 2913022398

    echo "Converted $bam_file to BigWig."
}

# Walk through each cell line file and download it
for path in "${!download_links[@]}"; do
    url="${download_links[$path]}"
    
    # Download into the appropriate folder
    download_file "$path" "$url"
done

# Build the set of folders that contain target files
declare -A target_folders
for target_file in "${target_files[@]}"; do
    folder=$(dirname "$target_file")
    target_folders["$folder"]=1
done

# Merge BAMs for each folder that has downloads
for dir in "${!target_folders[@]}"; do
    # Count how many BAM files live in this directory
    local bam_count=0
    for target_file in "${target_files[@]}"; do
        if [[ "$(dirname "$target_file")" == "$dir" && "$target_file" == *.bam ]]; then
            ((bam_count++))
        fi
    done
    
    # Only merge when there are BAM files present
    if [[ $bam_count -gt 0 ]]; then
        merge_target_bams_in_folder "$dir"
    fi
done

# Convert every merged BAM into BigWig
for merged_bam in "${merged_files[@]}"; do
    # Derive the cell line name for the output path
    cell_line=$(basename $(dirname $(dirname "$merged_bam")))
    convert_merged_bam_to_bigwig "$merged_bam" "$cell_line"
done

echo "Download, merging, and BigWig conversion of target files completed."