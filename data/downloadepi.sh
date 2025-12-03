#!/bin/bash

# Function to retrieve the md5sum from an ENCODE file page
get_first_md5_from_json() {
    local url=$1

    # Use curl to fetch JSON and jq to extract the md5sum
    md5_value=$(curl -s "$url" | jq -r '.md5sum')

    if [[ -n "$md5_value" ]]; then
        echo "$md5_value"
    else
        echo "MD5 not found for $url"
    fi
}

# Define cell lines and their download links
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
    IMR90
    ["IMR90/ctcf_chip-seq/ENCFF584CBK"]="https://www.encodeproject.org/files/ENCFF584CBK/@@download/ENCFF584CBK.bam"
    ["IMR90/ctcf_chip-seq/ENCFF160QTH"]="https://www.encodeproject.org/files/ENCFF160QTH/@@download/ENCFF160QTH.bam"
    ["IMR90/H3K4me3_chip-seq/ENCFF355WLE"]="https://www.encodeproject.org/files/ENCFF355WLE/@@download/ENCFF355WLE.bam"
    ["IMR90/H3K4me3_chip-seq/ENCFF021CLP"]="https://www.encodeproject.org/files/ENCFF021CLP/@@download/ENCFF021CLP.bam"
    ["IMR90/H3K27me3_chip-seq/ENCFF547BFL"]="https://www.encodeproject.org/files/ENCFF547BFL/@@download/ENCFF547BFL.bam"
    ["IMR90/H3K27me3_chip-seq/ENCFF084CZB"]="https://www.encodeproject.org/files/ENCFF084CZB/@@download/ENCFF084CZB.bam"
    ["IMR90/H3K27ac_chip-seq/ENCFF064XTS"]="https://www.encodeproject.org/files/ENCFF064XTS/@@download/ENCFF064XTS.bam"
    ["IMR90/H3K27ac_chip-seq/ENCFF827KCG"]="https://www.encodeproject.org/files/ENCFF827KCG/@@download/ENCFF827KCG.bam"
    ["IMR90/RAD21_chip-seq/ENCFF918IXN"]="https://www.encodeproject.org/files/ENCFF918IXN/@@download/ENCFF918IXN.bam"
    ["IMR90/RAD21_chip-seq/ENCFF824JOL"]="https://www.encodeproject.org/files/ENCFF824JOL/@@download/ENCFF824JOL.bam"
    ["IMR90/DNase-seq/ENCFF264LKY"]="https://www.encodeproject.org/files/ENCFF264LKY/@@download/ENCFF264LKY.bam"

    # H1-hESC
    ["H1-hESC/ctcf_chip-seq/ENCFF767YLK"]="https://www.encodeproject.org/files/ENCFF767YLK/@@download/ENCFF767YLK.bam"
    ["H1-hESC/H3K4me3_chip-seq/ENCFF638NIJ"]="https://www.encodeproject.org/files/ENCFF638NIJ/@@download/ENCFF638NIJ.bam"
    ["H1-hESC/H3K4me3_chip-seq/ENCFF940GKD"]="https://www.encodeproject.org/files/ENCFF940GKD/@@download/ENCFF940GKD.bam"
    ["H1-hESC/H3K27me3_chip-seq/ENCFF495SRZ"]="https://www.encodeproject.org/files/ENCFF495SRZ/@@download/ENCFF495SRZ.bam"
    ["H1-hESC/H3K27me3_chip-seq/ENCFF915QOK"]="https://www.encodeproject.org/files/ENCFF915QOK/@@download/ENCFF915QOK.bam"
    ["H1-hESC/H3K27ac_chip-seq/ENCFF324DHZ"]="https://www.encodeproject.org/files/ENCFF324DHZ/@@download/ENCFF324DHZ.bam"
    ["H1-hESC/H3K27ac_chip-seq/ENCFF262JMM"]="https://www.encodeproject.org/files/ENCFF262JMM/@@download/ENCFF262JMM.bam"
    ["H1-hESC/RAD21_chip-seq/ENCFF200HXK"]="https://www.encodeproject.org/files/ENCFF200HXK/@@download/ENCFF200HXK.bam"
    ["H1-hESC/RAD21_chip-seq/ENCFF100EES"]="https://www.encodeproject.org/files/ENCFF100EES/@@download/ENCFF100EES.bam"
    ["H1-hESC/DNase-seq/ENCFF175QBB"]="https://www.encodeproject.org/files/ENCFF175QBB/@@download/ENCFF175QBB.bam"

    # # K562
    ["K562/ctcf_chip-seq/ENCFF111JKR"]="https://www.encodeproject.org/files/ENCFF111JKR/@@download/ENCFF111JKR.bam"
    ["K562/ctcf_chip-seq/ENCFF945CJG"]="https://www.encodeproject.org/files/ENCFF945CJG/@@download/ENCFF945CJG.bam"
    ["K562/ctcf_chip-seq/ENCFF706PLK"]="https://www.encodeproject.org/files/ENCFF706PLK/@@download/ENCFF706PLK.bam"
    ["K562/H3K4me3_chip-seq/ENCFF181ANT"]="https://www.encodeproject.org/files/ENCFF181ANT/@@download/ENCFF181ANT.bam"
    ["K562/H3K4me3_chip-seq/ENCFF747HEB"]="https://www.encodeproject.org/files/ENCFF747HEB/@@download/ENCFF747HEB.bam"
    ["K562/H3K27me3_chip-seq/ENCFF652TXG"]="https://www.encodeproject.org/files/ENCFF652TXG/@@download/ENCFF652TXG.bam"
    ["K562/H3K27me3_chip-seq/ENCFF508LLH"]="https://www.encodeproject.org/files/ENCFF508LLH/@@download/ENCFF508LLH.bam"
    ["K562/H3K27ac_chip-seq/ENCFF600THN"]="https://www.encodeproject.org/files/ENCFF600THN/@@download/ENCFF600THN.bam"
    ["K562/H3K27ac_chip-seq/ENCFF704LGA"]="https://www.encodeproject.org/files/ENCFF704LGA/@@download/ENCFF704LGA.bam"
    ["K562/H3K27ac_chip-seq/ENCFF232RQF"]="https://www.encodeproject.org/files/ENCFF232RQF/@@download/ENCFF232RQF.bam"
    ["K562/RAD21_chip-seq/ENCFF636DAM"]="https://www.encodeproject.org/files/ENCFF636DAM/@@download/ENCFF636DAM.bam"
    ["K562/RAD21_chip-seq/ENCFF099DUX"]="https://www.encodeproject.org/files/ENCFF099DUX/@@download/ENCFF099DUX.bam"
    ["K562/DNase-seq/ENCFF533TKL"]="https://www.encodeproject.org/files/ENCFF533TKL/@@download/ENCFF533TKL.bam"
    ["K562/POLR2A/ENCFF380XSC"]="https://www.encodeproject.org/files/ENCFF380XSC/@@download/ENCFF380XSC.bam"
    ["K562/POLR2A/ENCFF717URX"]="https://www.encodeproject.org/files/ENCFF717URX/@@download/ENCFF717URX.bam"
)



# Initialize MD5 dictionary
declare -A md5_values=()

# Retrieve and cache MD5 for each file
for path in "${!download_links[@]}"; do
    file_id=$(basename "$path")
    json_url="https://www.encodeproject.org/files/${file_id}/?format=json"
    md5=$(get_first_md5_from_json "$json_url")
    
    # Print fetched MD5 value
    echo "Retrieved MD5 for $path: $md5"
    
    md5_values["$path"]="$md5"
done


# Validate file integrity via MD5
check_md5() {
    local file=$1
    local expected_md5=$2

    # Compute local file MD5
    local actual_md5=$(md5sum "$file" | awk '{print $1}')

    if [[ "$actual_md5" == "$expected_md5" ]]; then
        echo "$file MD5 matches."
        return 0
    else
        echo "Warning: $file MD5 does not match."
        return 1
    fi
}

# Verify BAM integrity with samtools
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

# Download helper: creates folders, resumes partial downloads, skips valid files
download_file() {
    local dir=$(dirname "$1")
    local file_name=$(basename "$url")
    local url=$2

    mkdir -p "$dir"

    # Lookup expected MD5 for this file
    local expected_md5=${md5_values["$1"]}

    # Loop until the downloaded file passes MD5 validation
    while true; do
        # If file exists locally, validate via MD5 first
        if [[ -f "$dir/$file_name" ]]; then
            if check_md5 "$dir/$file_name" "$expected_md5"; then
                echo "$file_name already exists and is valid. Skipping download."
                break
            else
                echo "$file_name is invalid. Re-downloading..."
                rm "$dir/$file_name"
            fi
        fi

        # Download file (quiet mode)
        wget -c -P "$dir" "$url" || echo "Failed to download: $file_name" >> download_errors.log

        # Validate newly downloaded file
        if [[ -f "$dir/$file_name" ]]; then
            echo "File $file_name found. Checking MD5..."
            if check_md5 "$dir/$file_name" "$expected_md5"; then
                echo "$file_name download complete and valid."
                break
            else
                echo "Removing invalid file: $file_name"
                rm "$dir/$file_name"
            fi
        fi
    done
}

# Merge all BAMs within the same folder
merge_bams_in_folder() {
    local dir=$1
    local cell_line=$(basename $(dirname "$dir"))  # Cell line name (parent folder)
    local epi=$(basename "$dir" | cut -d'_' -f1)  # Epigenetic mark prefix from folder name
    local output_bam="$dir/${cell_line}_${epi}.bam"  # Merged BAM output path

    # Find all BAM files in the folder
    bam_files=($(find "$dir" -maxdepth 1 -type f -name "*.bam"))

    if [[ "${#bam_files[@]}" -gt 1 ]]; then
        echo "Merging BAM files in $dir into $output_bam..."
        samtools merge -f "$output_bam" "${bam_files[@]}"
        echo "Merged BAM files into $output_bam"
    else
        echo "Only one BAM file found in $dir. Copying to $output_bam..."
        cp "${bam_files[0]}" "$output_bam"
    fi
}

# Convert merged BAM files to BigWig
convert_bam_to_bigwig() {
    local bam_file=$1
    local cell_line=$2
    local output_dir="./${cell_line}/bigWig_files"

    # Create output folder for BigWigs
    mkdir -p "$output_dir"

    # Convert BAM to BigWig
    local bigwig_file="$output_dir/$(basename ${bam_file%.bam}).bw"
    echo "Converting $bam_file to $bigwig_file..."
    bamCoverage -b "$bam_file" -o "$bigwig_file" --binSize 10 --normalizeUsing RPGC --effectiveGenomeSize 2913022398

    echo "Converted $bam_file to BigWig."
}

# Download files for every cell line/link pair
for path in "${!download_links[@]}"; do
    url="${download_links[$path]}"
    
    # Download into the correct folder
    download_file "$path" "$url"
done

# Merge BAMs within each *_chip-seq folder
for dir in $(find . -type d -name "*_chip-seq"); do
    merge_bams_in_folder "$dir"
done


echo "Download, merging completed."