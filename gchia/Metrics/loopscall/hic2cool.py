# Convert hic to cool
import subprocess
import os
import logging
import numpy as np
import cooler

def convert_hic_to_cool(input_file, output_file, resolution=None, num_proc=1):
    """
    Convert .hic file to .cool file using hic2cool command
    
    Parameters
    ----------
    input_file : str
        Path to input .hic file
    output_file : str
        Path to output .cool file
    resolution : int, optional
        Resolution of the output cool file in base pairs
    num_proc : int, optional
        Number of processes to use for conversion
    
    Returns
    -------
    bool
        True if conversion was successful, False otherwise
    """
    try:
        cmd = ['hic2cool', 'convert', input_file, output_file]
        
        if resolution:
            cmd.extend(['-r', str(resolution)])
        
        if num_proc > 1:
            cmd.extend(['-p', str(num_proc)])
        
        logging.info(f"Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if os.path.exists(output_file):
            logging.info(f"Successfully converted {input_file} to {output_file}")
            return True
        else:
            logging.error(f"Converted file {output_file} does not exist")
            return False
            
    except subprocess.CalledProcessError as e:
        logging.error(f"Conversion failed: {e.stderr.decode('utf-8') if e.stderr else str(e)}")
        return False
    except Exception as e:
        logging.error(f"Error occurred during conversion: {str(e)}")
        return False

def add_weight_column(fcool):
    '''
    Add weight column to a cool file with all bins' bias set to 1.0
    
    Parameters
    ----------
    fcool : str
        Path to the cool file
    '''
    clr = cooler.Cooler(fcool)
    n_bins = clr.bins().shape[0]

    if 'weight' not in clr.bins().columns:
        h5opts = dict(compression='gzip', compression_opts=6)
        with clr.open('r+') as f:
            # Create a weight column
            f['bins'].create_dataset('weight', data=np.ones(n_bins), **h5opts)
        logging.info(f"Added weight column to {fcool}")
    else:
        logging.info(f"Weight column already exists in {fcool}")

def balance_matrix(fcool, **kwargs):
    '''
    Perform matrix balancing on a cool file using the cooler balance command
    
    Parameters
    ----------
    fcool : str
        Path to the cool file
    **kwargs : dict
        Additional parameters to pass to cooler balance command
    
    Returns
    -------
    bool
        True if balancing was successful, False otherwise
    '''
    try:
        logging.info(f"Balancing matrix in {fcool}")
        cmd = ['cooler', 'balance', fcool]
        
        # Add optional command line parameters
        for key, value in kwargs.items():
            logging.info(f"Adding parameter: {key}={value}")
            if isinstance(value, bool):
                if value:  # Only add flag if True
                    cmd.append(f'--{key.replace("_", "-")}')
            else:
                cmd.append(f'--{key.replace("_", "-")}')
                cmd.append(str(value))
        
        logging.info(f"Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logging.info(f"Successfully balanced matrix in {fcool}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Balancing failed: {e.stderr.decode('utf-8') if e.stderr else str(e)}")
        return False
    except Exception as e:
        logging.error(f"Error occurred during matrix balancing: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert .hic files to .cool files')
    parser.add_argument('input', help='Path to input .hic file')
    parser.add_argument('output', help='Path to output .cool file')
    parser.add_argument('-r', '--resolution', type=int, help='Resolution of output file in base pairs')
    parser.add_argument('-p', '--processes', type=int, default=1, help='Number of processes to use for conversion')
    parser.add_argument('-w', '--add-weights', action='store_true', help='Add weight column (all 1.0) after conversion')
    parser.add_argument('-b', '--balance', action='store_true', help='Balance matrix after conversion')
    parser.add_argument('--balance-only', action='store_true', help='Only balance existing cool file')
    parser.add_argument('--cis-only', action='store_true', help='Balance only cis interactions')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.balance_only:
        if os.path.exists(args.input):
            balance_kwargs = {}
            if args.cis_only:
                balance_kwargs['cis_only'] = True
                
            success = balance_matrix(args.input, **balance_kwargs)
            if success:
                print(f"Successfully balanced matrix in: {args.input}")
            else:
                print("Matrix balancing failed, see log for details")
                exit(1)
        else:
            print(f"Error: File {args.input} does not exist")
            exit(1)
    else:
        success = convert_hic_to_cool(args.input, args.output, args.resolution, args.processes)
        
        if success:
            print(f"Conversion successful: {args.output}")
            
            if args.balance:
                balance_kwargs = {}
                if args.cis_only:
                    balance_kwargs['cis_only'] = True
                    
                success = balance_matrix(args.output, **balance_kwargs)
                if success:
                    print(f"Successfully balanced matrix in: {args.output}")
                else:
                    print("Matrix balancing failed, see log for details")
        else:
            print("Conversion failed, see log for details")
            exit(1)
