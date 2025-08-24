import pandas as pd
import os
from Bio import SeqIO
import logging
import gzip
import argparse
import sys
import pyarrow.parquet as pq
import pyarrow as pa
import gc

logger = logging.getLogger()
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.DEBUG,                   # Set logging level to INFO or DEBUG
        format='%(asctime)s - %(message)s',   # Customize format
        handlers=[logging.StreamHandler()],   # Output to console
        datefmt='%Y-%m-%d %H:%M:%S'           # Custom date format (excluding milliseconds)
    )

def read_parquet_smalltypes(path):
    # Read only needed columns
    table = pq.read_table(path, columns=['id','len','loss','bact_prob'])

    # Correct type casting: use parentheses for instances
    table = table.cast(pa.schema([
        ('id', pa.string()),
        ('len', pa.uint16()),
        ('loss', pa.float32()),
        ('bact_prob', pa.float32())
    ]))

    return table.to_pandas()

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)      # show all columns
    # pd.set_option("display.max_rows", None)         # show all rows (use with care!)
    pd.set_option("display.max_colwidth", None)     # show full content in each cell
    pd.set_option("display.width", 0)  
    
    parser = argparse.ArgumentParser(description="Merging two score files.")
    parser.add_argument("--accession_file", required=True, help="Path to file with accession numbers")
    parser.add_argument("--start", type=int, required=True, help="Start index (inclusive)")
    parser.add_argument("--stop", type=int, required=True, help="Stop index (exclusive)")
    parser.add_argument("--source_dir", type=str, help="Path to the SRA score dir")
    args = parser.parse_args()
    
    accession_filepath = args.accession_file
    start_idx = args.start
    stop_idx = args.stop
    source_dir = args.source_dir
    
    df = pd.read_csv(accession_filepath, header=None)
    num_rows = len(df)

    # Clamp stop_idx to num_rows
    if start_idx >= num_rows:
        raise ValueError(f"start_idx {start_idx} exceeds the number of rows ({num_rows}) in the file.")
    if stop_idx > num_rows:
        print(f"Warning: stop_idx {stop_idx} exceeds the number of rows ({num_rows}); clamping to {num_rows}.")
        stop_idx = num_rows

    selected_accessions = df.iloc[start_idx:stop_idx, 0].tolist()
    N = len(selected_accessions)
    
    for acc_idx, acc in enumerate(selected_accessions):
        logging.info(f'{acc_idx+1}/{N} ' +  '-'*150)   
        logging.info(acc)   
        accession_dir = os.path.join(source_dir, acc)
        score_files = [f for f in os.listdir(accession_dir) if  '_scores_len.parquet' in f]
        
        if any(['-good_long_scores' in f for f in score_files]):
            logging.info(f'good long-reads scores file already exists. {acc} skipped.')
            continue

        else:
            # read R1 and R2 good scores files and check their length
            good_files = [[f for f in score_files if f'good_{R_id}_scores_len' in f] for R_id in [1, 2]]
            
            for R_id, files in enumerate(good_files, start=1):
                for f in files:
                    file_path = os.path.join(accession_dir, f)
                    try:
                        size_bytes = os.path.getsize(file_path)
                        size_mb = size_bytes / (1024 * 1024)
                        logging.info(f'R{R_id} file: {f}, size: {size_mb:.2f} MB')
                    except FileNotFoundError:
                        logging.warning(f'File not found: {file_path}')
            
            if len(good_files[0]) == 0 or len(good_files[1]) == 0:
                logging.info(f'At least one of the R_scores files not found. {acc} skipped')
                continue
                
            elif len(good_files[0]) == 1 and len(good_files[1]) == 1:
                input_score_file = [good_files[id][0] for id in [0,1]]
                long_scores_file = f'{acc}_good_long_scores.tsv.gz'
                short_scores_file = f'{acc}_good_short_scores.tsv.gz'

                if os.path.exists(os.path.join(accession_dir, long_scores_file)):
                    logging.info(f"Output file '{long_scores_file}' exists. {acc} skipped.")
                    continue
                elif os.path.exists(os.path.join(accession_dir, long_scores_file.replace('gz', 'gzip'))):
                    logging.info(f"Output file '{long_scores_file.replace('gz', 'gzip')}' exists. {acc} skipped.")
                    continue
                else:
                    logging.info(f'{input_score_file} -> {long_scores_file} + {short_scores_file}')

                scores = [read_parquet_smalltypes(os.path.join(accession_dir, f)) for f in input_score_file]
                     
                if any([scores[id].shape[0] == 0 for id in [0, 1]]):
                    logging.info(f"One of the score files is empty. {acc} skipped.")
                    continue
                else:
                    for id in [0,1]:
                        logging.info(f"R{id+1} scores shape: {scores[id].shape}")
                        logging.info(f"R{id+1} scores columns: {scores[id].columns}")
        

                if scores[0].shape[0] != scores[1].shape[0]:
                    logging.info(f"!!! reads number are not equal in two scores files: {scores[0].shape[0]} and {scores[1].shape[0]} !!!")
                else:
                    reads_num = scores[0].shape[0]
                    
                # logging.info("Before renaming:")
                # for id in [0,1]:
                #     logging.info(f"R{id+1}_scores columns: {scores[id].columns}")
                
                for id in [0,1]:
                    scores[id] = scores[id].rename(columns={'loss': f'R{id+1}_loss',
                                                            'len': f'R{id+1}_len',
                                                            'bact_prob': f'R{id+1}_bact_prob'}) 
                
                # merge and change dimensionality of scores
                scores = scores[0].merge(scores[1], on='id')
                logging.info('Merged scores dataset:')
                logging.info(scores.shape[0])
                logging.info(scores.head(3))
                logging.info(scores.tail(3))
                                                   
                # filtering R1 and R2 reads with len >= 100 and calculating scores (mean or the longest read)
                scores.loc[(scores['R1_len'] >= 100) & (scores['R2_len'] >= 100), 'loss'] = (scores['R1_loss'] + scores['R2_loss']) / 2
                scores.loc[(scores['R1_len'] >= 100) & (scores['R2_len'] >= 100), 'bact_prob'] = (scores['R1_bact_prob'] + scores['R2_bact_prob']) / 2
                
                scores.loc[(scores['R1_len'] >= 100) & (scores['R2_len'] < 100), 'loss'] = scores['R1_loss']
                scores.loc[(scores['R1_len'] >= 100) & (scores['R2_len'] < 100), 'bact_prob'] = scores['R1_bact_prob']
                
                scores.loc[(scores['R1_len'] < 100) & (scores['R2_len'] >= 100), 'loss'] = scores['R2_loss']
                scores.loc[(scores['R1_len'] < 100) & (scores['R2_len'] >= 100), 'bact_prob'] = scores['R2_bact_prob']
                
                # filtering short scores
                short_scores = scores[scores['loss'].isna()]
                short_scores = short_scores.drop(columns=['loss', 'bact_prob'])
                short_scores.reset_index(drop=True, inplace=True)
                
                logging.info('Short scores:')
                logging.info(short_scores)

                # drop out NA scores
                scores.dropna(subset=['loss'], inplace=True)
                scores = scores.drop(columns=['R1_loss', 'R2_loss', 'R1_len', 'R2_len', 'R1_bact_prob', 'R2_bact_prob'])
                scores.reset_index(drop=True, inplace=True)
                
                logging.info('Long scores:')
                logging.info(scores)
                
                if not short_scores.shape[0]:
                    logging.info('All reads are good')
                else:
                    logging.info(f'Short read pairs: {short_scores.shape[0]}/{reads_num}')
                    try:
                        with gzip.open(os.path.join(accession_dir, short_scores_file), 'wt', encoding='utf-8') as f:
                            short_scores.to_csv(f, sep='\t', index=False)
                        logging.info(f"{short_scores_file} was successfully written.")
                    except Exception as e:
                        logging.error(f"Error writing {short_scores_file}: {e}")

                if not scores.shape[0]:
                    logging.info('All reads are short!')
                else:
                    try:
                        with gzip.open(os.path.join(accession_dir, long_scores_file), 'wt', encoding='utf-8') as f:
                            scores.to_csv(f, sep='\t', index=False)
                        logging.info(f"{long_scores_file} was successfully written.")
                    except Exception as e:
                        logging.error(f"Error writing {long_scores_file}: {e}")
            
            del scores  # delete the variable
            gc.collect()
                    
                
                
    # for acc in accessions:
   
        
    #     if any(['-good_scores' in f for f in score_files]):
    #         logging.info(f'good scores file already exists. {acc} skipped.')
    #         sys.exit(0)
            
    #     else:
            
    #         # read R1 and R2 good scores files and check their length
    #         good_files = [[f for f in score_files if f'good_{R_id}' in f] for R_id in ['1', '2']]
    #         logging.info(f'input files: {good_files}')
            
    #         if len(good_files[0]) == 0 or len(good_files[0]) == 0:
    #             logging.info(f'The merged scores file is missing, and at least one of the R_scores files is also absent. {acc} skipped')
    #             sys.exit(0)
                
    #         elif len(good_files[0]) == 1 and len(good_files[1]) == 1:
    #             input_score_file = [good_files[id][0] for id in [0,1]]
    #             long_scores_file = input_score_file[0].split('good_1')[0] + 'good_long_scores.parquet'
    #             short_scores_file = input_score_file[0].split('good_1')[0] + 'good_short_scores.parquet'
                
    #             if os.path.exists(os.path.join(accession_dir, long_scores_file)):
    #                 logging.info(f"Output file '{long_scores_file}' exists. {acc} skipped.")
    #                 sys.exit(0)
    #             else:
    #                 logging.info(f'{input_score_file} -> {long_scores_file} + {short_scores_file}')

    #             scores = [
    #                 pd.read_parquet(
    #                     os.path.join(accession_dir, f'{input_score_file[id]}'),
    #                     columns=['id', 'loss', 'bact_prob']
    #                 )
    #                 for id in [0, 1]
    #             ]
                
    #             for id in [0,1]:
    #                 # scores[0].reset_index(drop=True, inplace=True)
    #                 if scores[id].shape[0] == 0:
    #                     logging.info(f"Empty scores file: {input_score_file[id]}. {acc} skipped.")
    #                     sys.exit(0)
    #                 else:
    #                     logging.info(f"R{id+1} scores shape: {scores[id].shape}")
    #                     logging.info(f"R{id+1} scores columns: {scores[id].columns}")
                
                
    #             if scores[0].shape[0] != scores[1].shape[0]:
    #                 logging.info("reads number are not equal in two scores files, merging skipped!")
    #                 sys.exit(0)
    #             else:
    #                 reads_num = scores[0].shape[0]

    #             # Check and process
    #             reads_len = [None]*2
    #             for id in [0,1]:
    #                 if 'len' not in scores[id].columns:
    #                     logging.info(f"'len' column not found in R{id+1}_scores. Reading original R1 fastq file.")
    #                     try:
    #                         fastq_path = os.path.join(fastq_dir, acc)
    #                         fastq_file = [f for f in os.listdir(fastq_path) if f'{acc}-good_{id+1}.fastq' in f][0]
    #                         logging.info(f'{fastq_file} found')
    #                         reads_len[id] = get_reads_len(os.path.join(fastq_path, fastq_file))
    #                         logging.info(f'R{id+1} reads length dataset: {reads_len[id].shape[0]}')

    #                         if scores[id].shape[0] == reads_len[id].shape[0]:
    #                             scores[id] = scores[id].merge(reads_len[id], on='id')
    #                             logging.info(f'Merged R{id+1} score dataset:')
    #                             logging.info(scores[id])
    #                         else:
    #                             logging.info(f'R{id+1} scores: {scores[id].shape[0]}, R{id+1} reads len: {reads_len[id].shape[0]}')
    #                             logging.info(f'R{id+1} merging skipped due to length mismatch.')
    #                             sys.exit(0)
    #                     except IndexError:
    #                         logging.info(f'R{id+1} fastq file not found for {acc}. Skipping.')
    #                         sys.exit(0)
    #                 else:
    #                     logging.info(f"'len' column already present in R{id+1}_scores")
                
    #             logging.info("Before renaming:")
    #             for id in [0,1]:
    #                 logging.info(f"R{id+1}_scores columns: {scores[id].columns}")
                
    #             for id in [0,1]:
    #                 scores[id] = scores[id].rename(columns={'loss': f'R{id+1}_loss',
    #                                                         'len': f'R{id+1}_len',
    #                                                         'bact_prob': f'R{id+1}_bact_prob'})

                
    #             # merge and change dimensionality of scores
    #             scores = scores[0].merge(scores[1], on='id')
                
    #             logging.info(merged_scores.shape[0])
    #             logging.info(merged_scores.head(3))
    #             logging.info(merged_scores.tail(3))
                                                   
    #             # filtering R1 and R2 reads with len >= 100 and calculating scores (mean or the longest read)
    #             scores.loc[(scores['R1_len'] >= 100) & (scores['R2_len'] >= 100), 'loss'] = (scores['R1_loss'] + scores['R2_loss']) / 2
    #             scores.loc[(scores['R1_len'] >= 100) & (scores['R2_len'] < 100), 'loss'] = scores['R1_loss']
    #             scores.loc[(scores['R1_len'] < 100) & (scores['R2_len'] >= 100), 'loss'] = scores['R2_loss']
                
    #             # filtering R1 and R2 reads with len >= 100 and calculating scores (mean or the longest read)
    #             scores.loc[(scores['R1_len'] >= 100) & (scores['R2_len'] >= 100), 'bact_prob'] = (scores['R1_bact_prob'] + scores['R2_bact_prob']) / 2
    #             scores.loc[(scores['R1_len'] >= 100) & (scores['R2_len'] < 100), 'bact_prob'] = scores['R1_bact_prob']
    #             scores.loc[(scores['R1_len'] < 100) & (scores['R2_len'] >= 100), 'bact_prob'] = scores['R2_bact_prob']
                
    #             # filtering short scores
    #             short_scores = scores[scores['loss'].isna()]
    #             short_scores = short_scores.drop(columns=['loss'])
    #             short_scores.reset_index(drop=True, inplace=True)
    #             logging.info(short_scores)

    #             # drop out NA scores
    #             scores.dropna(subset=['loss'], inplace=True)
    #             scores = scores.drop(columns=['R1_loss', 'R2_loss', 'R1_len', 'R2_len'])
    #             scores.reset_index(drop=True, inplace=True)
    #             logging.info(scores)
                

    #             if not short_scores.shape[0]:
    #                 logging.info('All reads are good')
    #             else:
    #                 logging.info(f'Bad reads: {short_scores.shape[0]}/{reads_num}')
    #                 try:
    #                     with open(os.path.join(source_dir, short_scores_filename), 'w', encoding='utf-8') as file:
    #                         short_scores.to_csv(file, index=False)
    #                     logging.info(f"{short_scores_filename} was successfully written.")
    #                 except Exception as e:
    #                     logging.error(f"Error writing {short_scores_filename}: {e}")
    #             try:
    #                 with open(os.path.join(source_dir, long_scores_filename), 'w', encoding='utf-8') as file:
    #                     scores.to_pandas(file, index=False)
    #                 logging.info(f"{long_scores_filename} was successfully written.")
    #             except Exception as e:
    #                 logging.error(f"Error writing {long_scores_filename}: {e}")