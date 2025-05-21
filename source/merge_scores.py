
import pandas as pd
import os
from Bio import SeqIO
import logging
import gzip
import argparse
import sys

logger = logging.getLogger()
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.DEBUG,                    # Set logging level to INFO or DEBUG
        format='%(asctime)s - %(message)s',     # Customize format
        handlers=[logging.StreamHandler()],     # Output to console
        datefmt='%Y-%m-%d %H:%M:%S'             # Custom date format (excluding milliseconds)
    )

parser = argparse.ArgumentParser(description="Merging twe R_score files.")
parser.add_argument("scores_dir", type=str, help="Path to the SRA score dir")
parser.add_argument("fastq_dir", type=str, help="Path to the FASTA, FASTQ or GZIP original input file")
    
args = parser.parse_args()
scores_dir = args.scores_dir
fastq_dir = args.fastq_dir

def get_reads_len(file_path):
    
    # Determine file type and open accordingly
    if file_path.endswith('.gz'):
        open_func = lambda x: gzip.open(x, 'rt')
        file_format = "fastq"
    elif file_path.endswith('.fasta'):
        open_func = lambda x: open(x, 'rt')
        file_format = "fasta"
    elif file_path.endswith('.fastq'):
        open_func = lambda x: open(x, 'rt')
        file_format = "fastq"
    else:
        raise ValueError("Unsupported file format. Only .gz, .fasta, and .fastq files are supported.")
    
    ids = []
    reads_len = []
    with open_func(file_path) as handle:
        for read in SeqIO.parse(handle, file_format):
            ids.append(read.id)
            reads_len.append(len(read.seq))
            
    return pd.DataFrame({'id': ids, 'len': reads_len})

if __name__ == "__main__":
    sra = os.path.basename(scores_dir)
    score_files = [f for f in os.listdir(scores_dir) if  '_scores.csv' in f]
    logging.info(score_files)

    if any(['-good_long_scores' in f for f in score_files]):
        logging.info(f'good long-reads scores file already exists. {sra} skipped.')
        sys.exit(0)
    else:
        # read R1 and R2 good scores files and check their length
        R1_files = [f for f in score_files if 'good_1' in f]
        R2_files = [f for f in score_files if 'good_2' in f]
        if len(R1_files) == 0 or len(R2_files) == 0:
            logging.info(f'The merged scores file is missing, and at least one of the R_scores files is also absent. {sra} skipped')
            sys.exit(0)
        elif len(R1_files) == 1 and len(R2_files) == 1:
            R1_file = R1_files[0]
            R2_file = R2_files[0]
            long_scores_filename = R1_file.split('good_1')[0] + 'good_long_scores.csv'
            short_scores_filename = R1_file.split('good_1')[0] + 'good_short_scores.csv'

            if os.path.exists(os.path.join(scores_dir, long_scores_filename)):
                logging.info(f"Output file '{long_scores_filename}' exists. {sra} skipped.")
                sys.exit(0)
            else:
                logging.info(f"Merging {R1_file} and {R2_file}")

                R1_scores = pd.read_csv(os.path.join(scores_dir, R1_file)).drop_duplicates().reset_index(drop=True)
                R2_scores = pd.read_csv(os.path.join(scores_dir, R2_file)).drop_duplicates().reset_index(drop=True)
                logging.info(f"R1 original reads number: {R1_scores.shape[0]}")
                logging.info(f"R2 original reads number: {R2_scores.shape[0]}")

                logging.info(f"R1 original columns: {list(R1_scores.columns)}")
                logging.info(f"R2 original columns: {list(R2_scores.columns)}")
                
                if R1_scores.shape[0] != R2_scores.shape[0]:
                    logging.info("reads number are not equal in two scores files, merging skipped!")
                    sys.exit(0)
                else:
                    reads_num = R1_scores.shape[0]

                # Check and process R1
                if 'len' not in R1_scores.columns:
                    logging.info("'len' column not found in R1_scores. Reading original R1 fastq file.")
                    try:
                        R1_fastq_file = [f for f in os.listdir(fastq_dir) if f'{sra}-good_1.fastq' in f][0]
                        logging.info(f'{R1_fastq_file} found')
                        R1_reads_len = get_reads_len(os.path.join(fastq_dir, R1_fastq_file))
                        logging.info('R1 reads length dataset:')
                        logging.info(R1_reads_len)
                        if R1_scores.shape[0] == R1_reads_len.shape[0]:
                            R1_scores = R1_scores.merge(R1_reads_len, on='id')
                        else:
                            logging.info(f'R1 scores: {R1_scores.shape[0]}, R1 reads len: {R1_reads_len.shape[0]}')
                            logging.info('R1 merging skipped due to length mismatch.')
                            sys.exit(0)
                    except IndexError:
                        logging.info(f'R1 fastq file not found for {sra}. Skipping.')
                        sys.exit(0)
                else:
                    logging.info("'len' column already present in R1_scores")

                # Check and process R2
                if 'len' not in R2_scores.columns:
                    logging.info("'len' column not found in R2_scores. Reading original R2 fastq file.")
                    try:
                        R2_fastq_file = [f for f in os.listdir(fastq_dir) if f'{sra}-good_2.fastq' in f][0]
                        logging.info(f'{R2_fastq_file} found')
                        R2_reads_len = get_reads_len(os.path.join(fastq_dir, R2_fastq_file))
                        logging.info('R2 reads length dataset:')
                        logging.info(R2_reads_len)
                        if R2_scores.shape[0] == R2_reads_len.shape[0]:
                            R2_scores = R2_scores.merge(R2_reads_len, on='id')
                        else:
                            logging.info(f'R2 scores: {R2_scores.shape[0]}, R2 reads len: {R2_reads_len.shape[0]}')
                            logging.info('R2 merging skipped due to length mismatch.')
                            sys.exit(0)
                    except IndexError:
                        logging.info(f'R2 fastq file not found for {sra}. Skipping.')
                        sys.exit(0)
                else:
                    logging.info("'len' column already present in R2_scores")

                print("Before renaming:")
                print("R1_scores columns:", R1_scores.columns.tolist())
                print("R2_scores columns:", R2_scores.columns.tolist())
                                
                R1_scores = R1_scores.rename(columns={'score': 'R1_score', 'len': 'R1_len'})
                R2_scores = R2_scores.rename(columns={'score': 'R2_score', 'len': 'R2_len'})
                scores = R1_scores.merge(R2_scores, on='id')
                
                print("After renaming:")
                print("R1_scores columns:", R1_scores.columns.tolist())
                print("R2_scores columns:", R2_scores.columns.tolist())
                
                logging.info(scores)
                        
                # filtering R1 and R2 reads with len >= 100 and calculating scores (mean or the longest read)
                scores.loc[(scores['R1_len'] >= 100) & (scores['R2_len'] >= 100), 'score'] = (scores['R1_score'] + scores['R2_score']) / 2
                scores.loc[(scores['R1_len'] >= 100) & (scores['R2_len'] < 100), 'score'] = scores['R1_score']
                scores.loc[(scores['R1_len'] < 100) & (scores['R2_len'] >= 100), 'score'] = scores['R2_score']
                
                # filtering short scores
                short_scores = scores[scores['score'].isna()]
                short_scores = short_scores.drop(columns=['score'])
                short_scores.reset_index(drop=True, inplace=True)
                logging.info(short_scores)

                # drop out NA scores
                scores.dropna(subset=['score'], inplace=True)
                scores = scores.drop(columns=['R1_score', 'R2_score', 'R1_len', 'R2_len'])
                scores.reset_index(drop=True, inplace=True)
                logging.info(scores)
                

                if not short_scores.shape[0]:
                    logging.info('All reads are good')
                else:
                    logging.info(f'Bad reads: {short_scores.shape[0]}/{reads_num}')
                    try:
                        with open(os.path.join(scores_dir, short_scores_filename), 'w', encoding='utf-8') as file:
                            short_scores.to_csv(file, index=False, float_format='%.6f')
                        logging.info(f"{short_scores_filename} was successfully written.")
                    except Exception as e:
                        logging.error(f"Error writing {short_scores_filename}: {e}")
                try:
                    with open(os.path.join(scores_dir, long_scores_filename), 'w', encoding='utf-8') as file:
                        scores.to_csv(file, index=False, float_format='%.6f')
                    logging.info(f"{long_scores_filename} was successfully written.")
                except Exception as e:
                    logging.error(f"Error writing {long_scores_filename}: {e}")