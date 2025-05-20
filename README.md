# BERT-DNA-classification
1. Environment.  
    1.1.  Create BBERT environment from .yml file:
    ```bash
    conda env create -f BBERT_env.yml --yes  
    ```
    or  
    ```bash
        conda create -n BBERT python=3.10  
        conda activate BBERT  
        conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia  
        conda install -c conda-forge transformers=4.30.2  
        conda install seaborn  
        conda install scikit-learn  
    ```
    1.2.  Activate env and check the installation using script like that (sbatch):  

    ```bash  
        #!/bin/bash  
        #SBATCH --output=BBERT_venv_ckeck_%A.txt  
        #SBATCH --gres=gpu:1  
        #SBATCH --time=01:00:00  
        #SBATCH --mem=8G  
        echo "Checking CUDA availability for PyTorch and TensorFlow..."  
        python3 - <<END  
        import torch  
        print("PyTorch CUDA available:", torch.cuda.is_available())  
        END  
    ```

2. Training.  
    Script:  `/source/train_cont.py`  
    To start the training process:  
    - If training the model from scratch, set all base parameters.  
    - If loading the model from a checkpoint, also specify the model name, batch size, and number of epochs (if needed).  

3. Scoring.  
    Script:  `/source/score.py $input_file.fasta $output_file.csv`  
    The model to be used should be checked before running the script.  
    To calculate scores for R1.fasta and R2.fasta, the scripts should be run separately.  

4. Labeling scores.  
    Script:  `/source/label_scores_R1_R2.py`    
    Input:  R1.fasta, R2.fasta, labels.csv  
    Output  - .csv file:  
    - 'base_id'   - read id (without /1 and /2 suffix)  
    - 'bact'      - true bacteria label  
    - 'score'     - mean score for two reads from R1.fasta and R2.fasta  

5. Cut point calculation.  
    Script:  `/source/cut_point_calc_mult.py`  
    Calculating the cut points and accuracy for a set of labeled scores.  
    Plotting the results.  

6. Benchmarks.  
    Script:  `/source/bertax_comparison.py`  
    Comparison of classification performance between BBERT and BERTax on a set of testing datasets.  

7. Test datasets preparation.
    7.1.  Script: `/source/ncbi-fna-iss-fastq-fasta.py`  
    Ncbi -> .fna files -> iss pricessing -> fastq files -> conversion to .fasta:  
    - obtaining a list of relevant bacterial and eukaryotic .fna files from NCBI.  
    - filtering out .fna wich genus intersects with Bertax trining datasets.  
    - downloading zip -> extracting .fna  
    - using 'iss generate' tool to generate .fastq files  
    - converting .fastq to .fasta and tgimming reads to 100 bases  

    7.2.  Script:  `/source/gen_datasets_R1_R2.py`
    Generation of 20 datasets, each containing 50 bact and 50 euk samples from generated .fasta files, with a 50/50 bact/euk ratio and a lognormal distribution.  
