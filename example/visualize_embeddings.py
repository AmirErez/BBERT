#!/usr/bin/env python3
"""
t-SNE Visualization of BBERT Embeddings
Analyzes embeddings from Pseudomonas and Saccharomyces samples,
colors by organism type and coding/non-coding classification.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path
import argparse

def extract_organism_info(seq_id):
    """Extract organism and coding information from sequence ID"""
    if 'Pseudomonas' in seq_id or 'NZ_HG974234' in seq_id:
        organism = 'Pseudomonas'
    elif 'Saccharomyces' in seq_id or 'NC_047487' in seq_id:
        organism = 'Saccharomyces'
    else:
        organism = 'Unknown'
    
    # Extract coding information from CDS_info field
    if 'CDS_100.0%_coding' in seq_id:
        coding_status = 'Coding'
    elif 'Non-CDS_0.0%_coding' in seq_id:
        coding_status = 'Non-coding'
    elif 'CDS_' in seq_id:
        coding_status = 'Coding'
    else:
        coding_status = 'Unknown'
    
    return organism, coding_status

def load_embeddings(results_dir):
    """Load embeddings from all parquet files"""
    results_path = Path(results_dir)
    parquet_files = list(results_path.glob("*_scores_len_emb.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No embedding parquet files found in {results_dir}")
    
    all_data = []
    
    for file in parquet_files:
        print(f"Loading {file.name}...")
        df = pd.read_parquet(file)
        
        # Extract sample info from filename
        filename = file.name
        if 'Pseudomonas' in filename:
            if 'R1' in filename:
                sample = 'Pseudomonas_R1'
            else:
                sample = 'Pseudomonas_R2'
        elif 'Saccharomyces' in filename:
            if 'R1' in filename:
                sample = 'Saccharomyces_R1'
            else:
                sample = 'Saccharomyces_R2'
        else:
            sample = filename.replace('_scores_len_emb.parquet', '')
        
        df['sample'] = sample
        all_data.append(df)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total sequences loaded: {len(combined_df)}")
    
    return combined_df

def prepare_data(df, max_samples_per_group=1000):
    """Prepare data for t-SNE analysis"""
    # Extract organism from sample name (which comes from filename)
    def get_organism_from_sample(sample):
        if 'Pseudomonas' in sample:
            return 'Pseudomonas'
        elif 'Saccharomyces' in sample:
            return 'Saccharomyces'
        else:
            return 'Unknown'
    
    # Extract coding and frame information from BBERT predictions
    def get_coding_from_predictions(coding_prob):
        # Use BBERT's coding prediction (>0.5 = coding)
        return 'Coding' if coding_prob > 0.5 else 'Non-coding'
    
    def get_frame_from_predictions(frame_prob):
        # Get the predicted reading frame and convert to biological frame
        label = np.argmax(frame_prob)  # 0-5 model labels
        # Convert label to biological frame using BBERT's mapping
        if label >= 3:
            bio_frame = label - 2  # Labels 3,4,5 → Frames +1,+2,+3
        else:
            bio_frame = label - 3  # Labels 0,1,2 → Frames -3,-2,-1
        return bio_frame
    
    def get_detailed_category(coding_prob, frame_prob):
        if coding_prob > 0.5:
            label = np.argmax(frame_prob)
            # Convert to biological frame
            if label >= 3:
                bio_frame = label - 2  # Labels 3,4,5 → Frames +1,+2,+3
                return f'Frame_+{bio_frame}'
            else:
                bio_frame = label - 3  # Labels 0,1,2 → Frames -3,-2,-1
                return f'Frame_{bio_frame}'  # Already negative
        else:
            return 'Non-coding'
    
    # Apply the functions
    df['organism'] = df['sample'].apply(get_organism_from_sample)
    df['coding_status'] = df['coding_prob'].apply(get_coding_from_predictions)
    df['frame'] = df['frame_prob'].apply(get_frame_from_predictions)
    df['detailed_category'] = df.apply(lambda row: get_detailed_category(row['coding_prob'], row['frame_prob']), axis=1)
    
    # Filter out unknowns (only organism unknowns, since coding is now from predictions)
    df = df[df['organism'] != 'Unknown'].copy()  # Use .copy() to avoid SettingWithCopyWarning
    print(f"After filtering unknowns: {len(df)} sequences")
    
    # Create combined category for coloring (now includes frame information)
    df['category'] = df['organism'] + '_' + df['detailed_category']
    
    # Sample data to make t-SNE computation feasible  
    # Now we expect up to 14 categories (2 organisms × 7 detailed categories each)
    if len(df) > max_samples_per_group * 14:  
        print(f"Sampling data to {max_samples_per_group} per category...")
        sampled_dfs = []
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            if len(category_df) > max_samples_per_group:
                category_df = category_df.sample(n=max_samples_per_group, random_state=42)
            sampled_dfs.append(category_df)
        df = pd.concat(sampled_dfs, ignore_index=True)
        print(f"Final dataset size: {len(df)}")
    
    return df

def run_tsne(embeddings, perplexity=30, n_iter=1000):
    """Run t-SNE on embeddings"""
    print("Running t-SNE...")
    print(f"Embedding shape: {embeddings.shape}")
    
    # Adjust perplexity if we have too few samples
    n_samples = embeddings.shape[0]
    perplexity = min(perplexity, max(5, n_samples // 4))
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=42,
        verbose=1
    )
    
    tsne_result = tsne.fit_transform(embeddings)
    print(f"t-SNE completed. Final embedding shape: {tsne_result.shape}")
    
    return tsne_result

def create_visualization(df, tsne_result, output_dir):
    """Create and save t-SNE visualization"""
    # Add t-SNE coordinates to dataframe
    df['tsne_1'] = tsne_result[:, 0]
    df['tsne_2'] = tsne_result[:, 1]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('t-SNE Visualization of BBERT Embeddings', fontsize=16, fontweight='bold')
    
    # Color palettes
    organism_colors = {'Pseudomonas': '#1f77b4', 'Saccharomyces': '#ff7f0e', 'Unknown': 'gray'}
    coding_colors = {'Coding': '#2ca02c', 'Non-coding': '#d62728', 'Unknown': 'gray'}
    
    # Plot 1: Color by organism
    ax1 = axes[0, 0]
    for organism in df['organism'].unique():
        mask = df['organism'] == organism
        ax1.scatter(df.loc[mask, 'tsne_1'], df.loc[mask, 'tsne_2'], 
                   c=organism_colors.get(organism, 'gray'), label=organism, alpha=0.6, s=20)
    ax1.set_title('Colored by Organism')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Color by coding status
    ax2 = axes[0, 1]
    for coding in df['coding_status'].unique():
        mask = df['coding_status'] == coding
        ax2.scatter(df.loc[mask, 'tsne_1'], df.loc[mask, 'tsne_2'], 
                   c=coding_colors.get(coding, 'gray'), label=coding, alpha=0.6, s=20)
    ax2.set_title('Colored by Coding Status')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Color by detailed category (organism + frame/non-coding)
    ax3 = axes[1, 0]
    categories = df['category'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))  # More colors for more categories
    for i, category in enumerate(categories):
        mask = df['category'] == category
        ax3.scatter(df.loc[mask, 'tsne_1'], df.loc[mask, 'tsne_2'], 
                   c=[colors[i]], label=category, alpha=0.6, s=20)
    ax3.set_title('Colored by Organism + Frame/Non-coding')
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Sample distribution
    ax4 = axes[1, 1]
    for sample in df['sample'].unique():
        mask = df['sample'] == sample
        ax4.scatter(df.loc[mask, 'tsne_1'], df.loc[mask, 'tsne_2'], 
                   label=sample, alpha=0.6, s=20)
    ax4.set_title('Colored by Sample')
    ax4.set_xlabel('t-SNE 1')
    ax4.set_ylabel('t-SNE 2')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'bbert_tsne_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Skip saving CSV - too large for practical use
    # data_output_path = Path(output_dir) / 'tsne_results.csv'
    # df.to_csv(data_output_path, index=False)
    # print(f"Data with t-SNE coordinates saved to: {data_output_path}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total sequences: {len(df)}")
    print("\nBy organism:")
    print(df['organism'].value_counts())
    print("\nBy coding status:")
    print(df['coding_status'].value_counts())
    print("\nBy category:")
    print(df['category'].value_counts())
    
    # Close the figure to free memory
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Visualize BBERT embeddings with t-SNE')
    parser.add_argument('--results_dir', default='example', 
                       help='Directory containing BBERT embedding results (default: example)')
    parser.add_argument('--output_dir', default='example',
                       help='Directory to save visualization (default: example)')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum samples per category for t-SNE (default: 1000)')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter (default: 30)')
    parser.add_argument('--n_iter', type=int, default=1000,
                       help='t-SNE iterations (default: 1000)')
    
    args = parser.parse_args()
    
    # Load embeddings
    df = load_embeddings(args.results_dir)
    
    # Prepare data
    df = prepare_data(df, max_samples_per_group=args.max_samples)
    
    # Extract embeddings (assuming they're stored in 'embedding' column)
    if 'embedding' not in df.columns:
        raise ValueError("No 'embedding' column found in the data")
    
    # Convert embeddings to numpy array
    # Use full embedding space (seq_len * embedding_dim) for each sequence
    print("Converting embeddings to full embedding vectors...")
    embedding_vectors = []
    
    for emb in df['embedding'].values:
        if isinstance(emb, np.ndarray) and emb.dtype == object:
            # Convert object array to proper 2D array, then flatten to use full space
            try:
                # Stack the individual vectors to form a 2D array
                emb_matrix = np.stack(emb)  # Shape: (seq_len, embedding_dim)
                # Flatten to use full embedding space
                flattened_emb = emb_matrix.flatten()  # Shape: (seq_len * embedding_dim,)
                embedding_vectors.append(flattened_emb)
            except Exception as e:
                print(f"Error processing embedding: {e}")
                print(f"Embedding shape: {emb.shape}, dtype: {emb.dtype}")
                # Fallback: just flatten the first element
                embedding_vectors.append(emb[0].flatten())
        elif isinstance(emb, np.ndarray) and len(emb.shape) == 2:
            # Already a 2D array, flatten it
            flattened_emb = emb.flatten()
            embedding_vectors.append(flattened_emb)
        else:
            # Handle other cases
            embedding_vectors.append(np.array(emb).flatten())
    
    embeddings = np.array(embedding_vectors)
    print(f"Final embedding matrix shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    
    # Run t-SNE
    tsne_result = run_tsne(embeddings, perplexity=args.perplexity, n_iter=args.n_iter)
    
    # Create visualization
    create_visualization(df, tsne_result, args.output_dir)
    
    # Explicit cleanup and exit
    plt.close('all')
    print("Visualization complete. Exiting...")

if __name__ == "__main__":
    main()