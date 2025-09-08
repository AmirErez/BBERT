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
from sklearn.decomposition import PCA
from pathlib import Path
import argparse


def load_embeddings_flexible(files_list, labels_list):
    """
    Load embeddings from specified files with explicit labels.
    
    Args:
        files_list: List of file paths (required)
        labels_list: List of labels for each file (required)
    
    Returns:
        Combined DataFrame with 'sample' column for grouping
    """
    # Validate required arguments
    if not files_list:
        raise ValueError("files_list is required and cannot be None or empty")
    if not labels_list:
        raise ValueError("labels_list is required and cannot be None or empty")
    
    # Use explicitly provided files
    parquet_files = [Path(f) for f in files_list]
    
    # Validate files exist
    for file in parquet_files:
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file}")
    
    # Validate labels match files
    if len(labels_list) != len(parquet_files):
        raise ValueError(f"Number of labels ({len(labels_list)}) must match number of files ({len(parquet_files)})")
    
    sample_labels = labels_list
    
    print(f"Loading {len(parquet_files)} files...")
    all_data = []
    
    for file, sample_label in zip(parquet_files, sample_labels):
        print(f"  Loading {file.name} as '{sample_label}'...")
        df = pd.read_parquet(file)
        
        df['sample'] = sample_label
        all_data.append(df)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total sequences loaded: {len(combined_df)}")
    
    # Print sample distribution
    print("\nSample distribution:")
    for sample, count in combined_df['sample'].value_counts().items():
        print(f"  {sample}: {count:,} sequences")
    
    return combined_df

def prepare_data(df, max_samples_per_group=1000):
    """Prepare data for t-SNE analysis"""
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
    
    # Apply the functions - now organism is just the sample label
    df = df.copy()  # Avoid SettingWithCopyWarning
    df['organism'] = df['sample']  # Use sample label directly as "organism"
    df['coding_status'] = df['coding_prob'].apply(get_coding_from_predictions)
    df['frame'] = df['frame_prob'].apply(get_frame_from_predictions)
    df['detailed_category'] = df.apply(lambda row: get_detailed_category(row['coding_prob'], row['frame_prob']), axis=1)
    
    print(f"Data after processing: {len(df)} sequences across {df['organism'].nunique()} samples")
    
    # Create combined category for coloring (sample + coding/frame information)
    df['category'] = df['organism'] + '_' + df['detailed_category']
    
    # Sample data to make t-SNE computation feasible
    # Dynamically determine number of categories
    num_samples = df['organism'].nunique()
    num_categories = df['category'].nunique()
    max_total_size = max_samples_per_group * num_categories
    
    if len(df) > max_total_size:  
        print(f"Sampling data to max {max_samples_per_group} per category...")
        print(f"Found {num_categories} categories across {num_samples} samples")
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

def run_pca(embeddings, n_components=2):
    """Run PCA on embeddings"""
    print("Running PCA...")
    print(f"Embedding shape: {embeddings.shape}")
    
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(embeddings)
    
    print(f"PCA completed. Final embedding shape: {pca_result.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    return pca_result

def create_visualization(df, result_2d, output_dir, output_name='bbert_visualization', method='t-SNE'):
    """Create and save dimensionality reduction visualization"""
    # Add 2D coordinates to dataframe
    df['dim_1'] = result_2d[:, 0]
    df['dim_2'] = result_2d[:, 1]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{method} Visualization of BBERT Embeddings', fontsize=16, fontweight='bold')
    
    # Dynamic color palettes
    unique_organisms = df['organism'].unique()
    unique_coding = df['coding_status'].unique()
    
    # Generate colors dynamically for organisms/samples
    organism_colors = {}
    colors_list = plt.cm.tab10(np.linspace(0, 1, len(unique_organisms)))
    for i, organism in enumerate(unique_organisms):
        organism_colors[organism] = colors_list[i]
    
    coding_colors = {'Coding': '#2ca02c', 'Non-coding': '#d62728', 'Unknown': 'gray'}
    
    # Plot 1: Color by sample/organism
    ax1 = axes[0, 0]
    for organism in unique_organisms:
        mask = df['organism'] == organism
        ax1.scatter(df.loc[mask, 'dim_1'], df.loc[mask, 'dim_2'], 
                   c=[organism_colors[organism]], label=organism, alpha=0.6, s=20)
    ax1.set_title('Colored by Sample/Species')
    ax1.set_xlabel(f'{method} 1')
    ax1.set_ylabel(f'{method} 2')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 2: Color by coding status
    ax2 = axes[0, 1]
    for coding in df['coding_status'].unique():
        mask = df['coding_status'] == coding
        ax2.scatter(df.loc[mask, 'dim_1'], df.loc[mask, 'dim_2'], 
                   c=coding_colors.get(coding, 'gray'), label=coding, alpha=0.6, s=20)
    ax2.set_title('Colored by Coding Status')
    ax2.set_xlabel(f'{method} 1')
    ax2.set_ylabel(f'{method} 2')
    ax2.legend()
    
    # Plot 3: Color by detailed category (organism + frame/non-coding)
    ax3 = axes[1, 0]
    categories = df['category'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))  # More colors for more categories
    for i, category in enumerate(categories):
        mask = df['category'] == category
        ax3.scatter(df.loc[mask, 'dim_1'], df.loc[mask, 'dim_2'], 
                   c=[colors[i]], label=category, alpha=0.6, s=20)
    ax3.set_title('Colored by Sample + Frame/Non-coding')
    ax3.set_xlabel(f'{method} 1')
    ax3.set_ylabel(f'{method} 2')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 4: Sample distribution
    ax4 = axes[1, 1]
    for sample in df['sample'].unique():
        mask = df['sample'] == sample
        ax4.scatter(df.loc[mask, 'dim_1'], df.loc[mask, 'dim_2'], 
                   label=sample, alpha=0.6, s=20)
    ax4.set_title('Colored by Sample')
    ax4.set_xlabel(f'{method} 1')
    ax4.set_ylabel(f'{method} 2')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save plot in both PNG and PDF formats
    output_path_png = Path(output_dir) / f'{output_name}.png'
    output_path_pdf = Path(output_dir) / f'{output_name}.pdf'
    
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    
    print(f"Visualization saved to:")
    print(f"  PNG: {output_path_png}")
    print(f"  PDF: {output_path_pdf}")
    
    # Skip saving CSV - too large for practical use
    # data_output_path = Path(output_dir) / 'tsne_results.csv'
    # df.to_csv(data_output_path, index=False)
    # print(f"Data with t-SNE coordinates saved to: {data_output_path}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total sequences: {len(df)}")
    print(f"\nBy sample/species:")
    for sample, count in df['organism'].value_counts().items():
        print(f"  {sample}: {count:,} sequences")
    print(f"\nBy coding status:")
    for status, count in df['coding_status'].value_counts().items():
        print(f"  {status}: {count:,} sequences")
    print(f"\nBy detailed category:")
    for category, count in df['category'].value_counts().head(10).items():  # Show top 10
        print(f"  {category}: {count:,} sequences")
    if df['category'].nunique() > 10:
        print(f"  ... and {df['category'].nunique() - 10} more categories")
    
    # Close the figure to free memory
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description='Visualize BBERT embeddings with t-SNE or PCA',
        epilog="""
EXAMPLES:
  # Basic usage with required parameters
  python example/visualize_embeddings.py \\
    --files example/Pseudomonas_aeruginosa_R1_scores_len_emb.parquet,example/Saccharomyces_paradoxus_R1_scores_len_emb.parquet \\
    --labels "P. aeruginosa,S. paradoxus" \\
    --output_dir example \\
    --output_name bacterial_vs_eukaryotic
  
  # Use PCA method (faster than t-SNE)
  python example/visualize_embeddings.py \\
    --files example/Pseudomonas_aeruginosa_R1_scores_len_emb.parquet,example/Saccharomyces_paradoxus_R1_scores_len_emb.parquet \\
    --labels "P. aeruginosa,S. paradoxus" \\
    --output_dir example \\
    --output_name bacterial_vs_eukaryotic_pca \\
    --method pca
      --output_name ecoli_paired_end
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
  
    parser.add_argument('--files', type=str, required=True,
                       help='Comma-separated list of embedding parquet files.')
    parser.add_argument('--labels', type=str, required=True,
                       help='Comma-separated list of labels for each file. Must match number of files.')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save visualization.')
    parser.add_argument('--output_name', type=str, required=True,
                       help='Output filename (without extension).')
    parser.add_argument('--method', choices=['tsne', 'pca'], default='tsne',
                       help='Dimensionality reduction method: tsne or pca (default: tsne)')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum samples per category for dimensionality reduction (default: 1000)')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter (default: 30)')
    parser.add_argument('--n_iter', type=int, default=1000,
                       help='t-SNE iterations (default: 1000)')
    
    args = parser.parse_args()
    
    # Parse file and label lists (both are required)
    files_list = [f.strip() for f in args.files.split(',')]
    labels_list = [l.strip() for l in args.labels.split(',')]
    
    print(f"Using specified files: {files_list}")
    print(f"Using specified labels: {labels_list}")
    
    # Load embeddings using explicit file and label lists
    df = load_embeddings_flexible(
        files_list=files_list,
        labels_list=labels_list
    )
    
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
    
    # Run dimensionality reduction based on method choice
    if args.method == 'pca':
        # Run PCA
        pca_result = run_pca(embeddings)
        create_visualization(df, pca_result, args.output_dir, args.output_name, method='PCA')
    else:  # args.method == 'tsne' (default)
        # Run t-SNE
        tsne_result = run_tsne(embeddings, perplexity=args.perplexity, n_iter=args.n_iter)
        create_visualization(df, tsne_result, args.output_dir, args.output_name, method='t-SNE')
    
    # Explicit cleanup and exit
    plt.close('all')
    print("Visualization complete. Exiting...")

if __name__ == "__main__":
    main()
