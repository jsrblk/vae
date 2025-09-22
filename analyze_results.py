#!/usr/bin/env python3
"""
Analysis script to compare generated VAE samples with actual data.
Plots marginals, correlations, and other statistical properties.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def load_data():
    """Load actual and generated data"""
    # Load actual returns
    actual_data = pd.read_parquet('artifacts/data/returns.parquet')
    
    # Load generated samples
    generated_data = pd.read_csv('artifacts/samples/returns.csv', header=None)
    
    print(f"Actual data shape: {actual_data.shape}")
    print(f"Generated data shape: {generated_data.shape}")
    
    return actual_data, generated_data

def plot_marginals(actual_data, generated_data, n_stocks=10, save_dir='artifacts/analysis'):
    """Plot marginal distributions for selected stocks"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Select first n_stocks for comparison
    actual_subset = actual_data.iloc[:, :n_stocks]
    generated_subset = generated_data.iloc[:, :n_stocks]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, stock in enumerate(actual_subset.columns):
        if i >= 10:
            break
            
        ax = axes[i]
        
        # Plot histograms
        ax.hist(actual_subset[stock].dropna(), bins=50, alpha=0.7, label='Actual', density=True, color='blue')
        ax.hist(generated_subset.iloc[:, i], bins=50, alpha=0.7, label='Generated', density=True, color='red')
        
        ax.set_title(f'{stock}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/marginals_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_matrices(actual_data, generated_data, save_dir='artifacts/analysis'):
    """Plot correlation matrices for actual vs generated data"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Use first 50 stocks for correlation analysis
    n_stocks = min(50, actual_data.shape[1], generated_data.shape[1])
    actual_subset = actual_data.iloc[:, :n_stocks]
    generated_subset = generated_data.iloc[:, :n_stocks]
    
    # Calculate correlation matrices
    actual_corr = actual_subset.corr()
    generated_corr = generated_subset.corr()
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot actual correlation
    sns.heatmap(actual_corr, ax=ax1, cmap='coolwarm', center=0, 
                square=True, cbar_kws={'shrink': 0.8})
    ax1.set_title('Actual Data Correlations')
    
    # Plot generated correlation
    sns.heatmap(generated_corr, ax=ax2, cmap='coolwarm', center=0, 
                square=True, cbar_kws={'shrink': 0.8})
    ax2.set_title('Generated Data Correlations')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/correlation_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return actual_corr, generated_corr

def plot_correlation_difference(actual_corr, generated_corr, save_dir='artifacts/analysis'):
    """Plot difference between correlation matrices"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate difference
    corr_diff = actual_corr - generated_corr
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_diff, cmap='RdBu_r', center=0, square=True, 
                cbar_kws={'shrink': 0.8})
    plt.title('Correlation Difference (Actual - Generated)')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/correlation_difference.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_statistical_comparison(actual_data, generated_data, save_dir='artifacts/analysis'):
    """Plot statistical comparison between actual and generated data"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate statistics
    actual_stats = {
        'Mean': actual_data.mean().mean(),
        'Std': actual_data.std().mean(),
        'Skewness': actual_data.skew().mean(),
        'Kurtosis': actual_data.kurtosis().mean()
    }
    
    generated_stats = {
        'Mean': generated_data.mean().mean(),
        'Std': generated_data.std().mean(),
        'Skewness': generated_data.skew().mean(),
        'Kurtosis': generated_data.kurtosis().mean()
    }
    
    # Create comparison plot
    stats_names = list(actual_stats.keys())
    actual_values = list(actual_stats.values())
    generated_values = list(generated_stats.values())
    
    x = np.arange(len(stats_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, actual_values, width, label='Actual', alpha=0.8)
    bars2 = ax.bar(x + width/2, generated_values, width, label='Generated', alpha=0.8)
    
    ax.set_xlabel('Statistics')
    ax.set_ylabel('Values')
    ax.set_title('Statistical Comparison: Actual vs Generated')
    ax.set_xticks(x)
    ax.set_xticklabels(stats_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/statistical_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return actual_stats, generated_stats

def plot_qq_plots(actual_data, generated_data, n_stocks=5, save_dir='artifacts/analysis'):
    """Plot Q-Q plots for selected stocks"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i in range(min(n_stocks, 6)):
        if i >= actual_data.shape[1]:
            break
            
        ax = axes[i]
        
        # Get data for this stock
        actual_stock = actual_data.iloc[:, i].dropna()
        generated_stock = generated_data.iloc[:, i]
        
        # Create Q-Q plot
        stats.probplot(actual_stock, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot: {actual_data.columns[i]} (Actual)')
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    if n_stocks < 6:
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/qq_plots_actual.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Q-Q plots for generated data
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i in range(min(n_stocks, 6)):
        if i >= generated_data.shape[1]:
            break
            
        ax = axes[i]
        
        # Get data for this stock
        generated_stock = generated_data.iloc[:, i]
        
        # Create Q-Q plot
        stats.probplot(generated_stock, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot: Stock {i} (Generated)')
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    if n_stocks < 6:
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/qq_plots_generated.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_correlation_metrics(actual_corr, generated_corr):
    """Calculate correlation-based metrics"""
    # Flatten correlation matrices (excluding diagonal)
    actual_flat = actual_corr.values[np.triu_indices_from(actual_corr.values, k=1)]
    generated_flat = generated_corr.values[np.triu_indices_from(generated_corr.values, k=1)]
    
    # Calculate correlation between correlation matrices
    corr_correlation = np.corrcoef(actual_flat, generated_flat)[0, 1]
    
    # Calculate mean absolute error
    mae = np.mean(np.abs(actual_flat - generated_flat))
    
    # Calculate root mean square error
    rmse = np.sqrt(np.mean((actual_flat - generated_flat)**2))
    
    print(f"\nCorrelation Analysis:")
    print(f"Correlation between correlation matrices: {corr_correlation:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Square Error: {rmse:.4f}")
    
    return {
        'correlation': corr_correlation,
        'mae': mae,
        'rmse': rmse
    }

def main():
    """Main analysis function"""
    print("Loading data...")
    actual_data, generated_data = load_data()
    
    print("\nPlotting marginal distributions...")
    plot_marginals(actual_data, generated_data)
    
    print("\nPlotting correlation matrices...")
    actual_corr, generated_corr = plot_correlation_matrices(actual_data, generated_data)
    
    print("\nPlotting correlation difference...")
    plot_correlation_difference(actual_corr, generated_corr)
    
    print("\nPlotting statistical comparison...")
    actual_stats, generated_stats = plot_statistical_comparison(actual_data, generated_data)
    
    print("\nPlotting Q-Q plots...")
    plot_qq_plots(actual_data, generated_data)
    
    print("\nCalculating correlation metrics...")
    corr_metrics = calculate_correlation_metrics(actual_corr, generated_corr)
    
    print("\nAnalysis complete! Check artifacts/analysis/ for all plots.")
    
    return {
        'actual_stats': actual_stats,
        'generated_stats': generated_stats,
        'correlation_metrics': corr_metrics
    }

if __name__ == "__main__":
    results = main()

