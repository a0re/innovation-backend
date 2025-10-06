"""
Clustering module for spam detection project.
Performs K-Means clustering on spam messages to identify spam subtypes.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Any
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from utils.helpers import load_config, ensure_dir_exists, setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class SpamClusterer:
    """
    K-Means clustering class for spam message analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the spam clusterer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.vectorizer = None
        self.cluster_results = {}
        self.best_k = None
        self.best_silhouette_score = 0
        
    def prepare_spam_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter and prepare spam messages for clustering.
        
        Args:
            df: DataFrame with text and label columns
            
        Returns:
            DataFrame with spam messages only
        """
        logger.info("Preparing spam data for clustering...")
        
        # Filter spam messages
        spam_df = df[df['label'] == 'spam'].copy()
        
        logger.info(f"Found {len(spam_df)} spam messages for clustering")
        
        return spam_df
    
    def create_vectorizer(self) -> TfidfVectorizer:
        """
        Create TF-IDF vectorizer for clustering.
        
        Returns:
            TF-IDF vectorizer
        """
        logger.info("Creating TF-IDF vectorizer for clustering...")
        
        vectorizer = TfidfVectorizer(
            ngram_range=tuple(self.config['preprocessing']['ngram_range_word']),
            min_df=self.config['preprocessing']['min_df'],
            max_features=self.config['preprocessing']['max_features'],
            norm='l2',
            lowercase=True,
            stop_words='english'
        )
        
        return vectorizer
    
    def perform_clustering(self, spam_df: pd.DataFrame, k_values: List[int]) -> Dict[int, Dict]:
        """
        Perform K-Means clustering for different k values.
        
        Args:
            spam_df: DataFrame with spam messages
            k_values: List of k values to try
            
        Returns:
            Dictionary with clustering results for each k
        """
        logger.info("Starting K-Means clustering...")
        
        # Create vectorizer
        self.vectorizer = self.create_vectorizer()
        
        # Vectorize text
        X = self.vectorizer.fit_transform(spam_df['text'])
        feature_names = self.vectorizer.get_feature_names_out()
        
        logger.info(f"Vectorized {X.shape[0]} messages into {X.shape[1]} features")
        
        results = {}
        
        for k in k_values:
            logger.info(f"Clustering with k={k}...")
            
            # Perform K-Means clustering
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.config['random_state'],
                max_iter=self.config['clustering']['max_iter'],
                n_init=10
            )
            
            cluster_labels = kmeans.fit_predict(X)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(X, cluster_labels)
            
            # Calculate inertia
            inertia = kmeans.inertia_
            
            # Get top terms for each cluster
            top_terms = self.get_top_terms_per_cluster(
                X, cluster_labels, feature_names, k, top_n=15
            )
            
            # Store results
            results[k] = {
                'kmeans': kmeans,
                'cluster_labels': cluster_labels,
                'silhouette_score': silhouette_avg,
                'inertia': inertia,
                'top_terms': top_terms
            }
            
            logger.info(f"k={k} - Silhouette Score: {silhouette_avg:.4f}, Inertia: {inertia:.2f}")
            
            # Update best k
            if silhouette_avg > self.best_silhouette_score:
                self.best_silhouette_score = silhouette_avg
                self.best_k = k
        
        self.cluster_results = results
        return results
    
    def get_top_terms_per_cluster(self, X: np.ndarray, cluster_labels: np.ndarray, 
                                 feature_names: np.ndarray, k: int, top_n: int = 15) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top TF-IDF terms for each cluster.
        
        Args:
            X: TF-IDF matrix
            cluster_labels: Cluster assignments
            feature_names: Feature names from vectorizer
            k: Number of clusters
            top_n: Number of top terms to return per cluster
            
        Returns:
            Dictionary mapping cluster_id to list of (term, score) tuples
        """
        top_terms = {}
        
        for cluster_id in range(k):
            # Get documents in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_docs = X[cluster_mask]
            
            if cluster_docs.shape[0] == 0:
                top_terms[cluster_id] = []
                continue
            
            # Calculate mean TF-IDF scores for this cluster
            mean_scores = np.asarray(cluster_docs.mean(axis=0)).flatten()
            
            # Get top terms
            top_indices = np.argsort(mean_scores)[-top_n:][::-1]
            cluster_terms = [(feature_names[i], mean_scores[i]) for i in top_indices]
            
            top_terms[cluster_id] = cluster_terms
        
        return top_terms
    
    def plot_silhouette_analysis(self, save_path: str = None) -> None:
        """
        Plot silhouette scores for different k values.
        
        Args:
            save_path: Path to save the plot
        """
        logger.info("Creating silhouette analysis plot...")
        
        k_values = list(self.cluster_results.keys())
        silhouette_scores = [self.cluster_results[k]['silhouette_score'] for k in k_values]
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, silhouette_scores, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Silhouette Analysis for K-Means Clustering', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Highlight best k
        best_k_idx = k_values.index(self.best_k)
        plt.plot(self.best_k, self.best_silhouette_score, 'ro', markersize=12, 
                label=f'Best k={self.best_k}')
        plt.legend()
        
        # Add value labels
        for k, score in zip(k_values, silhouette_scores):
            plt.annotate(f'{score:.3f}', (k, score), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir_exists(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Silhouette analysis plot saved to {save_path}")
        
        plt.show()
    
    def plot_inertia_analysis(self, save_path: str = None) -> None:
        """
        Plot inertia (within-cluster sum of squares) for different k values.
        
        Args:
            save_path: Path to save the plot
        """
        logger.info("Creating inertia analysis plot...")
        
        k_values = list(self.cluster_results.keys())
        inertias = [self.cluster_results[k]['inertia'] for k in k_values]
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, inertias, 'go-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
        plt.title('Inertia Analysis for K-Means Clustering', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for k, inertia in zip(k_values, inertias):
            plt.annotate(f'{inertia:.0f}', (k, inertia), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir_exists(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Inertia analysis plot saved to {save_path}")
        
        plt.show()
    
    def plot_cluster_visualization(self, spam_df: pd.DataFrame, k: int, save_path: str = None) -> None:
        """
        Plot 2D visualization of clusters using PCA.
        
        Args:
            spam_df: DataFrame with spam messages
            k: Number of clusters to visualize
            save_path: Path to save the plot
        """
        logger.info(f"Creating cluster visualization for k={k}...")
        
        if k not in self.cluster_results:
            logger.error(f"No clustering results found for k={k}")
            return
        
        # Get cluster labels
        cluster_labels = self.cluster_results[k]['cluster_labels']
        
        # Vectorize text
        X = self.vectorizer.transform(spam_df['text'])
        
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2, random_state=self.config['random_state'])
        X_pca = pca.fit_transform(X.toarray())
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot points colored by cluster
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                            cmap='tab10', alpha=0.7, s=50)
        
        plt.xlabel(f'First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})', 
                  fontsize=12)
        plt.ylabel(f'Second Principal Component (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})', 
                  fontsize=12)
        plt.title(f'K-Means Clustering Visualization (k={k})', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir_exists(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster visualization saved to {save_path}")
        
        plt.show()
    
    def print_cluster_analysis(self) -> None:
        """
        Print detailed cluster analysis results.
        """
        print("\n" + "="*60)
        print("K-MEANS CLUSTERING ANALYSIS")
        print("="*60)
        
        print(f"\nBest k: {self.best_k}")
        print(f"Best Silhouette Score: {self.best_silhouette_score:.4f}")
        
        print(f"\nSilhouette Scores by k:")
        for k in sorted(self.cluster_results.keys()):
            score = self.cluster_results[k]['silhouette_score']
            print(f"  k={k}: {score:.4f}")
        
        print(f"\nTop Terms per Cluster (k={self.best_k}):")
        best_result = self.cluster_results[self.best_k]
        for cluster_id, terms in best_result['top_terms'].items():
            print(f"\n  Cluster {cluster_id}:")
            for term, score in terms[:10]:  # Show top 10 terms
                print(f"    {term}: {score:.4f}")
    
    def save_cluster_results(self, output_dir: str) -> None:
        """
        Save clustering results to files.
        
        Args:
            output_dir: Directory to save results
        """
        logger.info("Saving clustering results...")
        
        ensure_dir_exists(output_dir)
        
        # Save silhouette scores
        silhouette_data = []
        for k, result in self.cluster_results.items():
            silhouette_data.append({
                'k': k,
                'silhouette_score': result['silhouette_score'],
                'inertia': result['inertia']
            })
        
        silhouette_df = pd.DataFrame(silhouette_data)
        silhouette_path = os.path.join(output_dir, 'clustering_silhouette_scores.csv')
        silhouette_df.to_csv(silhouette_path, index=False)
        logger.info(f"Silhouette scores saved to {silhouette_path}")
        
        # Save top terms for best k
        best_result = self.cluster_results[self.best_k]
        top_terms_data = []
        for cluster_id, terms in best_result['top_terms'].items():
            for rank, (term, score) in enumerate(terms, 1):
                top_terms_data.append({
                    'cluster_id': cluster_id,
                    'rank': rank,
                    'term': term,
                    'tfidf_score': score
                })
        
        top_terms_df = pd.DataFrame(top_terms_data)
        top_terms_path = os.path.join(output_dir, 'clustering_top_terms.csv')
        top_terms_df.to_csv(top_terms_path, index=False)
        logger.info(f"Top terms saved to {top_terms_path}")

def run_clustering(df: pd.DataFrame) -> SpamClusterer:
    """
    Main function to run K-Means clustering on spam messages.
    
    Args:
        df: DataFrame with text and label columns
        
    Returns:
        Trained SpamClusterer object
    """
    logger.info("Starting spam clustering analysis...")
    
    # Load configuration
    config = load_config()
    
    # Initialize clusterer
    clusterer = SpamClusterer(config)
    
    # Prepare spam data
    spam_df = clusterer.prepare_spam_data(df)
    
    if len(spam_df) < 10:
        logger.warning("Not enough spam messages for clustering. Need at least 10 messages.")
        return clusterer
    
    # Perform clustering
    k_values = config['clustering']['k_values']
    results = clusterer.perform_clustering(spam_df, k_values)
    
    # Create visualizations
    output_dir = os.path.join(config['data']['output_dir'], 'plots')
    
    clusterer.plot_silhouette_analysis(
        save_path=os.path.join(output_dir, 'clustering_silhouette_analysis.png')
    )
    
    clusterer.plot_inertia_analysis(
        save_path=os.path.join(output_dir, 'clustering_inertia_analysis.png')
    )
    
    clusterer.plot_cluster_visualization(
        spam_df, clusterer.best_k,
        save_path=os.path.join(output_dir, f'clustering_visualization_k{clusterer.best_k}.png')
    )
    
    # Print analysis
    clusterer.print_cluster_analysis()
    
    # Save results
    clusterer.save_cluster_results(output_dir)
    
    logger.info("Clustering analysis completed!")
    
    return clusterer

if __name__ == "__main__":
    # Load data and run clustering
    from data.preprocess import preprocess_data
    
    # Get preprocessed data
    train_df, val_df, test_df = preprocess_data()
    
    # Combine all data for clustering
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Run clustering
    clusterer = run_clustering(all_data)
    
    print("Clustering analysis completed!")
