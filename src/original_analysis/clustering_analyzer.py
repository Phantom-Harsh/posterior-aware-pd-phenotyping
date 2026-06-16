"""
Clustering Analysis Module with Uncertainty Quantification
Author: Research Team
Date: October 2025

This module implements clustering methods with Bayesian uncertainty quantification:
- Gaussian Mixture Models (GMM) with uncertainty
- K-Means clustering
- Hierarchical clustering
- Cluster validation metrics
- Uncertainty-based mixed pathology detection
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ClusteringAnalyzer:
    """
    Advanced clustering analysis with Bayesian uncertainty quantification.
    Implements mechanism-based patient subtyping.
    """
    
    def __init__(self, logger):
        """
        Initialize the clustering analyzer.
        
        Args:
            logger: Logger instance for tracking operations
        """
        self.logger = logger
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def prepare_data(self, data: pd.DataFrame, 
                     feature_cols: List[str],
                     scale: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Prepare data for clustering by handling missing values and scaling.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature columns to use
            scale: Whether to standardize features
            
        Returns:
            Tuple of (processed array, cleaned DataFrame)
        """
        self.logger.info(f"Preparing {len(feature_cols)} features for clustering...")
        
        # Select features and drop rows with missing values
        valid_features = [f for f in feature_cols if f in data.columns]
        self.logger.info(f"Using {len(valid_features)} valid features")
        
        df_clean = data[valid_features].copy()
        
        # Handle missing values
        n_before = len(df_clean)
        df_clean = df_clean.dropna()
        n_after = len(df_clean)
        self.logger.info(f"Removed {n_before - n_after} rows with missing values ({n_after} remaining)")
        
        # Convert to numpy array
        X = df_clean.values
        
        # Scale if requested
        if scale:
            X = self.scaler.fit_transform(X)
            self.logger.info("Features standardized (zero mean, unit variance)")
        
        return X, df_clean
    
    def bayesian_gmm_clustering(self, X: np.ndarray, 
                                n_components_range: Tuple[int, int] = (2, 6),
                                random_state: int = 42) -> Dict:
        """
        Perform Bayesian Gaussian Mixture Model clustering with uncertainty.
        
        Args:
            X: Input data array (samples x features)
            n_components_range: Range of cluster numbers to try
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with clustering results and uncertainty estimates
        """
        self.logger.info("Performing Bayesian GMM clustering with uncertainty quantification...")
        
        best_bic = np.inf
        best_n = 0
        bic_scores = []
        
        # Try different numbers of components
        for n_components in range(n_components_range[0], n_components_range[1] + 1):
            bgm = BayesianGaussianMixture(
                n_components=n_components,
                covariance_type='full',
                max_iter=200,
                random_state=random_state,
                weight_concentration_prior_type='dirichlet_process'
            )
            bgm.fit(X)
            
            # Calculate BIC approximation for Bayesian GMM
            # Use lower bound (negative log likelihood) as proxy
            bic = -bgm.lower_bound_
            bic_scores.append(bic)
            
            self.logger.info(f"n_components={n_components}, BIC={bic:.2f}")
            
            if bic < best_bic:
                best_bic = bic
                best_n = n_components
        
        # Fit final model with best number of components
        self.logger.info(f"Best number of components: {best_n} (BIC={best_bic:.2f})")
        
        final_bgm = BayesianGaussianMixture(
            n_components=best_n,
            covariance_type='full',
            max_iter=200,
            random_state=random_state,
            weight_concentration_prior_type='dirichlet_process'
        )
        final_bgm.fit(X)
        
        # Get cluster assignments and probabilities
        labels = final_bgm.predict(X)
        probabilities = final_bgm.predict_proba(X)
        
        # Calculate uncertainty (1 - max probability)
        uncertainties = 1 - np.max(probabilities, axis=1)
        
        # Identify high uncertainty cases (potential mixed pathology)
        high_uncertainty_threshold = 0.7
        mixed_pathology_mask = uncertainties > high_uncertainty_threshold
        
        self.logger.info(f"Identified {mixed_pathology_mask.sum()} patients with high uncertainty (>70%)")
        
        # Calculate cluster validation metrics
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
        else:
            silhouette = davies_bouldin = calinski_harabasz = np.nan
        
        results = {
            'model': final_bgm,
            'n_clusters': best_n,
            'labels': labels,
            'probabilities': probabilities,
            'uncertainties': uncertainties,
            'mixed_pathology_mask': mixed_pathology_mask,
            'bic_scores': bic_scores,
            'best_bic': best_bic,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz
        }
        
        self.models['bayesian_gmm'] = final_bgm
        self.results['bayesian_gmm'] = results
        
        self.logger.info(f"Bayesian GMM clustering completed")
        self.logger.info(f"Silhouette Score: {silhouette:.3f}")
        self.logger.info(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
        self.logger.info(f"Calinski-Harabasz Score: {calinski_harabasz:.3f}")
        
        return results
    
    def kmeans_clustering(self, X: np.ndarray,
                         n_clusters_range: Tuple[int, int] = (2, 6),
                         random_state: int = 42) -> Dict:
        """
        Perform K-Means clustering with elbow method.
        
        Args:
            X: Input data array
            n_clusters_range: Range of cluster numbers to try
            random_state: Random seed
            
        Returns:
            Dictionary with clustering results
        """
        self.logger.info("Performing K-Means clustering...")
        
        inertias = []
        silhouette_scores = []
        
        # Try different numbers of clusters
        for n_clusters in range(n_clusters_range[0], n_clusters_range[1] + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            
            if n_clusters > 1:
                sil_score = silhouette_score(X, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
            
            self.logger.info(f"n_clusters={n_clusters}, inertia={kmeans.inertia_:.2f}, silhouette={silhouette_scores[-1]:.3f}")
        
        # Select best number of clusters (highest silhouette)
        best_idx = np.argmax(silhouette_scores)
        best_n = n_clusters_range[0] + best_idx
        
        # Fit final model
        final_kmeans = KMeans(n_clusters=best_n, random_state=random_state, n_init=10)
        labels = final_kmeans.fit_predict(X)
        
        results = {
            'model': final_kmeans,
            'n_clusters': best_n,
            'labels': labels,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'cluster_centers': final_kmeans.cluster_centers_
        }
        
        self.models['kmeans'] = final_kmeans
        self.results['kmeans'] = results
        
        self.logger.info(f"K-Means clustering completed with {best_n} clusters")
        
        return results
    
    def hierarchical_clustering(self, X: np.ndarray,
                                n_clusters: int = 3,
                                linkage: str = 'ward') -> Dict:
        """
        Perform hierarchical clustering.
        
        Args:
            X: Input data array
            n_clusters: Number of clusters
            linkage: Linkage method ('ward', 'complete', 'average')
            
        Returns:
            Dictionary with clustering results
        """
        self.logger.info(f"Performing hierarchical clustering with {linkage} linkage...")
        
        hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = hc.fit_predict(X)
        
        # Calculate validation metrics
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
        else:
            silhouette = davies_bouldin = np.nan
        
        results = {
            'model': hc,
            'n_clusters': n_clusters,
            'labels': labels,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin
        }
        
        self.models['hierarchical'] = hc
        self.results['hierarchical'] = results
        
        self.logger.info(f"Hierarchical clustering completed")
        
        return results
    
    def characterize_clusters(self, X: np.ndarray,
                             labels: np.ndarray,
                             feature_names: List[str]) -> pd.DataFrame:
        """
        Characterize clusters by their feature profiles.
        
        Args:
            X: Input data array
            labels: Cluster labels
            feature_names: Names of features
            
        Returns:
            DataFrame with cluster characterization
        """
        self.logger.info("Characterizing cluster profiles...")
        
        characterization = []
        
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_data = X[mask]
            
            char = {
                'cluster_id': cluster_id,
                'n_samples': mask.sum(),
                'percentage': 100 * mask.sum() / len(labels)
            }
            
            # Calculate mean and std for each feature
            for i, feature in enumerate(feature_names):
                char[f'{feature}_mean'] = np.mean(cluster_data[:, i])
                char[f'{feature}_std'] = np.std(cluster_data[:, i])
            
            characterization.append(char)
        
        char_df = pd.DataFrame(characterization)
        self.logger.info(f"Characterized {len(char_df)} clusters")
        
        return char_df
    
    def identify_discriminative_features(self, X: np.ndarray,
                                        labels: np.ndarray,
                                        feature_names: List[str],
                                        top_n: int = 10) -> pd.DataFrame:
        """
        Identify features that best discriminate between clusters.
        
        Args:
            X: Input data array
            labels: Cluster labels
            feature_names: Names of features
            top_n: Number of top features to return
            
        Returns:
            DataFrame with discriminative features ranked by F-statistic
        """
        self.logger.info("Identifying discriminative features...")
        
        from scipy.stats import f_oneway
        
        discriminative_features = []
        
        for i, feature in enumerate(feature_names):
            groups = [X[labels == label, i] for label in np.unique(labels)]
            f_stat, p_value = f_oneway(*groups)
            
            discriminative_features.append({
                'feature': feature,
                'f_statistic': f_stat,
                'p_value': p_value
            })
        
        disc_df = pd.DataFrame(discriminative_features)
        disc_df = disc_df.sort_values('f_statistic', ascending=False)
        disc_df = disc_df.head(top_n)
        
        self.logger.info(f"Top {top_n} discriminative features identified")
        
        return disc_df
    
    def pca_reduction(self, X: np.ndarray, 
                     n_components: int = 2) -> Tuple[np.ndarray, PCA]:
        """
        Reduce dimensionality using PCA for visualization.
        
        Args:
            X: Input data array
            n_components: Number of principal components
            
        Returns:
            Tuple of (transformed data, PCA model)
        """
        self.logger.info(f"Performing PCA reduction to {n_components} components...")
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        self.logger.info(f"Explained variance: {explained_var}")
        self.logger.info(f"Cumulative variance: {cumulative_var[-1]:.3f}")
        
        return X_pca, pca


if __name__ == "__main__":
    # Test the clustering analyzer
    from logger_config import setup_logging
    
    logger_system = setup_logging()
    logger = logger_system.get_logger("clustering_test", "clustering")
    
    analyzer = ClusteringAnalyzer(logger)
    
    # Generate test data
    np.random.seed(42)
    X1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
    X2 = np.random.multivariate_normal([3, 3], [[1, 0], [0, 1]], 100)
    X3 = np.random.multivariate_normal([0, 3], [[1, 0], [0, 1]], 100)
    X = np.vstack([X1, X2, X3])
    
    # Test Bayesian GMM
    results = analyzer.bayesian_gmm_clustering(X, n_components_range=(2, 5))
    print(f"\nBayesian GMM Results:")
    print(f"Number of clusters: {results['n_clusters']}")
    print(f"Silhouette Score: {results['silhouette_score']:.3f}")
    print(f"Mixed pathology cases: {results['mixed_pathology_mask'].sum()}")

