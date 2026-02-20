# Authors: T. Moudiki

import numpy as np
import pandas as pd
import warnings
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.special import softmax
from .mts import MTS


class DiscreteTokenMTS(MTS):
    """
    MTS for discrete token forecasting via nearest-neighbor in embedding space.
    
    Maps continuous predictions to discrete tokens using nearest-neighbor lookup
    in a vocabulary (embedding space). Supports probabilistic decoding with
    temperature-controlled softmax and uncertainty quantification in token space.
    
    Parameters
    ----------
    obj : object
        Base learner with fit() and predict() methods
    
    vocab : np.ndarray of shape (vocab_size, n_series)
        Token vocabulary - each row is a token embedding vector
    
    metric : {'euclidean', 'cosine'}, default='euclidean'
        Distance metric for nearest-neighbor lookup
    
    return_mode : {'token_id', 'token_vector', 'both', 'probs'}, default='token_id'
        Output format:
        - 'token_id': integer token indices
        - 'token_vector': token embedding vectors
        - 'both': single DataFrame with token_id + dimensions
        - 'probs': probability distribution over all tokens
    
    softmax_temperature : float, default=1.0
        Temperature for softmax when return_mode='probs'
        Lower values (0.1-0.5) → sharper distributions (more deterministic)
        Higher values (2.0-10.0) → smoother distributions (more exploratory)
    
    normalize_vocab : bool, default=False
        Whether to center and scale vocabulary to zero mean, unit variance
    
    **mts_kwargs : dict
        Additional parameters passed to MTS base class
    
    Attributes
    ----------
    vocab : np.ndarray
        Normalized vocabulary (if normalize_vocab=True)
    
    vocab_mean_ : np.ndarray
        Mean used for normalization (if normalize_vocab=True)
    
    vocab_std_ : np.ndarray
        Std used for normalization (if normalize_vocab=True)
    
    discretization_errors_ : pd.DataFrame or None
        Distances from predictions to nearest tokens
    
    Warnings
    --------
    - Prediction intervals (lower/upper) are NOT discretized - only the mean
    - For uncertainty in token space, use predict_token_distribution()
    - Vocabulary quality strongly affects results - use diagnose_vocabulary()
    
    Examples
    --------
    >>> # Basic token prediction
    >>> vocab = np.random.randn(100, 10)  # 100 tokens, 10 dimensions
    >>> model = DiscreteTokenMTS(
    ...     obj=Ridge(),
    ...     vocab=vocab,
    ...     lags=5,
    ...     return_mode='token_id'
    ... )
    >>> model.fit(X_train)
    >>> tokens = model.predict(h=10)
    
    >>> # Probabilistic with temperature control
    >>> model = DiscreteTokenMTS(
    ...     obj=Ridge(),
    ...     vocab=vocab,
    ...     lags=5,
    ...     return_mode='probs',
    ...     softmax_temperature=1.5
    ... )
    >>> probs = model.predict(h=10)  # Returns probability distributions
    
    >>> # Uncertainty-aware token distributions
    >>> freqs, entropy, mode = model.predict_token_distribution(
    ...     h=10,
    ...     replications=100
    ... )
    """
    
    def __init__(
        self,
        obj,
        vocab,
        metric='euclidean',
        return_mode='token_id',
        softmax_temperature=1.0,
        normalize_vocab=False,
        **mts_kwargs
    ):
        super().__init__(obj, **mts_kwargs)
        
        # Convert and validate vocabulary
        self.vocab_original = np.asarray(vocab, dtype=np.float64)
        self._validate_vocabulary()
        
        self.vocab_size = self.vocab_original.shape[0]
        self.vocab_mean_ = None
        self.vocab_std_ = None
        self.normalize_vocab = normalize_vocab
        
        # Normalize if requested
        if normalize_vocab:
            self._normalize_vocabulary()
        else:
            self.vocab = self.vocab_original.copy()
        
        # Validate and set metric
        assert metric in ['euclidean', 'cosine'], \
            "metric must be 'euclidean' or 'cosine'"
        self.metric = metric
        self.distance_func = (
            euclidean_distances if metric == 'euclidean' else cosine_distances
        )
        
        # Validate and set return mode
        assert return_mode in ['token_id', 'token_vector', 'both', 'probs'], \
            "return_mode must be 'token_id', 'token_vector', 'both', or 'probs'"
        self.return_mode = return_mode
        
        # Validate temperature
        assert softmax_temperature > 0, "softmax_temperature must be positive"
        self.softmax_temperature = softmax_temperature
        
        # Initialize error tracking
        self.discretization_errors_ = None
    
    def _validate_vocabulary(self):
        """Comprehensive vocabulary validation"""
        # Check shape
        assert self.vocab_original.ndim == 2, "vocab must be 2D array (vocab_size, n_series)"
        assert self.vocab_original.shape[0] > 0, "vocab must have at least one token"
        
        # Check for NaN/Inf
        if np.any(np.isnan(self.vocab_original)) or np.any(np.isinf(self.vocab_original)):
            raise ValueError("Vocabulary contains NaN or Inf values")
        
        # Check for duplicates
        unique_rows = np.unique(self.vocab_original, axis=0)
        if len(unique_rows) < len(self.vocab_original):
            n_duplicates = len(self.vocab_original) - len(unique_rows)
            warnings.warn(
                f"Vocabulary contains {n_duplicates} duplicate vectors. "
                "This reduces effective vocabulary size.",
                UserWarning
            )
        
        # Check for near-duplicates
        if len(self.vocab_original) > 1:
            dists = euclidean_distances(self.vocab_original)
            np.fill_diagonal(dists, np.inf)
            min_dist = dists.min()
            
            if min_dist < 1e-6:
                warnings.warn(
                    f"Vocabulary contains very close vectors (min distance: {min_dist:.2e}). "
                    "Consider increasing token diversity.",
                    UserWarning
                )
    
    def _normalize_vocabulary(self):
        """Center and scale vocabulary"""
        self.vocab_mean_ = self.vocab_original.mean(axis=0)
        self.vocab_std_ = self.vocab_original.std(axis=0) + 1e-8
        self.vocab = (self.vocab_original - self.vocab_mean_) / self.vocab_std_
    
    def fit(self, X, **kwargs):
        """
        Fit model and validate vocabulary dimensions match data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_series)
            Training data
        
        **kwargs : dict
            Additional parameters passed to parent fit
        
        Returns
        -------
        self : object
            Fitted estimator
        """
        # Call parent fit
        super().fit(X, **kwargs)
        
        # Validate vocabulary dimensions
        n_series = X.shape[1] if X.ndim > 1 else 1
        if self.vocab.shape[1] != n_series:
            raise ValueError(
                f"Vocabulary dimension ({self.vocab.shape[1]}) must match "
                f"number of series ({n_series})"
            )
        
        # Additional check for cosine distance
        if self.metric == 'cosine':
            norms = np.linalg.norm(self.vocab, axis=1)
            zero_vectors = norms < 1e-10
            if np.any(zero_vectors):
                raise ValueError(
                    f"Vocabulary contains {zero_vectors.sum()} zero/near-zero vectors. "
                    "Cosine distance requires non-zero vectors."
                )
        
        return self
    
    def _vectorized_map_to_tokens(self, continuous_preds):
        """
        Vectorized token mapping for efficiency.
        
        Parameters
        ----------
        continuous_preds : np.ndarray of shape (h, n_series)
            Continuous predictions
        
        Returns
        -------
        result : depends on return_mode
        errors : np.ndarray
            Distances to nearest tokens
        """
        # Normalize predictions if vocabulary was normalized
        if self.normalize_vocab:
            continuous_preds = (continuous_preds - self.vocab_mean_) / self.vocab_std_
        
        # Compute all distances at once
        dists = self.distance_func(continuous_preds, self.vocab)
        
        # Find nearest tokens
        nearest_indices = np.argmin(dists, axis=1)
        min_dists = dists[np.arange(len(dists)), nearest_indices]
        
        if self.return_mode == 'token_id':
            return nearest_indices, min_dists
        
        elif self.return_mode == 'token_vector':
            token_vecs = self.vocab[nearest_indices]
            # Denormalize if vocabulary was normalized
            if self.normalize_vocab:
                token_vecs = token_vecs * self.vocab_std_ + self.vocab_mean_
            return token_vecs, min_dists
        
        elif self.return_mode == 'both':
            # Return combined array: [token_id, dim_0, dim_1, ...]
            token_ids = nearest_indices.reshape(-1, 1)
            token_vecs = self.vocab[nearest_indices]
            # Denormalize if vocabulary was normalized
            if self.normalize_vocab:
                token_vecs = token_vecs * self.vocab_std_ + self.vocab_mean_
            combined = np.column_stack([token_ids, token_vecs])
            return combined, min_dists
        
        elif self.return_mode == 'probs':
            # Softmax of negative distances
            probs = softmax(-dists / self.softmax_temperature, axis=1)
            return probs, min_dists
    
    def predict(self, h=5, level=95, quantiles=None,
                return_discretization_error=False, **kwargs):
        """
        Generate discrete token predictions.
        
        Parameters
        ----------
        h : int, default=5
            Forecast horizon
        
        level : int, default=95
            Confidence level (only affects continuous forecasts)
        
        quantiles : list of float, optional
            Quantile levels
        
        return_discretization_error : bool, default=False
            If True, return (predictions, errors) tuple
        
        **kwargs : dict
            Additional parameters for parent predict
        
        Returns
        -------
        predictions : pd.DataFrame
            Discrete predictions. Format depends on return_mode:
            - 'token_id': single column 'token_id'
            - 'token_vector': columns 'dim_0', 'dim_1', ...
            - 'both': columns 'token_id', 'dim_0', 'dim_1', ...
            - 'probs': columns 'token_0_prob', 'token_1_prob', ...
        
        errors : pd.DataFrame (if return_discretization_error=True)
            Discretization errors (distances to nearest tokens)
        
        Warnings
        --------
        When prediction intervals are requested but only mean is discretized,
        a warning is issued. Use predict_token_distribution() for uncertainty
        in token space.
        """
        # Get continuous predictions from parent
        continuous_result = super().predict(h=h, level=level, 
                                           quantiles=quantiles, **kwargs)
        
        # FIXED: Robust type detection using duck typing
        if hasattr(continuous_result, '_fields'):  # Namedtuple
            if hasattr(continuous_result, 'sims') and continuous_result.sims is not None:
                # Simulation-based forecast
                return self._discretize_simulations(
                    continuous_result.sims,
                    return_discretization_error
                )
            elif hasattr(continuous_result, 'mean'):
                # Interval-based forecast - warn about information loss
                warnings.warn(
                    "Prediction intervals cannot be meaningfully discretized. "
                    "Only mean predictions are converted to tokens. "
                    "Use predict_token_distribution(replications=N) for "
                    "uncertainty in token space.",
                    UserWarning
                )
                return self._discretize_dataframe(
                    continuous_result.mean,
                    return_discretization_error
                )
        elif isinstance(continuous_result, pd.DataFrame):
            # Deterministic forecast
            return self._discretize_dataframe(
                continuous_result,
                return_discretization_error
            )
        else:
            raise NotImplementedError(
                f"Unhandled predict output type: {type(continuous_result)}"
            )
    
    def _discretize_dataframe(self, df, return_error=False):
        """Discretize a continuous prediction DataFrame"""
        # Use vectorized mapping
        result, errors = self._vectorized_map_to_tokens(df.values)
        
        # FIXED: Always return single DataFrame (even for 'both' mode)
        if self.return_mode == 'probs':
            result_df = pd.DataFrame(
                result,
                index=df.index,
                columns=[f'token_{i}_prob' for i in range(self.vocab_size)]
            )
        elif self.return_mode == 'both':
            # Combined format: token_id + dimensions
            columns = ['token_id'] + [f'dim_{i}' for i in range(self.vocab.shape[1])]
            result_df = pd.DataFrame(result, index=df.index, columns=columns)
            result_df['token_id'] = result_df['token_id'].astype(int)
        elif self.return_mode == 'token_id':
            result_df = pd.DataFrame(
                result.reshape(-1, 1),
                index=df.index,
                columns=['token_id']
            )
        else:  # 'token_vector'
            result_df = pd.DataFrame(
                result,
                index=df.index,
                columns=[f'dim_{i}' for i in range(self.vocab.shape[1])]
            )
        
        if return_error:
            error_df = pd.DataFrame(
                errors.reshape(-1, 1),
                index=df.index,
                columns=['discretization_error']
            )
            self.discretization_errors_ = error_df
            return result_df, error_df
        
        return result_df
    
    def _discretize_simulations(self, sims, return_error=False):
        """Discretize simulation paths"""
        discrete_sims = []
        all_errors = []
        
        for sim_df in sims:
            result, errors = self._vectorized_map_to_tokens(sim_df.values)
            
            if self.return_mode == 'probs':
                discrete_df = pd.DataFrame(
                    result,
                    index=sim_df.index,
                    columns=[f'token_{i}_prob' for i in range(self.vocab_size)]
                )
            elif self.return_mode == 'both':
                columns = ['token_id'] + [f'dim_{i}' for i in range(self.vocab.shape[1])]
                discrete_df = pd.DataFrame(result, index=sim_df.index, columns=columns)
                discrete_df['token_id'] = discrete_df['token_id'].astype(int)
            elif self.return_mode == 'token_id':
                discrete_df = pd.DataFrame(
                    result.reshape(-1, 1),
                    index=sim_df.index,
                    columns=['token_id']
                )
            else:  # 'token_vector'
                discrete_df = pd.DataFrame(
                    result,
                    index=sim_df.index,
                    columns=[f'dim_{i}' for i in range(self.vocab.shape[1])]
                )
            
            discrete_sims.append(discrete_df)
            
            if return_error:
                error_df = pd.DataFrame(
                    errors.reshape(-1, 1),
                    index=sim_df.index,
                    columns=['discretization_error']
                )
                all_errors.append(error_df)
        
        if return_error:
            return tuple(discrete_sims), tuple(all_errors)
        return tuple(discrete_sims)
    
    # ========== NEW: Uncertainty Quantification in Token Space ==========
    
    def predict_top_k(self, h=5, k=5, **kwargs):
        """
        Predict top-k most probable tokens per timestep.
        
        Parameters
        ----------
        h : int
            Forecast horizon
        k : int
            Number of top tokens to return
        **kwargs : dict
            Additional parameters for parent predict
        
        Returns
        -------
        predictions : pd.DataFrame
            Columns: token_1, prob_1, token_2, prob_2, ..., token_k, prob_k
        """
        continuous_result = super().predict(h=h, **kwargs)
        
        # Handle different return types
        if hasattr(continuous_result, 'mean'):
            preds = continuous_result.mean.values
            index = continuous_result.mean.index
        elif isinstance(continuous_result, pd.DataFrame):
            preds = continuous_result.values
            index = continuous_result.index
        else:
            raise ValueError("Cannot extract continuous predictions")
        
        # Compute probabilities
        dists = self.distance_func(preds, self.vocab)
        probs = softmax(-dists / self.softmax_temperature, axis=1)
        
        # Get top-k
        top_k_indices = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
        top_k_probs = np.take_along_axis(probs, top_k_indices, axis=1)
        
        # Format as DataFrame
        columns = []
        data = []
        for i in range(k):
            columns.extend([f'token_{i+1}', f'prob_{i+1}'])
            data.append(top_k_indices[:, i])
            data.append(top_k_probs[:, i])
        
        return pd.DataFrame(
            np.column_stack(data),
            index=index,
            columns=columns
        )
    
    def predict_token_distribution(self, h=5, replications=100, **kwargs):
        """
        Generate token probability distribution from simulation ensemble.
        
        This method provides meaningful uncertainty quantification in token space
        by discretizing multiple simulation paths and computing token frequencies.
        
        Parameters
        ----------
        h : int
            Forecast horizon
        replications : int
            Number of simulation paths
        **kwargs : dict
            Additional parameters for parent predict
        
        Returns
        -------
        frequencies : pd.DataFrame
            Token frequencies across simulations
            Columns: token_0_freq, token_1_freq, ..., token_V_freq
        
        entropy : pd.Series
            Shannon entropy per timestep (uncertainty measure)
        
        mode_tokens : pd.DataFrame
            Most frequent token per timestep
        
        Examples
        --------
        >>> freqs, entropy, mode = model.predict_token_distribution(h=10, replications=100)
        >>> # High entropy → uncertain prediction
        >>> uncertain_steps = entropy[entropy > 2.0]
        >>> # Use mode tokens for point predictions
        >>> predictions = mode['mode_token'].values
        """
        # Force simulation mode
        kwargs['replications'] = replications
        continuous_result = super().predict(h=h, **kwargs)
        
        # Extract simulations
        if hasattr(continuous_result, 'sims') and continuous_result.sims is not None:
            sims = continuous_result.sims
            index = continuous_result.mean.index
        else:
            raise ValueError(
                "predict_token_distribution requires simulation-based forecasting. "
                "Ensure replications > 0 and type_pi supports simulations."
            )
        
        # Discretize all paths
        all_tokens = []
        for sim in sims:
            tokens, _ = self._vectorized_map_to_tokens(sim.values)
            if self.return_mode == 'probs':
                # For probs mode, get argmax token
                tokens = np.argmax(tokens, axis=1)
            elif self.return_mode == 'both':
                # Extract token_id column
                tokens = tokens[:, 0].astype(int)
            elif self.return_mode == 'token_vector':
                # Map back to token IDs
                dists = self.distance_func(tokens, self.vocab)
                tokens = np.argmin(dists, axis=1)
            # else: token_id mode, already correct
            
            all_tokens.append(tokens)
        
        all_tokens = np.array(all_tokens)  # (replications, h)
        
        # Compute frequency distribution
        h_actual = all_tokens.shape[1]
        token_freqs = np.zeros((h_actual, self.vocab_size))
        
        for t in range(h_actual):
            unique, counts = np.unique(all_tokens[:, t], return_counts=True)
            token_freqs[t, unique] = counts / replications
        
        # Compute entropy
        epsilon = 1e-10
        entropy = -np.sum(
            token_freqs * np.log(token_freqs + epsilon),
            axis=1
        )
        
        # Get mode
        mode_tokens = np.argmax(token_freqs, axis=1)
        
        # Package results
        freq_df = pd.DataFrame(
            token_freqs,
            index=index,
            columns=[f'token_{i}_freq' for i in range(self.vocab_size)]
        )
        
        entropy_series = pd.Series(
            entropy,
            index=index,
            name='entropy'
        )
        
        mode_df = pd.DataFrame(
            mode_tokens,
            index=index,
            columns=['mode_token']
        )
        
        return freq_df, entropy_series, mode_df
    
    # ========== Utility Methods ==========
    
    def tokens_to_vectors(self, token_ids):
        """Convert token IDs to embedding vectors (in original scale)"""
        token_ids = np.asarray(token_ids).astype(int)
        assert np.all((token_ids >= 0) & (token_ids < self.vocab_size)), \
            f"Token IDs must be in range [0, {self.vocab_size-1}]"
        vectors = self.vocab[token_ids]
        # Denormalize if vocabulary was normalized
        if self.normalize_vocab:
            vectors = vectors * self.vocab_std_ + self.vocab_mean_
        return vectors
    
    def get_token_neighbors(self, token_id, k=5):
        """Find k nearest neighbors of a token"""
        assert 0 <= token_id < self.vocab_size, \
            f"token_id must be in range [0, {self.vocab_size-1}]"
        
        token_vec = self.vocab[token_id].reshape(1, -1)
        dists = self.distance_func(token_vec, self.vocab).flatten()
        
        sorted_indices = np.argsort(dists)
        sorted_indices = sorted_indices[sorted_indices != token_id][:k]
        
        return pd.DataFrame({
            'neighbor_id': sorted_indices,
            'distance': dists[sorted_indices]
        })
    
    def compute_vocab_coverage(self, predictions):
        """Compute vocabulary usage statistics"""
        if 'token_id' not in predictions.columns:
            raise ValueError("predictions must have 'token_id' column")
        
        token_ids = predictions['token_id'].values
        unique_tokens = np.unique(token_ids)
        freq = pd.Series(token_ids).value_counts().sort_index()
        
        return {
            'unique_tokens': len(unique_tokens),
            'coverage_pct': 100 * len(unique_tokens) / self.vocab_size,
            'token_frequencies': freq,
            'most_common_token': freq.idxmax() if len(freq) > 0 else None,
            'least_common_token': freq.idxmin() if len(freq) > 0 else None
        }
    
    def diagnose_vocabulary(self):
        """
        Comprehensive vocabulary quality diagnostics.
        
        Returns
        -------
        report : dict
            Quality metrics including distances, condition number, coverage
        """
        # Use original vocabulary for diagnostics to get meaningful statistics
        vocab_to_diagnose = self.vocab_original
        
        report = {
            'vocab_size': self.vocab_size,
            'embedding_dim': vocab_to_diagnose.shape[1],
            'normalized': self.normalize_vocab,
        }
        
        # Pairwise distances
        dists = euclidean_distances(vocab_to_diagnose)
        np.fill_diagonal(dists, np.inf)
        
        report['min_pairwise_distance'] = dists.min()
        report['max_pairwise_distance'] = dists.max()
        report['mean_pairwise_distance'] = dists[dists != np.inf].mean()
        
        # Condition number
        U, s, Vt = np.linalg.svd(vocab_to_diagnose, full_matrices=False)
        report['condition_number'] = s.max() / (s.min() + 1e-10)
        
        # Coverage volume
        ranges = vocab_to_diagnose.max(axis=0) - vocab_to_diagnose.min(axis=0)
        report['coverage_volume'] = np.prod(ranges)
        
        # Duplicates
        unique_rows = np.unique(vocab_to_diagnose, axis=0)
        report['duplicate_count'] = len(vocab_to_diagnose) - len(unique_rows)
        
        return report
    
    def print_vocabulary_report(self):
        """Print human-readable vocabulary diagnostics"""
        report = self.diagnose_vocabulary()
        
        print("=" * 60)
        print("VOCABULARY QUALITY REPORT")
        print("=" * 60)
        print(f"Vocabulary size: {report['vocab_size']} tokens")
        print(f"Embedding dimension: {report['embedding_dim']}")
        print(f"\nPairwise Distances:")
        print(f"  Min:  {report['min_pairwise_distance']:.6f}")
        print(f"  Mean: {report['mean_pairwise_distance']:.6f}")
        print(f"  Max:  {report['max_pairwise_distance']:.6f}")
        print(f"\nVocabulary Health:")
        print(f"  Condition number: {report['condition_number']:.2f}")
        if report['condition_number'] > 1000:
            print("  ⚠️  WARNING: High condition number may indicate redundant tokens")
        print(f"  Duplicate tokens: {report['duplicate_count']}")
        if report['duplicate_count'] > 0:
            print("  ⚠️  WARNING: Duplicates reduce effective vocabulary size")
        print(f"  Coverage volume: {report['coverage_volume']:.2e}")
        print("=" * 60)