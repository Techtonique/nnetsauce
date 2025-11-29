import jax
import jax.numpy as jnp
from jax import random, grad, jit
import numpy as np
from typing import Optional, Tuple, Dict
from functools import partial


class AttentionMechanism:
    """
    A comprehensive class implementing various attention mechanisms
    for both univariate time series and tabular data using JAX.

    Supported attention types:
    - Scaled Dot-Product Attention
    - Additive (Bahdanau) Attention
    - Multi-Head Attention
    - Self-Attention
    - Temporal Attention (for sequences)
    - Feature Attention (for tabular data)
    - Cross-Attention
    - Context Vector Attention
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden dimension for attention computations
            num_heads: Number of attention heads for multi-head attention
            dropout: Dropout rate
            seed: Random seed for parameter initialization
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Initialize random key
        self.rng = random.PRNGKey(seed)

        # Initialize parameters
        self.params = self._initialize_parameters()

        assert (
            hidden_dim % num_heads == 0
        ), "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads

    def _initialize_parameters(self) -> Dict:
        """Initialize all network parameters using JAX"""
        keys = random.split(self.rng, 20)

        def init_weight(key, shape):
            return random.normal(key, shape) * np.sqrt(2.0 / shape[0])

        def init_bias(shape):
            return jnp.zeros(shape)

        params = {
            # Scaled Dot-Product Attention
            "query_w": init_weight(keys[0], (self.input_dim, self.hidden_dim)),
            "query_b": init_bias((self.hidden_dim,)),
            "key_w": init_weight(keys[1], (self.input_dim, self.hidden_dim)),
            "key_b": init_bias((self.hidden_dim,)),
            "value_w": init_weight(keys[2], (self.input_dim, self.hidden_dim)),
            "value_b": init_bias((self.hidden_dim,)),
            # Additive Attention
            "additive_query_w": init_weight(
                keys[3], (self.input_dim, self.hidden_dim)
            ),
            "additive_query_b": init_bias((self.hidden_dim,)),
            "additive_key_w": init_weight(
                keys[4], (self.input_dim, self.hidden_dim)
            ),
            "additive_key_b": init_bias((self.hidden_dim,)),
            "additive_v_w": init_weight(keys[5], (self.hidden_dim, 1)),
            "additive_v_b": init_bias((1,)),
            # Multi-Head Attention
            "mha_query_w": init_weight(
                keys[6], (self.input_dim, self.hidden_dim)
            ),
            "mha_query_b": init_bias((self.hidden_dim,)),
            "mha_key_w": init_weight(
                keys[7], (self.input_dim, self.hidden_dim)
            ),
            "mha_key_b": init_bias((self.hidden_dim,)),
            "mha_value_w": init_weight(
                keys[8], (self.input_dim, self.hidden_dim)
            ),
            "mha_value_b": init_bias((self.hidden_dim,)),
            "mha_output_w": init_weight(
                keys[9], (self.hidden_dim, self.hidden_dim)
            ),
            "mha_output_b": init_bias((self.hidden_dim,)),
            # Feature Attention
            "feature_w1": init_weight(
                keys[10], (self.input_dim, self.hidden_dim)
            ),
            "feature_b1": init_bias((self.hidden_dim,)),
            "feature_w2": init_weight(
                keys[11], (self.hidden_dim, self.input_dim)
            ),
            "feature_b2": init_bias((self.input_dim,)),
            # Temporal Attention
            "temporal_query_w": init_weight(
                keys[12], (self.input_dim, self.hidden_dim)
            ),
            "temporal_query_b": init_bias((self.hidden_dim,)),
            "temporal_key_w": init_weight(
                keys[13], (self.input_dim, self.hidden_dim)
            ),
            "temporal_key_b": init_bias((self.hidden_dim,)),
            # Context Vector Attention
            "context_vector": random.normal(keys[14], (1, 1, self.hidden_dim)),
            "context_query_w": init_weight(
                keys[15], (self.hidden_dim, self.hidden_dim)
            ),
            "context_query_b": init_bias((self.hidden_dim,)),
            "context_key_w": init_weight(
                keys[16], (self.input_dim, self.hidden_dim)
            ),
            "context_key_b": init_bias((self.hidden_dim,)),
            "context_value_w": init_weight(
                keys[17], (self.input_dim, self.hidden_dim)
            ),
            "context_value_b": init_bias((self.hidden_dim,)),
        }

        return params

    @staticmethod
    @jit
    def _apply_dropout(
        x: jnp.ndarray,
        key: jax.random.PRNGKey,
        rate: float,
        training: bool = True,
    ) -> jnp.ndarray:
        """Apply dropout"""
        if training and rate > 0:
            keep_prob = 1 - rate
            mask = random.bernoulli(key, keep_prob, x.shape)
            return jnp.where(mask, x / keep_prob, 0)
        return x

    @partial(jit, static_argnums=(0,))
    def scaled_dot_product_attention(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        params: Dict,
        mask: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Scaled Dot-Product Attention

        Args:
            query: (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            key: (batch_size, seq_len, input_dim)
            value: (batch_size, seq_len, input_dim)
            params: Parameter dictionary
            mask: Optional mask (batch_size, seq_len)
            training: Whether in training mode

        Returns:
            context: Attended context vector
            attention_weights: Attention weights
        """
        # Project inputs
        Q = jnp.dot(query, params["query_w"]) + params["query_b"]
        K = jnp.dot(key, params["key_w"]) + params["key_b"]
        V = jnp.dot(value, params["value_w"]) + params["value_b"]

        # Compute attention scores
        scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1))
        scores = scores / jnp.sqrt(self.hidden_dim)

        # Apply mask if provided
        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)

        # Compute attention weights
        attention_weights = jax.nn.softmax(scores, axis=-1)

        # Apply attention to values
        context = jnp.matmul(attention_weights, V)

        return context, attention_weights

    @partial(jit, static_argnums=(0,))
    def additive_attention(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        params: Dict,
        mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Additive (Bahdanau) Attention

        Args:
            query: (batch_size, hidden_dim) or (batch_size, 1, hidden_dim)
            key: (batch_size, seq_len, hidden_dim)
            value: (batch_size, seq_len, hidden_dim)
            params: Parameter dictionary
            mask: Optional mask

        Returns:
            context: Attended context vector
            attention_weights: Attention weights
        """
        # Ensure query has seq_len dimension
        if query.ndim == 2:
            query = jnp.expand_dims(query, axis=1)

        # Project query and key
        Q = (
            jnp.dot(query, params["additive_query_w"])
            + params["additive_query_b"]
        )
        K = jnp.dot(key, params["additive_key_w"]) + params["additive_key_b"]

        # Additive attention: score = v^T tanh(W_q Q + W_k K)
        combined = jnp.tanh(Q + K)
        scores = (
            jnp.dot(combined, params["additive_v_w"]) + params["additive_v_b"]
        )
        scores = jnp.squeeze(scores, axis=-1)

        # Apply mask if provided
        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)

        # Compute attention weights
        attention_weights = jax.nn.softmax(scores, axis=-1)

        # Apply attention to values
        context = jnp.matmul(jnp.expand_dims(attention_weights, axis=1), value)
        context = jnp.squeeze(context, axis=1)

        return context, attention_weights

    @partial(jit, static_argnums=(0,))
    def multi_head_attention(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        params: Dict,
        mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Multi-Head Attention

        Args:
            query: (batch_size, seq_len_q, input_dim)
            key: (batch_size, seq_len_k, input_dim)
            value: (batch_size, seq_len_v, input_dim)
            params: Parameter dictionary
            mask: Optional mask

        Returns:
            output: Multi-head attention output
            attention_weights: Attention weights from all heads
        """
        batch_size = query.shape[0]

        # Project and reshape for multi-head attention
        Q = jnp.dot(query, params["mha_query_w"]) + params["mha_query_b"]
        K = jnp.dot(key, params["mha_key_w"]) + params["mha_key_b"]
        V = jnp.dot(value, params["mha_value_w"]) + params["mha_value_b"]

        Q = Q.reshape(batch_size, -1, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, -1, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, -1, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, num_heads, seq_len, head_dim)
        Q = jnp.transpose(Q, (0, 2, 1, 3))
        K = jnp.transpose(K, (0, 2, 1, 3))
        V = jnp.transpose(V, (0, 2, 1, 3))

        # Compute attention scores
        scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / jnp.sqrt(
            self.head_dim
        )

        # Apply mask if provided
        if mask is not None:
            mask_expanded = jnp.expand_dims(jnp.expand_dims(mask, 1), 2)
            scores = jnp.where(mask_expanded == 0, -1e9, scores)

        # Attention weights
        attention_weights = jax.nn.softmax(scores, axis=-1)

        # Apply attention to values
        context = jnp.matmul(attention_weights, V)

        # Reshape back: (batch, seq_len, hidden_dim)
        context = jnp.transpose(context, (0, 2, 1, 3))
        context = context.reshape(batch_size, -1, self.hidden_dim)

        # Final linear projection
        output = (
            jnp.dot(context, params["mha_output_w"]) + params["mha_output_b"]
        )

        return output, attention_weights

    @partial(jit, static_argnums=(0,))
    def self_attention(
        self, x: jnp.ndarray, params: Dict, mask: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Self-Attention mechanism"""
        return self.scaled_dot_product_attention(x, x, x, params, mask)

    @partial(jit, static_argnums=(0,))
    def temporal_attention(
        self, x: jnp.ndarray, params: Dict, mask: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Temporal Attention for time series data

        Args:
            x: (batch_size, seq_len, input_dim)
            params: Parameter dictionary
            mask: Optional mask

        Returns:
            context: Temporally attended context
            attention_weights: Temporal attention weights
        """
        # Use last time step as query
        query = x[:, -1:, :]

        Q = (
            jnp.dot(query, params["temporal_query_w"])
            + params["temporal_query_b"]
        )
        K = jnp.dot(x, params["temporal_key_w"]) + params["temporal_key_b"]

        # Compute attention scores
        scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / jnp.sqrt(
            self.hidden_dim
        )
        scores = jnp.squeeze(scores, axis=1)

        # Apply mask if provided
        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)

        # Attention weights
        attention_weights = jax.nn.softmax(scores, axis=-1)

        # Apply attention
        context = jnp.matmul(jnp.expand_dims(attention_weights, axis=1), x)
        context = jnp.squeeze(context, axis=1)

        return context, attention_weights

    @partial(jit, static_argnums=(0,))
    def feature_attention_tabular(
        self, x: jnp.ndarray, params: Dict
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Feature Attention for tabular data

        Args:
            x: (batch_size, num_features)
            params: Parameter dictionary

        Returns:
            output: Feature-weighted output
            attention_weights: Feature importance weights
        """
        # Compute feature attention weights
        hidden = jnp.dot(x, params["feature_w1"]) + params["feature_b1"]
        hidden = jnp.tanh(hidden)
        logits = jnp.dot(hidden, params["feature_w2"]) + params["feature_b2"]
        attention_weights = jax.nn.softmax(logits, axis=-1)

        # Apply attention to features
        output = x * attention_weights

        return output, attention_weights

    @partial(jit, static_argnums=(0,))
    def context_vector_attention(
        self, x: jnp.ndarray, params: Dict, mask: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Context Vector Attention
        Uses a learnable global context vector as the query.

        Args:
            x: (batch_size, seq_len, input_dim)
            params: Parameter dictionary
            mask: Optional mask (batch_size, seq_len)

        Returns:
            context: Global context representation (batch_size, hidden_dim)
            attention_weights: Attention weights (batch_size, seq_len)
        """
        batch_size = x.shape[0]

        # Expand context vector for batch
        context_vec = jnp.broadcast_to(
            params["context_vector"], (batch_size, 1, self.hidden_dim)
        )

        # Project context vector and input
        Q = (
            jnp.dot(context_vec, params["context_query_w"])
            + params["context_query_b"]
        )
        K = jnp.dot(x, params["context_key_w"]) + params["context_key_b"]
        V = jnp.dot(x, params["context_value_w"]) + params["context_value_b"]

        # Compute attention scores
        scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / jnp.sqrt(
            self.hidden_dim
        )
        scores = jnp.squeeze(scores, axis=1)

        # Apply mask if provided
        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)

        # Compute attention weights
        attention_weights = jax.nn.softmax(scores, axis=-1)

        # Apply attention to values
        context = jnp.matmul(jnp.expand_dims(attention_weights, axis=1), V)
        context = jnp.squeeze(context, axis=1)

        return context, attention_weights

    @partial(jit, static_argnums=(0,))
    def cross_attention(
        self,
        query: jnp.ndarray,
        key_value: jnp.ndarray,
        params: Dict,
        mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Cross-Attention between two different sequences"""
        return self.scaled_dot_product_attention(
            query, key_value, key_value, params, mask
        )

    def __call__(
        self,
        x: jnp.ndarray,
        attention_type: str = "scaled_dot_product",
        query: Optional[jnp.ndarray] = None,
        key_value: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass with specified attention mechanism

        Args:
            x: Input tensor
            attention_type: Type of attention to use
            query: Optional query for cross-attention
            key_value: Optional key-value for cross-attention
            mask: Optional mask
            training: Whether in training mode

        Returns:
            output: Attention output
            attention_weights: Attention weights
        """
        if attention_type == "scaled_dot_product":
            return self.scaled_dot_product_attention(
                x, x, x, self.params, mask, training
            )
        elif attention_type == "additive":
            return self.additive_attention(
                x[:, -1:, :], x, x, self.params, mask
            )
        elif attention_type == "multi_head":
            return self.multi_head_attention(x, x, x, self.params, mask)
        elif attention_type == "self":
            return self.self_attention(x, self.params, mask)
        elif attention_type == "temporal":
            return self.temporal_attention(x, self.params, mask)
        elif attention_type == "feature":
            return self.feature_attention_tabular(x, self.params)
        elif attention_type == "cross":
            if query is None or key_value is None:
                raise ValueError(
                    "Cross-attention requires both query and key_value"
                )
            return self.cross_attention(query, key_value, self.params, mask)
        elif attention_type == "context_vector":
            return self.context_vector_attention(x, self.params, mask)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
