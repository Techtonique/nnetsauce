# import os 
# import nnetsauce as ns 
# import numpy as np 
# try: 
#     import jax.numpy as jnp
#     JAX_AVAILABLE = True
# except ImportError: 
#     JAX_AVAILABLE = False  
# from nnetsauce.attention import AttentionMechanism
# from sklearn.datasets import load_diabetes, fetch_california_housing
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
# from time import time 

# print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# # Set random seed for reproducibility
# np.random.seed(42)

# if JAX_AVAILABLE: 

#     # Example 1: Univariate time series with temporal attention
#     print("=" * 50)
#     print("Example 1: Univariate Time Series")
#     print("=" * 50)
#     batch_size, seq_len, input_dim = 32, 10, 1
#     x_univariate = jnp.array(np.random.randn(batch_size, seq_len, input_dim))

#     attention = AttentionMechanism(input_dim=input_dim, hidden_dim=32, num_heads=4)
#     context, weights = attention(x_univariate, attention_type='temporal')

#     print(f"Input shape: {x_univariate.shape}")
#     print(f"Context shape: {context.shape}")
#     print(f"Attention weights shape: {weights.shape}")
#     print(f"Sample attention weights (first batch): {np.array(weights[0])}")

#     # Example 2: Tabular data with feature attention
#     print("\n" + "=" * 50)
#     print("Example 2: Tabular Data with Feature Attention")
#     print("=" * 50)
#     batch_size, num_features = 32, 10
#     x_tabular = jnp.array(np.random.randn(batch_size, num_features))

#     attention_tab = AttentionMechanism(input_dim=num_features, hidden_dim=32)
#     output, feature_weights = attention_tab(x_tabular, attention_type='feature')

#     print(f"Input shape: {x_tabular.shape}")
#     print(f"Output shape: {output.shape}")
#     print(f"Feature weights shape: {feature_weights.shape}")
#     print(f"Feature importance (first batch): {np.array(feature_weights[0])}")

#     # Example 3: Multi-head attention on sequences
#     print("\n" + "=" * 50)
#     print("Example 3: Multi-Head Attention")
#     print("=" * 50)
#     batch_size, seq_len, input_dim = 16, 8, 16
#     x_seq = jnp.array(np.random.randn(batch_size, seq_len, input_dim))

#     attention_mha = AttentionMechanism(input_dim=input_dim, hidden_dim=64, num_heads=8)
#     output_mha, weights_mha = attention_mha(x_seq, attention_type='multi_head')

#     print(f"Input shape: {x_seq.shape}")
#     print(f"Output shape: {output_mha.shape}")
#     print(f"Attention weights shape (with heads): {weights_mha.shape}")

#     # Example 4: Cross-attention
#     print("\n" + "=" * 50)
#     print("Example 4: Cross-Attention")
#     print("=" * 50)
#     batch_size = 16
#     query_seq = jnp.array(np.random.randn(batch_size, 5, input_dim))
#     kv_seq = jnp.array(np.random.randn(batch_size, 10, input_dim))

#     cross_output, cross_weights = attention_mha(
#         None, 
#         attention_type='cross',
#         query=query_seq,
#         key_value=kv_seq
#     )

#     print(f"Query shape: {query_seq.shape}")
#     print(f"Key-Value shape: {kv_seq.shape}")
#     print(f"Cross-attention output shape: {cross_output.shape}")
#     print(f"Cross-attention weights shape: {cross_weights.shape}")

#     # Example 5: Context Vector Attention
#     print("\n" + "=" * 50)
#     print("Example 5: Context Vector Attention")
#     print("=" * 50)
#     batch_size, seq_len, input_dim = 32, 15, 8
#     x_context = jnp.array(np.random.randn(batch_size, seq_len, input_dim))

#     attention_ctx = AttentionMechanism(input_dim=input_dim, hidden_dim=64)
#     context_output, context_weights = attention_ctx(x_context, attention_type='context_vector')

#     print(f"Input shape: {x_context.shape}")
#     print(f"Context output shape: {context_output.shape}")
#     print(f"Context attention weights shape: {context_weights.shape}")
#     print(f"Sample context weights (first batch): {np.array(context_weights[0])}")
#     print(f"\nNote: Context vector attention produces a fixed-size global representation")
#     print(f"regardless of input sequence length, making it ideal for classification tasks.")

#     # Demonstrate JAX's JIT compilation benefit
#     print("\n" + "=" * 50)
#     print("JAX Performance Benefits")
#     print("=" * 50)
#     print("All methods are JIT-compiled for fast execution!")
#     print("JAX provides automatic differentiation and GPU/TPU acceleration.")