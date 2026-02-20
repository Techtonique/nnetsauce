"""
DiscreteTokenMTS - Complete Working Example
===========================================

This example demonstrates correct usage patterns for DiscreteTokenMTS,
handling all the quirks of simulation-based vs deterministic forecasting.

Key insights:
1. type_pi controls WHETHER simulations are possible
2. replications controls IF simulations are ACTUALLY generated
3. Return type depends on whether simulations were generated

Author: T. Moudiki
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from nnetsauce import DiscreteTokenMTS

np.random.seed(42)

print("="*70)
print("DISCRETETOKENMTS - COMPLETE WORKING EXAMPLE")
print("="*70)

# ============================================================================
# SETUP
# ============================================================================

corpus = """
the quick brown fox jumps over the lazy dog
the cat sat on the mat
the dog ran in the park
"""

corpus = corpus.lower().strip()
chars = sorted(set(corpus))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# Create embeddings
embedding_dim = 8
vocab = np.zeros((len(chars), embedding_dim))
for i, char in enumerate(chars):
    vocab[i, 0] = 1.0 if char in 'aeiou' else -1.0
    vocab[i, 1] = 1.0 if char.isspace() else -1.0
    if char.isalpha():
        vocab[i, 2] = (ord(char) - ord('a')) / 25.0
    vocab[i, 3:] = np.random.randn(embedding_dim - 3) * 0.3

# Training data
indices = [char_to_idx[c] for c in corpus]
embeddings = vocab[indices]
df = pd.DataFrame(embeddings, columns=[f'dim_{i}' for i in range(embedding_dim)])

print(f"\nCorpus length: {len(corpus)} characters")
print(f"Vocabulary: {len(chars)} unique characters")
print(f"Training data: {df.shape}")

# ============================================================================
# CASE 1: DETERMINISTIC - No simulations (simplest)
# ============================================================================

print("\n" + "="*70)
print("CASE 1: DETERMINISTIC (No type_pi, No replications)")
print("="*70)

model1 = DiscreteTokenMTS(
    obj=Ridge(alpha=1.0),
    vocab=vocab,
    lags=5,
    return_mode='token_id'
    # NO type_pi → no simulation capability
)

model1.fit(df)
result1 = model1.predict(h=30)

print(f"Result type: {type(result1)}")
print(f"Result: {type(result1).__name__}")

# Handle result based on type
if hasattr(result1, '_fields'):  # namedtuple
    if hasattr(result1, 'mean'):
        predictions1 = result1.mean
    else:
        predictions1 = result1[0]  # First field
else:
    predictions1 = result1

text1 = ''.join([idx_to_char[int(t)] for t in predictions1['token_id'].values])
print(f"Generated: '{text1}'")

# ============================================================================
# CASE 2: GAUSSIAN TYPE_PI (Doesn't require replications)
# ============================================================================

print("\n" + "="*70)
print("CASE 2: PREDICTION INTERVALS (type_pi='gaussian', no simulations)")
print("="*70)

model2 = DiscreteTokenMTS(
    obj=Ridge(alpha=1.0),
    vocab=vocab,
    lags=5,
    return_mode='token_id',
    type_pi='gaussian'  # Gaussian intervals don't require replications
)

model2.fit(df)
result2 = model2.predict(h=30)

print(f"Result type: {type(result2)}")

# This returns namedtuple with mean/lower/upper but NO sims
if hasattr(result2, '_fields'):
    print(f"Fields: {result2._fields}")
    if hasattr(result2, 'sims'):
        print(f"Has sims? {result2.sims is not None}")
    predictions2 = result2.mean if hasattr(result2, 'mean') else result2[0]
else:
    predictions2 = result2

text2 = ''.join([idx_to_char[int(t)] for t in predictions2['token_id'].values])
print(f"Generated: '{text2}'")

# ============================================================================
# CASE 3: SIMULATIONS ENABLED - Correct usage
# ============================================================================

print("\n" + "="*70)
print("CASE 3: SIMULATIONS (type_pi + replications)")
print("="*70)

model3 = DiscreteTokenMTS(
    obj=Ridge(alpha=1.0),
    vocab=vocab,
    lags=5,
    return_mode='token_id',
    type_pi='bootstrap',
    replications=50  # Generate 50 simulation paths
)

model3.fit(df)
result3 = model3.predict(h=30)

print(f"Result type: {type(result3)}")

# With replications, result is a tuple of DataFrames (simulations)
if isinstance(result3, tuple):
    print(f"Number of simulations: {len(result3)}")
    print(f"Each simulation type: {type(result3[0])}")
    
    # Get first simulation as example
    predictions3 = result3[0]
else:
    predictions3 = result3

text3 = ''.join([idx_to_char[int(t)] for t in predictions3['token_id'].values])
print(f"First simulation: '{text3}'")

# ============================================================================
# CASE 4: PREDICT_TOKEN_DISTRIBUTION - Best for uncertainty
# ============================================================================

print("\n" + "="*70)
print("CASE 4: UNCERTAINTY QUANTIFICATION (predict_token_distribution)")
print("="*70)

# For this, type_pi MUST be set
model4 = DiscreteTokenMTS(
    obj=Ridge(alpha=1.0),
    vocab=vocab,
    lags=5,
    return_mode='token_id',
    type_pi='bootstrap',  # Required for simulations
    replications=50  # Generates 50 paths internally
)

model4.fit(df)

# Method 1: Pass replications to method (recommended)
freqs, entropy, mode = model4.predict_token_distribution(
    h=30,    
)

print(f"\nEntropy stats:")
print(f"  Mean: {entropy.mean():.3f}")
print(f"  Max:  {entropy.max():.3f}")
print(f"  Min:  {entropy.min():.3f}")

mode_text = ''.join([idx_to_char[int(t)] for t in mode['mode_token'].values])
print(f"Mode prediction: '{mode_text}'")

# ============================================================================
# CASE 5: PROBABILISTIC WITH TEMPERATURE
# ============================================================================

print("\n" + "="*70)
print("CASE 5: PROBABILISTIC (return_mode='probs' + temperature)")
print("="*70)

def safe_extract_dataframe(result):
    """
    Safely extract a DataFrame from any MTS/DiscreteTokenMTS result type.
    
    Handles:
    - Direct DataFrame
    - Tuple of DataFrames (simulations)
    - Namedtuple with .mean attribute
    - Namedtuple without .mean
    """
    if isinstance(result, pd.DataFrame):
        return result
    elif isinstance(result, tuple):
        return result[0]  # First simulation
    elif hasattr(result, '_fields'):
        # It's a namedtuple
        if hasattr(result, 'mean'):
            return result.mean
        else:
            return result[0]
    else:
        # Unknown type, try to return as-is
        return result


for temp in [0.5, 1.5]:
    model_prob = DiscreteTokenMTS(
        obj=Ridge(alpha=1.0),
        vocab=vocab,
        lags=5,
        return_mode='probs',
        softmax_temperature=temp
    )
    
    model_prob.fit(df)
    result_prob = model_prob.predict(h=25)
    
    # Use helper to extract DataFrame
    probs_df = safe_extract_dataframe(result_prob)
    
    # Sample
    sampled = []
    for i in range(len(probs_df)):
        prob_dist = probs_df.iloc[i].values
        token = np.random.choice(len(chars), p=prob_dist)
        sampled.append(idx_to_char[token])
    
    print(f"T={temp}: '{''.join(sampled)}'")

# ============================================================================
# BEST PRACTICES SUMMARY
# ============================================================================

print("\n" + "="*70)
print("BEST PRACTICES")
print("="*70)
print("""
1. SIMPLE PREDICTIONS:
   - Don't set type_pi OR use type_pi='gaussian'
   - Don't set replications
   → Returns DataFrame or namedtuple.mean

2. PREDICTION INTERVALS (No simulations):
   - Set type_pi='gaussian'
   → Returns namedtuple(mean, lower, upper)
   → Assumes Gaussian residuals, fast

3. ENABLE SIMULATIONS:
   - Set type_pi='bootstrap' (or 'kde')
   - MUST set replications=N (required!)
   → Returns tuple of N DataFrames

4. UNCERTAINTY QUANTIFICATION:
   - Set type_pi='bootstrap'
   - Use predict_token_distribution(replications=N)
   → Returns (frequencies, entropy, mode)

5. PROBABILISTIC SAMPLING:
   - Set return_mode='probs'
   - Set softmax_temperature
   - Sample from probability distributions
   
6. HANDLING RETURNS:
   Always check: hasattr(result, '_fields') or isinstance(result, tuple)
   
7. KEY INSIGHT:
   - type_pi='gaussian' → intervals without simulations
   - type_pi='bootstrap'/'kde' → REQUIRES replications
   - For token uncertainty, use predict_token_distribution()
   
8. AVOID:
   - Using type_pi='bootstrap' without replications (will error)
   - Setting both constructor replications AND predict replications
   - Using return_discretization_error with simulations (complex)
""")

print("\n✓ Example complete!")
