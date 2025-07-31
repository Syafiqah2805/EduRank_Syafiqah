import numpy as np
import pandas as pd

def normalize_matrix(matrix, criteria_type):
    """
    Normalize the decision matrix based on benefit or cost type.
    
    Args:
        matrix (2D array): The decision matrix
        criteria_type (list): List of "benefit" or "cost" per criterion

    Returns:
        2D array: Normalized matrix
    """
    norm_matrix = np.zeros_like(matrix, dtype=float)
    for j in range(matrix.shape[1]):
        if criteria_type[j] == 'benefit':
            norm_matrix[:, j] = matrix[:, j] / np.max(matrix[:, j])
        elif criteria_type[j] == 'cost':
            norm_matrix[:, j] = np.min(matrix[:, j]) / matrix[:, j]
        else:
            raise ValueError(f"Invalid criteria_type '{criteria_type[j]}'. Use 'benefit' or 'cost'.")
    return norm_matrix

def compute_q1(norm_matrix, weights):
    """
    Compute Q1 using Weighted Sum Model (WSM)
    """
    return np.sum(norm_matrix * weights, axis=1)

def compute_q2(norm_matrix, weights):
    """
    Compute Q2 using Weighted Product Model (WPM)
    """
    return np.prod(np.power(norm_matrix, weights), axis=1)

def compute_final_q(q1, q2, lambd=0.5):
    """
    Combine Q1 and Q2 using λ
    """
    return lambd * q1 + (1 - lambd) * q2

def rank_alternatives(q):
    """
    Rank alternatives based on Q values (descending)
    """
    return np.argsort(q)[::-1] + 1  # +1 to rank from 1

def waspas(decision_matrix, weights, criteria_type, lambd=0.5):
    """
    Full WASPAS method
    
    Args:
        decision_matrix (2D array): Alternatives x Criteria matrix
        weights (1D array): Criteria weights
        criteria_type (list): "benefit" or "cost" for each criterion
        lambd (float): Weight for combining Q1 and Q2, usually 0.5

    Returns:
        DataFrame: Results with Q1, Q2, Q, and Rankings
    """
    norm_matrix = normalize_matrix(decision_matrix, criteria_type)
    q1 = compute_q1(norm_matrix, weights)
    q2 = compute_q2(norm_matrix, weights)
    q = compute_final_q(q1, q2, lambd)
    rank = rank_alternatives(q)
    
    df_result = pd.DataFrame({
        'Q1 (WSM)': q1,
        'Q2 (WPM)': q2,
        'Q (Final)': q,
        'Rank': rank
    })
    return df_result

# Example usage:
if __name__ == "__main__":
    # Sample data
    decision_matrix = np.array([
        [250, 16, 12, 5],
        [200, 20, 8, 3],
        [300, 11, 10, 4],
        [275, 12, 11, 2]
    ])
    
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    criteria_type = ['benefit', 'cost', 'benefit', 'cost']
    
    result_df = waspas(decision_matrix, weights, criteria_type, lambd=0.5)
    print(result_df)

