import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# Sample Decision Matrix
# -----------------------------
data = {
    'Alternative': [
        'Traditional Learning', 'Gamification', 'AI Tools', 'Mobile Learning'
    ],
    'Engagement': [0.80, 0.92, 0.75, 0.85],
    'Retention Rate': [0.75, 0.90, 0.72, 0.80],
    'Cognitive Load': [0.60, 0.50, 0.70, 0.65],
    'Application in Context': [0.70, 0.85, 0.78, 0.80],
    'Accessibility': [0.85, 0.80, 0.90, 0.88],
    'Feedback Mechanism': [0.90, 0.85, 0.80, 0.82]
}

df = pd.DataFrame(data)

# Criteria weights (must sum to 1)
weights = {
    'Engagement': 0.15,
    'Retention Rate': 0.20,
    'Cognitive Load': 0.10,
    'Application in Context': 0.15,
    'Accessibility': 0.20,
    'Feedback Mechanism': 0.20
}

# Convert weights to DataFrame for display
weights_df = pd.DataFrame(list(weights.items()), columns=['Criterion', 'Weight'])

# -----------------------------
# WASPAS Calculation
# -----------------------------
def calculate_waspas(df, weights):
    # Extract only numeric part
    criteria_cols = df.columns[1:]
    numeric_df = df[criteria_cols].copy()

    # Step 1: Normalize matrix
    normalized_df = numeric_df / np.sqrt((numeric_df**2).sum())
    
    # Step 2: Weighted normalized matrix
    weighted_norm_df = normalized_df.copy()
    for col in criteria_cols:
        weighted_norm_df[col] = weighted_norm_df[col] * weights[col]
    
    # Step 3: WASPAS score (WSM + WPM hybrid)
    # WSM component (sum of weighted normalized values)
    wsm_score = weighted_norm_df.sum(axis=1)
    
    # WPM component (product of values raised to weights)
    wpm_score = np.prod(normalized_df ** list(weights.values()), axis=1)
    
    # WASPAS final score (λ=0.5)
    final_score = 0.5 * wsm_score + 0.5 * wpm_score
    
    result_df = df[['Alternative']].copy()
    result_df['WASPAS Score'] = final_score
    result_df['Rank'] = result_df['WASPAS Score'].rank(ascending=False)
    
    return normalized_df, weighted_norm_df, result_df

normalized_df, weighted_norm_df, result_df = calculate_waspas(df, weights)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("SmartRankEDU 360: Big Data-Enhanced Educational Strategy Evaluation Using WASPAS")

st.subheader("Step 1: Decision Matrix")
st.dataframe(df)

st.subheader("Step 2: Criteria Weights")
st.dataframe(weights_df)

st.subheader("Step 3: Normalized Decision Matrix")
st.dataframe(normalized_df)

st.subheader("Step 4: Weighted Normalized Matrix")
st.dataframe(weighted_norm_df)

st.subheader("Step 5: Final WASPAS Scores and Ranking")
st.dataframe(result_df)

st.subheader("Step 6: Ranking Bar Chart")
st.bar_chart(result_df.set_index('Alternative')['WASPAS Score'])

# Option to download result
csv = result_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Final Results as CSV",
    data=csv,
    file_name='waspas_results.csv',
    mime='text/csv',
)
