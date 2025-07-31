import streamlit as st
import pandas as pd
import numpy as np

# Sample Data: Replace this with your Big Data processing or loading mechanism.
data = {
    'Educational Strategy': ['Traditional Learning', 'Gamification', 'AI Tools', 'Mobile Learning'],
    'Engagement': [0.80, 0.92, 0.75, 0.85],
    'Retention Rate': [0.75, 0.90, 0.72, 0.80],
    'Cognitive Load': [0.60, 0.50, 0.70, 0.65],
    'Application in Context': [0.70, 0.85, 0.78, 0.80],
    'Accessibility': [0.85, 0.80, 0.90, 0.88],
    'Feedback Mechanism': [0.90, 0.85, 0.80, 0.82]
}

# Convert the data to DataFrame
df = pd.DataFrame(data)

# WASPAS calculation (simplified)
def calculate_waspas(df):
    # Normalize Data (vector normalization)
    normalized_df = df.copy()
    for col in df.columns[1:]:
        normalized_df[col] = df[col] / np.sqrt((df[col]**2).sum())
    
    # Weight for each criterion (just for example)
    weights = {
        'Engagement': 0.15,
        'Retention Rate': 0.20,
        'Cognitive Load': 0.10,
        'Application in Context': 0.15,
        'Accessibility': 0.20,
        'Feedback Mechanism': 0.20
    }

    # Weighted Normalized Matrix
    for col in df.columns[1:]:
        normalized_df[col] = normalized_df[col] * weights[col]
    
    # Calculate Total Score (for each strategy)
    normalized_df['Total Score'] = normalized_df.iloc[:, 1:].sum(axis=1)

    # Rank strategies based on Total Score
    normalized_df['Rank'] = normalized_df['Total Score'].rank(ascending=False)

    return normalized_df[['Educational Strategy', 'Total Score', 'Rank']]

# Streamlit UI
st.title('SmartRankEDU 360: Educational Strategy Evaluation Using WASPAS')

st.write("This application evaluates educational strategies based on various criteria using the WASPAS method.")

# Display Data
st.subheader('Educational Strategies Data')
st.dataframe(df)

# Display Calculated Rankings
st.subheader('Rankings Based on WASPAS Method')
ranked_df = calculate_waspas(df)
st.dataframe(ranked_df)

# Plot Rankings
st.subheader('Ranking Chart')
st.bar_chart(ranked_df.set_index('Educational Strategy')['Rank'])

# Running the app:
if __name__ == '__main__':
    st.write("Welcome to the SmartRankEDU 360 App!")
