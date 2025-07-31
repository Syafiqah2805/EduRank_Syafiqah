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
    # Step 2: Normalize Data (vector normalization)
    normalized_df = df.copy()
    for col in df.columns[1:]:
        normalized_df[col] = df[col] / np.sqrt((df[col]**2).sum())
    
    # Display Normalized Data
    st.subheader('Step 2: Normalized Data')
    st.dataframe(normalized_df)

    # Step 3: Weight for each criterion (just for example)
    weights = {
        'Engagement': 0.15,
        'Retention Rate': 0.20,
        'Cognitive Load': 0.10,
        'Application in Context': 0.15,
        'Accessibility': 0.20,
        'Feedback Mechanism': 0.20
    }

    # Step 4: Weighted Normalized Matrix
    for col in df.columns[1:]:
        normalized_df[col] = normalized_df[col] * weights[col]
    
    # Display Weighted Normalized Data
    st.subheader('Step 4: Weighted Normalized Data')
    st.dataframe(normalized_df)
    
    # Step 5: Calculate Ideal and Anti-Ideal Solutions (PIS & NIS)
    pis = normalized_df.max(axis=0)  # Ideal Solution (PIS)
    nis = normalized_df.min(axis=0)  # Anti-Ideal Solution (NIS)

    # Display Ideal and Anti-Ideal Solutions
    st.subheader('Step 5: Ideal (PIS) and Anti-Ideal (NIS) Solutions')
    st.write(f"PIS: \n{pis}\n\nNIS: \n{nis}")

    # Step 6: Calculate Euclidean distance to PIS and NIS
    pis_distance = np.sqrt(((normalized_df - pis) ** 2).sum(axis=1))
    nis_distance = np.sqrt(((normalized_df - nis) ** 2).sum(axis=1))

    # Display Euclidean Distances
    st.subheader('Step 6: Euclidean Distances to PIS and NIS')
    st.write(f"PIS Distance: \n{pis_distance}\n\nNIS Distance: \n{nis_distance}")

    # Step 7: Calculate Relative Closeness (Pi)
    closeness = nis_distance / (pis_distance + nis_distance)

    # Display Relative Closeness (Pi)
    st.subheader('Step 7: Relative Closeness (Pi)')
    st.write(closeness)

    # Step 8: Rank the Alternatives based on Relative Closeness (Pi)
    ranked_df = pd.DataFrame({
        'Educational Strategy': df['Educational Strategy'],
        'Closeness (Pi)': closeness,
        'Rank': closeness.rank(ascending=False)
    }).sort_values(by='Rank')

    return ranked_df, normalized_df, pis, nis, pis_distance, nis_distance, closeness

# Streamlit UI
st.title('SmartRankEDU 360: Educational Strategy Evaluation Using WASPAS')

st.write("This application evaluates educational strategies based on various criteria using the WASPAS method.")

# Display Data
st.subheader('Step 1: Educational Strategies Data')
st.dataframe(df)

# Calculate and Display the Step-by-Step Process
ranked_df, normalized_df, pis, nis, pis_distance, nis_distance, closeness = calculate_waspas(df)

# Display Calculated Rankings
st.subheader('Step 8: Rankings Based on WASPAS Method')
st.dataframe(ranked_df)

# Plot Rankings
st.subheader('Step 8: Ranking Chart')
st.bar_chart(ranked_df.set_index('Educational Strategy')['Rank'])

# Running the app:
if __name__ == '__main__':
    st.write("Welcome to the SmartRankEDU 360 App!")
