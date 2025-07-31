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

# Step 2: Normalize Data using vector normalization
def normalize_data(df):
    normalized_df = df.copy()
    for col in df.columns[1:]:
        normalized_df[col] = df[col] / np.sqrt((df[col]**2).sum())
    return normalized_df

normalized_df = normalize_data(df)

# Step 3: Assign weights to each criterion
weights = {
    'Engagement': 0.15,
    'Retention Rate': 0.20,
    'Cognitive Load': 0.10,
    'Application in Context': 0.15,
    'Accessibility': 0.20,
    'Feedback Mechanism': 0.20
}

# Step 4: Weighted Normalization
def apply_weights(normalized_df, weights):
    for col in df.columns[1:]:
        normalized_df[col] = normalized_df[col] * weights[col]
    return normalized_df

weighted_df = apply_weights(normalized_df, weights)

# Step 5: Calculate Ideal and Anti-Ideal Solutions (PIS & NIS)
def ideal_anti_ideal(weighted_df):
    pis = weighted_df.max(axis=0)  # Ideal Solution (PIS)
    nis = weighted_df.min(axis=0)  # Anti-Ideal Solution (NIS)
    return pis, nis

pis, nis = ideal_anti_ideal(weighted_df)

# Step 6: Calculate Euclidean distance to PIS and NIS
def calculate_distances(weighted_df, pis, nis):
    # Broadcasting PIS and NIS to match the DataFrame's shape
    pis_distance = np.sqrt(((weighted_df - pis) ** 2).sum(axis=1))
    nis_distance = np.sqrt(((weighted_df - nis) ** 2).sum(axis=1))
    return pis_distance, nis_distance

pis_distance, nis_distance = calculate_distances(weighted_df, pis, nis)

# Step 7: Calculate Relative Closeness (Pi)
def calculate_closeness(pis_distance, nis_distance):
    closeness = nis_distance / (pis_distance + nis_distance)
    return closeness

closeness = calculate_closeness(pis_distance, nis_distance)

# Step 8: Rank the Alternatives based on Relative Closeness (Pi)
def rank_alternatives(closeness):
    return pd.DataFrame({
        'Educational Strategy': df['Educational Strategy'],
        'Closeness (Pi)': closeness,
        'Rank': closeness.rank(ascending=False)
    }).sort_values(by='Rank')

ranked_df = rank_alternatives(closeness)

# Streamlit UI
st.title('SmartRankEDU 360: Educational Strategy Evaluation Using WASPAS')

st.write("This application evaluates educational strategies based on various criteria using the WASPAS method.")

# Display Data
st.subheader('Step 1: Educational Strategies Data')
st.dataframe(df)

# Step 2: Display Normalized Data
st.subheader('Step 2: Normalized Data')
st.dataframe(normalized_df)

# Step 3: Display Weighted Normalized Data
st.subheader('Step 4: Weighted Normalized Data')
st.dataframe(weighted_df)

# Step 5: Display Ideal (PIS) and Anti-Ideal (NIS) Solutions
st.subheader('Step 5: Ideal (PIS) and Anti-Ideal (NIS) Solutions')
st.write(f"PIS (Ideal Solution): \n{pis}")
st.write(f"NIS (Anti-Ideal Solution): \n{nis}")

# Step 6: Display Euclidean Distances to PIS and NIS
st.subheader('Step 6: Euclidean Distances to PIS and NIS')
st.write(f"PIS Distance: \n{pis_distance}")
st.write(f"NIS Distance: \n{nis_distance}")

# Step 7: Display Relative Closeness (Pi)
st.subheader('Step 7: Relative Closeness (Pi)')
st.write(closeness)

# Step 8: Display Rankings
st.subheader('Step 8: Rankings Based on WASPAS Method')
st.dataframe(ranked_df)

# Plot Rankings
st.subheader('Step 8: Ranking Chart')
st.bar_chart(ranked_df.set_index('Educational Strategy')['Rank'])

# Running the app:
if __name__ == '__main__':
    st.write("Welcome to the SmartRankEDU 360 App!")
