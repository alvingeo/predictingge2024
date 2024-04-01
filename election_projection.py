import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def load_primary_data(year):
    file_path = 'election_data.xlsx'
    primary_data = pd.read_excel(file_path, sheet_name=str(year))
    return primary_data

def load_winners_data():
    file_path = 'election_data.xlsx'
    winners_data = pd.read_excel(file_path, sheet_name='Winners')
    return winners_data

def preprocess_data(data, label_encoder=None):
    data['DVote'] = data['DVote'].astype(str).str.replace(',', '').astype(float)
    data['DPercentage'] = data['DPercentage'].astype(str).str.replace(',', '.').astype(float)
    data['RVote'] = data['RVote'].astype(str).str.replace(',', '').astype(float)
    data['RPercentage'] = data['RPercentage'].astype(str).str.replace(',', '.').astype(float)
    
    if label_encoder is None:
        label_encoder = LabelEncoder()
    data['State'] = label_encoder.fit_transform(data['State'])
    return data, label_encoder

# Initialize LabelEncoder to be used for all years
label_encoder = LabelEncoder()

# Load and preprocess all primary data and winners data
all_primary_data = []
winners_data = load_winners_data()
for year in [2012, 2016, 2020]:
    data = load_primary_data(year)
    data.dropna(subset=['DPercentage', 'RPercentage'], inplace=True)
    data, label_encoder = preprocess_data(data, label_encoder)
    data['Year'] = year  # Add year column to distinguish different years
    # Add the winner data
    data['Winner'] = data['Year'].apply(lambda x: winners_data.loc[winners_data['Year'] == x, 'Winner'].iloc[0])
    all_primary_data.append(data)

# Combine all years into one DataFrame
all_data = pd.concat(all_primary_data, ignore_index=True)

# Prepare data for model training
X = all_data[['DVote', 'DPercentage', 'RVote', 'RPercentage', 'State']]
y = all_data['Winner']

# Train RandomForestClassifier on all data from 2012 to 2020
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Load primary data for 2024 and preprocess it
primary_data_2024 = load_primary_data(2024)
primary_data_2024, _ = preprocess_data(primary_data_2024, label_encoder)

# Predict the winner for the year 2024 using the trained model
predictions_2024 = model.predict(primary_data_2024[['DVote', 'DPercentage', 'RVote', 'RPercentage', 'State']])

# Determine the overall winner for 2024 based on the most common prediction
from collections import Counter
predicted_winner_2024 = Counter(predictions_2024).most_common(1)[0][0]

# Print the predicted winner for 2024
print(f"Predicted Winner for 2024: {'Democrat' if predicted_winner_2024 == 'D' else 'Republican'}")
