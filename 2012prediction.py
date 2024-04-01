import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Function to load primary data from Excel sheet for a specific year
def load_primary_data(year):
    file_path = 'election_data.xlsx'
    primary_data = pd.read_excel(file_path, sheet_name=str(year))
    return primary_data

# Function to load winners data from Excel sheet
def load_winners_data():
    file_path = 'election_data.xlsx'
    winners_data = pd.read_excel(file_path, sheet_name='Winners')
    return winners_data

# Main function to analyze election data for a specific year
def analyze_election_data(year):
    primary_data = load_primary_data(year)
    winners_data = load_winners_data()

    # Drop rows with NaN values
    primary_data.dropna(subset=['DPercentage', 'RPercentage'], inplace=True)

    # Prepare data for model training
    X = primary_data[['DVote', 'DPercentage', 'RVote', 'RPercentage', 'State']]

    # Encode state names using LabelEncoder
    label_encoder = LabelEncoder()
    #X['State'] = label_encoder.fit_transform(X['State'])
    X.loc[:, 'State'] = label_encoder.fit_transform(X['State'])


    # Create target variable y for the specific year
    y = winners_data.loc[winners_data['Year'] == year, 'Winner'].iloc[0]

    # Train RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, [y] * len(X))

    # Print the trained model's accuracy
    accuracy = model.score(X, [y] * len(X))
    print(f"Model accuracy for {year}: {accuracy}")

    # Return the trained model
    return model

# Train models for the years 2012, 2016, and 2020
models = {}
for year in [2012, 2016, 2020]:
    models[year] = analyze_election_data(year)

# Load primary data for 2024 and handle formatting issues
primary_data_2024 = load_primary_data(2024)
primary_data_2024['DVote'] = primary_data_2024['DVote'].astype(str).str.replace(',', '').astype(float)
primary_data_2024['DPercentage'] = primary_data_2024['DPercentage'].astype(str).str.replace(',', '.').astype(float)
primary_data_2024['RVote'] = primary_data_2024['RVote'].astype(str).str.replace(',', '').astype(float)
primary_data_2024['RPercentage'] = primary_data_2024['RPercentage'].astype(str).str.replace(',', '.').astype(float)

# Encode state names for 2024 data
label_encoder = LabelEncoder()

primary_data_2024['State'] = label_encoder.fit_transform(primary_data_2024['State'])

# Predict the winner for the year 2024 using the trained models
predictions_2024 = {}
for year in [2012, 2016, 2020]:
    model = models[year]
    predictions_2024[year] = model.predict(primary_data_2024[['DVote', 'DPercentage', 'RVote', 'RPercentage', 'State']])

# Determine the overall winner for 2024 based on the most common prediction
predicted_winner_2024 = max(set(tuple(predictions_2024[year]) for year in [2012, 2016, 2020]), key=list(tuple(predictions_2024[2012])).count)

# Print the predicted winner for 2024
print(f"Predicted Winner for 2024: {'Democrat' if predicted_winner_2024 == 'D' else 'Republican'}")
