import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

# Load the data from Excel
primary_data_path = 'election_data.xlsx'  # Update with the actual path to your Excel file
general_election_data_path = 'election_data.xlsx'  # Update with the actual path to your Excel file
primary_data = pd.read_excel(primary_data_path, sheet_name='PrimaryData')
general_election_data = pd.read_excel(general_election_data_path, sheet_name='GeneralElection')

# Define battleground states
battleground_states = [
    "Arizona", "Florida", "Georgia", "Michigan", "New Hampshire",
    "North Carolina", "Ohio", "Pennsylvania", "Wisconsin",
    "Colorado", "Iowa", "Minnesota", "Nevada", "Virginia"
]

# Merge primary data with general election data for historical analysis
merged_data = pd.merge(primary_data, general_election_data[['State', 'Year', 'Democrat(%)', 'Republican(%)']], on=['State', 'Year'], how='left')

# Feature Engineering for all data
merged_data['Primary_Vote_Percentage_Diff'] = merged_data['Democrat_Primary (%)'] - merged_data['Republican_Primary (%)']

# Clean data by removing rows with NaN values in 'Democrat(%)' and 'Republican(%)'
merged_data_clean = merged_data.dropna(subset=['Democrat(%)', 'Republican(%)'])

# Model evaluation for historical data (excluding 2024)
years = merged_data_clean['Year'].unique()
test_accuracies = {}
for test_year in years:
    if test_year != 2024:  # Exclude 2024
        train_data = merged_data_clean[merged_data_clean['Year'] != test_year]
        test_data = merged_data_clean[merged_data_clean['Year'] == test_year]
        
        X_train = train_data[['Primary_Vote_Percentage_Diff']]
        y_train = train_data['Democrat(%)'] - train_data['Republican(%)']
        X_test = test_data[['Primary_Vote_Percentage_Diff']]
        y_test = test_data['Democrat(%)'] - test_data['Republican(%)']
        
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        predictions_binary = [1 if pred > 0 else -1 for pred in predictions]
        y_test_binary = [1 if val > 0 else -1 for val in y_test]
        accuracy = accuracy_score(y_test_binary, predictions_binary)
        test_accuracies[test_year] = accuracy

# Print accuracy for each historical year
for year, acc in test_accuracies.items():
    print(f"Year: {year}, Accuracy: {acc}")

# Prediction for 2024 using engineered features
predict_data_2024 = primary_data[(primary_data['Year'] == 2024) & (primary_data['State'].isin(battleground_states))].copy()
predict_data_2024['Primary_Vote_Percentage_Diff'] = predict_data_2024['Democrat_Primary (%)'] - predict_data_2024['Republican_Primary (%)']

X_predict_2024 = predict_data_2024[['Primary_Vote_Percentage_Diff']]
predictions_2024 = model.predict(X_predict_2024)
predict_data_2024['Projected_Difference'] = predictions_2024
predict_data_2024['Projected_Winner'] = ['D' if pred > 0 else 'R' for pred in predictions_2024]

# Displaying projected winners for each battleground state in 2024, along with projected difference
print(predict_data_2024[['State', 'Projected_Winner', 'Projected_Difference']])
