import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils import resample
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)  # Suppress specific warnings

# Load the combined data
matches = pd.read_csv("matches_combined.csv")

# Convert date datatype from object to numeric value
matches["Date"] = pd.to_datetime(matches['Date'], errors='coerce')
matches["Fixture_Date"] = pd.to_datetime(matches['Fixture_Date'], errors='coerce')

# Merge 'Date' and 'Fixture_Date' columns
matches['Date'] = matches['Date'].combine_first(matches['Fixture_Date'])

# Drop rows where 'Date' is still missing
matches = matches.dropna(subset=['Date'])

# Convert home/away information into 0 or 1
matches['Venue_code'] = matches["Venue"].astype("category").cat.codes

# Convert opponent into numeric value so that system can read
matches['Opp_code'] = matches["Opponent"].astype("category").cat.codes

# Convert time of game to only look into hour and store it as int
matches['Hour'] = matches["Time"].str.replace(":.+", "", regex=True).astype("int")

# Convert day of week into numeric value
matches['Day_code'] = matches["Date"].dt.dayofweek

# Convert result to numeric value. 2 for W, 1 for D, 0 for L
matches['Target'] = matches['Result'].map({'W': 2, 'D': 1, 'L': 0})

# Add more features: last 5 games form
matches['Points'] = matches['Target'].map({2: 3, 1: 1, 0: 0})
matches['Form'] = matches.groupby('Team')['Points'].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)

# Separate historical and future data
historical_matches = matches[matches['Date'] < '2024-07-01']
future_matches = matches[matches['Date'] >= '2024-07-01']

# Drop rows where 'Target' is NaN in historical data
historical_matches = historical_matches.dropna(subset=['Target'])

# Ensure future matches do not have 'Target' values
future_matches = future_matches.copy()  # Avoid SettingWithCopyWarning
future_matches.loc[:, 'Target'] = np.nan

# Handle class imbalance by oversampling the minority classes
train = historical_matches

train_majority = train[train.Target == 0]
train_draw = train[train.Target == 1]
train_win = train[train.Target == 2]

train_draw_upsampled = resample(train_draw, 
                                replace=True,     # sample with replacement
                                n_samples=len(train_majority),    # to match majority class
                                random_state=123) # reproducible results

train_win_upsampled = resample(train_win, 
                               replace=True,     # sample with replacement
                               n_samples=len(train_majority),    # to match majority class
                               random_state=123) # reproducible results

train_upsampled = pd.concat([train_majority, train_draw_upsampled, train_win_upsampled])

# Define predictors
predictors = ['Venue_code', 'Opp_code', 'Hour', 'Day_code', 'Form']

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [4, 6, 8],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5)
grid_search.fit(train_upsampled[predictors], train_upsampled['Target'])
best_params = grid_search.best_params_

rf = RandomForestClassifier(**best_params)
rf.fit(train_upsampled[predictors], train_upsampled['Target'])

# Perform cross-validation on the training data
cv_scores = cross_val_score(rf, train_upsampled[predictors], train_upsampled['Target'], cv=5)
print(f'Cross-Validation Scores: {cv_scores}')

def make_predictions(data, predictors):
    train = data[data['Date'] < '2024-07-01']
    test = data[(data['Date'] >= '2024-01-01') & (data['Date'] < '2024-07-01')]  # Use latest historical data for testing
    rf.fit(train[predictors], train['Target'])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test['Target'], predicted=preds), index=test.index)
    precision = precision_score(test['Target'], preds, average='weighted', zero_division=0)
    accuracy = accuracy_score(test['Target'], preds)
    return combined, precision, accuracy

combined, precision, final_acc = make_predictions(historical_matches, predictors)
combined = combined.merge(historical_matches[['Date', 'Team', 'Opponent', 'Result']], left_index=True, right_index=True)

# Combine home and away data
class MissingDict(dict):
    __missing__ = lambda self, key: key

# Creates a map so that full name, which is listed in the team column, matches shortened name, which is listed in Opponent column
map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "Wolverhampton Wanderers": "Wolves"
}

mapping = MissingDict(**map_values)
combined["New_team"] = combined['Team'].map(mapping)
merged = combined.merge(combined, left_on=['Date', 'New_team'], right_on=['Date', 'Opponent'])

# Print the value counts for specific conditions
print(merged[(merged["predicted_x"] == 2) & (merged['predicted_y'] == 0)]['actual_x'].value_counts())

# Final assessment
print(f'Final Accuracy: {final_acc}')
print(f'Final Precision: {precision}')

# Predict future matches for the 24-25 season

# Ensure the future matches data has the same predictors as the training data
future_predictors = predictors

# Filter for the first week of future matches
first_week_start = future_matches['Date'].min()
first_week_end = first_week_start + pd.Timedelta(days=7)
first_week_matches = future_matches[(future_matches['Date'] >= first_week_start) & (future_matches['Date'] <= first_week_end)].copy()

# Make predictions using the trained model
future_preds = rf.predict(first_week_matches[future_predictors])

# Add predictions to the future matches DataFrame
first_week_matches['Prediction'] = future_preds

# Ensure the 'Prediction' column is of type int before mapping
first_week_matches['Prediction'] = first_week_matches['Prediction'].astype(int)
first_week_matches['Prediction'] = first_week_matches['Prediction'].map({2: 'Win', 1: 'Draw', 0: 'Loss'})

# Display the predictions
print(first_week_matches[['Date', 'Team', 'Opponent', 'Prediction']])