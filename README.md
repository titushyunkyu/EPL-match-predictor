# EPL Match Predictor

This project is an English Premier League (EPL) match predictor that uses historical match data to predict the outcomes of future matches. It employs machine learning techniques, specifically a Random Forest Classifier, to predict whether a team will win, draw, or lose based on various features.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Model Details](#model-details)
- [Contact](#contact)

## Introduction

The EPL Match Predictor uses historical data to predict the outcomes of upcoming EPL matches. The project includes web scraping scripts to gather the latest match data, preprocess the data, train a machine learning model, and make predictions for future matches.
![image](https://github.com/user-attachments/assets/03cfb6fe-1d00-46cb-89d2-3841629ef5c5)


## Features

- Scrapes historical and future EPL match data from the web.
- Preprocesses and cleans the data for model training.
- Uses a Random Forest Classifier with hyperparameter tuning for predictions.
- Provides accuracy and precision metrics for the model.
- Allows for interactive predictions for specific match weeks.
- Handles class imbalance by oversampling the minority classes.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/epl-match-predictor.git
   cd epl-match-predictor
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
## Usage
1. Scrape and Prepare Data:
   Run the scraper to gather and prepare the match data (you can adjust the range of seasons to retrieve data).
   ```bash
   python scraper.py
2. Train the Model and Make Predictions:
   Run the machine learning script to train the model and predict future matches.
   ```bash
   python ml.py
3. Interactive Predictions:
   During the execution of ml.py, you can interactively enter the week number to get predictions for that specific week.
## Files
- scraper.py: Script for web scraping and preparing match data.
- ml.py: Script for training the model, making predictions, and providing interactive predictions.
- matches_combined.csv: The combined dataset of historical and future matches (current data is from 19-20 season ~ 22-23 season. If you would want data from a different range, adjust line 50 in scraper.py and run the code).
- requirements.txt: List of dependencies required to run the project.
## Model Details

### Data Processing
- **Date Handling**: 
  - Merges `Date` and `Fixture_Date` columns.
  - Handles missing values by dropping rows where the date is still missing after the merge.

- **Feature Engineering**: 
  - Converts categorical features to numeric codes:
    - `Venue_code`: Encodes the venue of the match.
    - `Opp_code`: Encodes the opponent team.
    - `Hour`: Extracts and encodes the hour at which the match is played.
    - `Day_code`: Encodes the day of the week the match is played.
  - Adds new feature:
    - `Form`: The average points from the last 5 games, representing the recent performance of the team.

- **Class Imbalance Handling**: 
  - Uses oversampling to balance the number of instances for win, draw, and loss classes.
  - Minority classes (win and draw) are oversampled to match the number of instances of the majority class (loss).

### Predictors
- **Venue_code**: Encodes the venue of the match.
- **Opp_code**: Encodes the opponent team.
- **Hour**: The hour at which the match is played.
- **Day_code**: The day of the week the match is played.
- **Form**: The average points from the last 5 games.

### Model Training
- **Algorithm**: 
  - Uses a Random Forest Classifier, which is an ensemble learning method that operates by constructing a multitude of decision trees during training.
  - The output of the Random Forest is the mode of the classes (classification) or the mean prediction (regression) of the individual trees.

- **Hyperparameter Tuning**: 
  - Utilizes Grid Search with cross-validation to find the best combination of hyperparameters.
  - Grid Search explores a specified parameter grid to optimize model performance.
  - Cross-validation ensures that the model's performance is validated on different subsets of the training data, which helps in preventing overfitting.

- **Evaluation**: 
  - Provides accuracy and precision metrics to evaluate model performance.
  - Accuracy is the proportion of true results (both true positives and true negatives) among the total number of cases examined.
  - Precision is the proportion of true positives among the total number of positive predictions made.

### Prediction
- **Future Matches**: 
  - Predicts the outcomes for future matches based on the trained model.
  - Ensures future matches data has the same predictors as the training data.
  - Allows filtering and prediction for specific weeks.

- **Interactive**: 
  - Allows users to input a week number to get predictions for that specific week.
 
## Contact
For any questions or suggestions, please open an issue or contact me at hvm4sg@virginia.edu
