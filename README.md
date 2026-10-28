# Rainfall Prediction and Seasonal Climate Pattern Analysis

## 1. Project Overview

Rainfall Prediction and Seasonal Climate Pattern Analysis is a machine learning project that uses historical monthly rainfall data to estimate rainfall for a selected district, year, and month.

The project also analyzes seasonal rainfall patterns across districts and presents the results through an interactive web application.

## 2. Objectives

- Analyze historical monthly rainfall patterns.
- Train a machine learning model to estimate rainfall.
- Evaluate model performance using standard regression metrics.
- Visualize average monthly rainfall across districts.
- Provide an interactive interface for making predictions.

## 3. Technologies Used

- **Python** — Core programming language.
- **Pandas** — Data loading, processing, and analysis.
- **Scikit-learn** — Machine learning model, preprocessing, and evaluation.
- **Random Forest Regressor** — Rainfall prediction algorithm.
- **Joblib** — Saving and loading the trained model.
- **Streamlit** — Interactive web application.
- **Matplotlib** — Data visualization.

## 4. Dataset

The project uses historical monthly climate data organized by district.

- Number of district files combined: 724
- Total records: 382,272
- Historical period: 1981–2024
- Data frequency: Monthly

The combined dataset contains these columns:

| Column | Description |
|---|---|
| `district` | District identifier |
| `year` | Observation year |
| `month` | Month number |
| `rainfall` | Recorded rainfall value |
| `max_temp` | Maximum temperature |
| `min_temp` | Minimum temperature |
| `avg_temp` | Average temperature |

**Note:** Confirm the original dataset's documentation for the rainfall measurement unit and the meaning of its district identifiers before presenting these details as verified facts.

## 5. Machine Learning Methodology

### Input Features

- District identifier
- Year
- Month

### Target Variable

- Rainfall

### Data Splitting

- Training period: 1981–2018
- Testing period: 2019–2024

### Data Preprocessing

The district identifier is converted into categorical features using `OneHotEncoder`. A Scikit-learn pipeline combines preprocessing with the prediction model.

### Algorithm

A `RandomForestRegressor` is trained using 50 decision trees, with additional settings to control tree depth and minimum leaf size.

## 6. Model Evaluation

The model was evaluated on records from 2019–2024.

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 66.2021 |
| Root Mean Squared Error (RMSE) | 108.6860 |
| R² Score | 0.5449 |

The R² score indicates moderate predictive performance on the test dataset. Results should be interpreted in the context of the dataset's rainfall units and distribution.

## 7. Application Features

- Select a district.
- Choose a year and month.
- Generate a rainfall estimate.
- View average monthly rainfall patterns.
- Identify the wettest and driest months in the dataset.
- Inspect sample dataset records.

## 8. Project Structure

```text
Rainfall-Prediction-Climate-Analysis/
│
├── Data/
│   ├── Raw/
│   │   └── Monthly/
│   └── Processed/
│       └── rainfall_combined.csv
│
├── Models/
│   └── rainfall_model.pkl
│
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

The raw district files and processed dataset may need to be obtained separately if they are not included with the project.

## 9. Installation and Execution

### Step 1: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python train_model.py
```

This trains the model, evaluates its performance, and saves it to `Models/rainfall_model.pkl`.

### Step 3: Launch the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal to use the application.

## 10. Limitations

- The model uses district, year, and month rather than live weather observations.
- Rainfall depends on many atmospheric and geographical factors not included in the model.
- Predictions for years beyond the training period may be unreliable.
- The application provides model-based estimates, not official weather forecasts.
- Model performance may vary across districts, seasons, and years.

## 11. Future Improvements

- Add more relevant weather and atmospheric features.
- Compare multiple regression algorithms.
- Evaluate performance separately by district and season.
- Improve prediction accuracy through feature engineering and model tuning.
- Deploy the application online.

## 12. Conclusion

This project demonstrates an end-to-end machine learning workflow, including data preparation, preprocessing, model training, evaluation, model persistence, and interactive visualization using Streamlit.
