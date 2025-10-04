# End-to-End Machine Learning Project: California Housing Price Prediction

![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)

### Project Description

This repository contains the code for an end-to-end machine learning project that predicts median housing prices in California districts. The goal is to follow a realistic data science workflow, from data ingestion and exploration to model training, tuning, and final evaluation.

This project is based on the example from Chapter 2 of Aurélien Géron's book, "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow."

---

## Project Overview

This project covers the following key machine learning concepts and steps:

* **Data Ingestion:** Fetching data and loading it into a Pandas DataFrame.
* **Data Exploration & Visualization:** Using Matplotlib to understand data distributions, correlations, and geographical patterns.
* **Data Cleaning & Preparation:** Building a full data preparation pipeline using Scikit-Learn's `Pipeline` and `ColumnTransformer` to handle missing values, categorical attributes, and feature scaling.
* **Feature Engineering:** Creating new, more predictive features from the existing data.
* **Model Training:** Training several regression models, including Linear Regression, Decision Trees, and Random Forests.
* **Model Evaluation:** Using K-fold cross-validation for robust performance measurement.
* **Hyperparameter Tuning:** Using `RandomizedSearchCV` to find the best combination of hyperparameters for the final model.
* **Final Evaluation:** Evaluating the final tuned model on the unseen test set to estimate its real-world performance.

---

## Repository Structure

This repository contains two distinct Jupyter notebooks that accomplish the same goal but serve different purposes:

1.  `1_Exploratory_California_Housing.ipynb`
    * **Purpose:** This is the detailed, step-by-step "learning" notebook. It follows our entire development process, including detailed explanations, cell-by-cell code execution, visualizations, and interpretation of intermediate results. It's designed to be read like a tutorial.

2.  `2_Consolidated_California_Housing.ipynb`
    * **Purpose:** This is the final, clean, and consolidated script. It combines all the necessary steps—from data loading to final evaluation—into a single, linear, and runnable file. This represents a more "production-style" script, free of exploratory code.

---

## Getting Started

No local installation is required. You can run both notebooks directly in your browser using Google Colab.

* **Run the Exploratory Notebook:**
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ItxJack/California-Housing-Price-Prediction/blob/main/1_Exploratory_California_Housing.ipynb)

* **Run the Consolidated Script:**
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ItxJack/California-Housing-Price-Prediction/blob/main/2_Consolidated_California_Housing.ipynb)

---

## Dataset

The project uses the **California Housing Prices** dataset, which is based on data from the 1990 California census. Key features include median income, housing median age, total rooms, population, latitude, and longitude.

---

## Model & Final Results

* **Final Model:** A fine-tuned `RandomForestRegressor`.
* **Performance Metric:** Root Mean Square Error (RMSE).
* **Final RMSE on Test Set:** Approximately **$47,730**. This means the model's typical prediction error is about $47,730.

---

## Technologies Used

* **Python**
* **Pandas** for data manipulation and analysis.
* **NumPy** for numerical operations.
* **Scikit-Learn** for data preprocessing, modeling, and evaluation.
* **Matplotlib** for data visualization.
* **XGBoost** as an example of an advanced model.
* **Google Colab / Jupyter Notebook** for the interactive environment.

---

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## Acknowledgements

* This project is a practical implementation of the concepts taught in Aurélien Géron's "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow."
