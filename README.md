# House Price Prediction Project

## Overview

This project aims to predict house prices in Bengaluru, India, using machine learning techniques. We primarily focused on regression models, including Linear Regression, Ridge Regression, Lasso Regression, and Elastic Net. The objective was to develop a model that can accurately estimate house prices based on various features such as location, size, and amenities.

## Project Structure

The project is organized as follows:

- **[1 - Introduction](#Introduction):** Provides an overview of the project, problem statement, and dataset description.
- **[2 - Import Libraries](#Import-Libraries):** Lists the Python libraries used in the project.
- **[3 - Data Loading and Exploration](#Data-Loading-and-Exploration):** Covers the steps to load and explore the dataset, including summary statistics and basic information.
- **[4 - Data Preprocessing](#Data-Preprocessing):** Describes the steps taken to clean and prepare the data, including handling missing values, encoding categorical variables, and feature scaling.
- **[5 - Data Splitting](#Data-Splitting):** Details the process of splitting the data into training, validation, and test sets.
- **[6 - Model Definition and Training](#Model-Definition):** Discusses the models used in the project and their training process.
- **[7 - Model Evaluation](#Model-Evaluation):** Evaluates the performance of the models using key metrics and visualizations.
- **[8 - Hypertuning of Model](#Hypertuning-of-Model):** Describes the process of fine-tuning the hyperparameters of the models.
- **[9 - Conclusion](#Conclusion):** Summarizes the findings and suggests potential areas for further improvement.

## Installation

To run this project locally, you need to have Python installed. Follow the steps below to set up the environment:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/leondmtrv/house-price-prediction.git
    cd house-price-prediction
    ```

2. **Create and activate a virtual environment:**

     ```bash
     python3 -m venv env
     source env/bin/activate  # On Windows use `env\Scripts\activate`
     ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook HousePrediction.ipynb
    ```

## Usage

After setting up the environment, you can explore the project by running the Jupyter Notebook HousePrediction.ipynb. The notebook walks you through each step of the project, from data preprocessing to model evaluation.

Key Components:
- Data Preprocessing: This section includes steps like handling missing values, encoding categorical features, and scaling numerical features.
- Model Training: Various regression models are trained on the processed dataset.
- Model Evaluation: The performance of the models is evaluated using metrics like Mean Squared Error (MSE) and R² score. Visualizations such as Actual vs. Predicted plots and Residual plots are also included.

## Results

The Ridge Regression model was identified as the best-performing model based on the test set MSE. However, further fine-tuning did not significantly improve the model's performance, suggesting that the default settings were near-optimal.

## Future Work

To further enhance the predictive power of the model, consider the following:

- Explore Non-Linear Models: Such as Random Forest or Gradient Boosting.
- Feature Engineering: Introduce new features or transform existing ones to capture non-linear relationships.
- Outlier Detection: Implement methods to detect and remove outliers to improve model robustness.
- Ensemble Methods: Use ensemble techniques like bagging or stacking to improve prediction accuracy.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

The dataset used in this project was sourced from Kaggle. Special thanks to the data providers and the open-source community for the tools and libraries that made this project possible.
