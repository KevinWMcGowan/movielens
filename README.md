# 🎥 MovieLens Rating Prediction Project

Welcome to the **MovieLens Rating Prediction Project**, an in-depth analysis and predictive modeling initiative leveraging the MovieLens 10M dataset. This repository includes an R Markdown report (`Movielens_report.Rmd`) and a companion R script (`Movielens_analysis.R`) to explore, process, and model movie ratings.

## 📚 Overview

The goal of this project is to develop a machine learning model capable of predicting movie ratings based on user behavior and movie characteristics. By achieving accurate predictions, this model serves as the foundational step for personalized movie recommendation systems used by platforms like Netflix and Hulu.

### Highlights:
- Utilized **Gaussian Elastic Net Regression** to model continuous ratings.
- Engineered features from raw data, such as movie release decade, user rating trends, and genre distributions.
- Evaluated model performance using **Root Mean Squared Error (RMSE)**.

## 📂 Repository Structure
- **movielens/**
  - **reports/**: Directory for report outputs
    - `Movielens_report.Rmd`: R Markdown file for the detailed report
    - `Movielens_report.pdf`: PDF version of the R Markdown report (to be added)
  - **scripts/**: Contains the R script
    - `Movielens_analysis.R`: Script for all analysis and modeling
## 📊 Dataset Information

The dataset used is the [MovieLens 10M Dataset](https://files.grouplens.org/datasets/movielens/ml-10m.zip). It includes:
- **10 million ratings** from **72,000 users** on **10,000 movies**.
- **Key fields**: `userId`, `movieId`, `rating`, `timestamp`, `title`, and `genres`.

Since the dataset exceeds GitHub's file size limit, it is not included in this repository. The analysis script and report automatically download and process the dataset.

## 🚀 How to Run

1. Clone the repository:
   ```bash
    git clone https://github.com/KevinWMcGowan/movielens.git
2. Install necessary R packages:
   glmnet, dplyr, stringr, tidyverse, and more (full list in the R script & performed automatically).
3. Open the R project in RStudio.
4. Run the R Markdown file to generate the report:
rmarkdown::render("Movielens_report.Rmd")


## 🧠 Methodology

1. Data Processing:
- Merged and cleaned the ratings and movies datasets.
- Engineered features like user rating counts, average ratings, and movie release decades.
3. Modeling:
- Used Elastic Net Regression with cross-validation to optimize lambda (regularization).
- Evaluated model performance on training, validation, and final holdout sets.
4. Performance:
- Achieved an RMSE of ~0.032 on the final holdout set, highlighting the model’s robustness.

## 📋 Key Results

- The model predicts movie ratings with a low error rate, providing a strong foundation for building recommendation systems.
- Features like user_movie_diff, user_avg_rating, and movie_avg_rating were the most influential predictors.

## 🤝 Connect

Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/kevin-w-mcgowan-m-s-iop/) for collaboration opportunities
