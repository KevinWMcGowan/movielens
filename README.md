# 🎥 MovieLens Rating Prediction Project

Welcome to the **MovieLens Rating Prediction Project**, an in-depth analysis and predictive modeling project leveraging the MovieLens 10M dataset. This repository includes an R Markdown report (`Movielens_report.Rmd`), a PDF version knit from the R Markdown report (`Movielens_report.Pdf`), and  a companion R script (`Movielens_analysis.R`) to explore, process, and model movie ratings.

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
    - `Movielens_report.pdf`: PDF version knit from R Markdown report
  - **scripts/**: Contains the R script
    - `Movielens_analysis.R`: Script for all analysis and modeling
## 📊 Dataset Information

The dataset used is the [MovieLens 10M Dataset](https://files.grouplens.org/datasets/movielens/ml-10m.zip). It includes:
- **10 million ratings** from **72,000 users** on **10,000 movies**.
- **Key fields**: `userId`, `movieId`, `rating`, `timestamp`, `title`, and `genres`.

Since the dataset exceeds GitHub's file size limit, it is not included in this repository. The analysis script and report automatically download and process the dataset.

## 🚀 How to Run

Instructions for Running the Project

This repository contains all the necessary files for the MovieLens project, including an R script for analysis, an R Markdown file for generating a comprehensive report, and the final PDF output.

Running the Code

1.	Download the R Script:
	- Navigate to the scripts/ folder and download the R script (Movielens_analysis.R).
	- Open the script in R or RStudio and run the code sequentially to reproduce the analysis.
2. Generate the Full Report:
	- Navigate to the reports/ folder and download the R Markdown file (Movielens_report.Rmd).
	- Open the file in RStudio and click the “Knit” button to generate the report in PDF format.
	- Note: This will re-run the entire analysis, including downloading the MovieLens dataset and processing it.
3. View the Final Report:
	- To see the results without re-running the analysis, simply open the pre-generated PDF report (Movielens_report.pdf) located in the reports/ folder.

Additional Notes

	- Both the R script and R Markdown file include code for downloading the MovieLens dataset directly from its official source.
	- Running the R script or knitting the R Markdown file will automatically install required R packages if they are not already installed.
	- The dataset is approximately 200 MB. Ensure you have a stable internet connection when running the analysis for the first time.


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
