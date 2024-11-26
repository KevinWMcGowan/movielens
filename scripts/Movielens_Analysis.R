# MovieLens Project Analysis Script
# This script processes the MovieLens 10M dataset, performs feature engineering,
# and trains an elastic net regression model to predict movie ratings.

# Requirements:
# - Ensure necessary libraries are installed (see library loading section).
# - The script will download the MovieLens dataset if not already present. 


################################################################################
# Load Libraries and Setup
################################################################################
# List of required packages
packages <- c("readr", "tinytex", "dplyr", "caret", "stringr", "tidyverse", "tidyr", "broom", "glmnet", "Matrix","coefplot")

# Check and install missing packages
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

# Load the libraries
lapply(packages, library, character.only = TRUE)


################################################################################
# Download Data
################################################################################

#Download, load, and split the dataset into edx (90% for training) and Final Holdout (10% for testing)

# Increase timeout for downloading large datasets
options(timeout = 120)

# Download and unzip the dataset
dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# load the user ratings half of the data set
ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

# load the movie half of the data set
movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

# Read and process the ratings file by specifying variable types
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

# Read and process the movies file
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

# Merge ratings and movies datasets by specifying variable types
movielens <- left_join(ratings, movies, by = "movieId")

# Split the data into training (edx) and final holdout test sets
set.seed(1, sample.kind="Rounding") # Use "Rounding" for R 3.6 or later
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Ensure all userId and movieId in the holdout set are present in the training set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add back any rows excluded from the holdout set into the training set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

# Clean up unnecessary variables
rm(dl, ratings, movies, test_index, temp, movielens, removed)



################################################################################
#check and inspect data set
################################################################################

##see top 5 rows of each column
head(edx)
#count number of rows
nrow(edx)

#count number of columns
ncol(edx)

#count how many times movies received a rating of 3
sum(edx$rating ==3)

#see how many individual movies are in the data set
n_distinct(edx$movieId)

#See how many individual users are in the data set
n_distinct(edx$userId)

# Count how many movies are assigned the following genres
genres = c("Drama", "Comedy", "Thriller", "Romance")
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})

#Starting with the highest, count how many ratings each movie has
count_ratings_by_movie <- edx %>%
  group_by(title) %>%
  summarise(rating_counts = n())%>%
  arrange(desc(rating_counts))
count_ratings_by_movie

#see the top 5 rating assigned to movies
most_common_ratings <- edx %>%
  group_by(rating) %>%               
  summarise(total_occurance = n()) %>%  
  arrange(desc(total_occurance)) %>%    
  head(5)
print(most_common_ratings)


# Classify ratings as whole-star or half-star and count how many of each
rating_summary <- edx %>%
  mutate(rating_type = ifelse(rating %% 1 == 0, "Whole Star", "Half Star")) %>%  # Classify ratings
  group_by(rating_type) %>%
  summarise(total_count = n())  # Count occurrences of each type
print(rating_summary)


################################################################################
# Feature Engineering
################################################################################

# This section includes a function that will succinctly build all 20 features to 
# get the most out of the dataset. Features fall into three categories: 
  #1. Temporal features (i.e., rating year, month, week)
  #2. User features (i.e., average user rating, recent_rating count, rating standard deviation)
  #3. Movie features (i.e., movie rating count, movie average rating, movie rating standard deviation)

feature_engineering <- function(data) {
  
  # Step 1: Extract movie year and other temporal features
  data <- data %>%
    mutate(
      rating_year = year(as_datetime(timestamp)),
      rating_month = month(as_datetime(timestamp)),
      rating_day_of_week = wday(as_datetime(timestamp), label = TRUE),
      movie_year = as.numeric(gsub("[^0-9]", "", str_extract(title, "\\((\\d{4})\\)"))),
      movie_age = rating_year - movie_year,
      decade = paste0(floor(movie_year / 10) * 10, "s") # Decade feature
    )
  
  # Step 2: User-specific features
  user_features <- data %>%
    group_by(userId) %>%
    summarise(
      user_rating_count = n(),
      user_avg_rating = mean(rating, na.rm = TRUE),
      rating_duration = max(rating_year) - min(rating_year),
      recent_rating_count = sum(rating_year >= (max(rating_year) - 1)),
      last_rating_date = max(as_datetime(timestamp)),
      avg_ratings_per_year = n() / (max(rating_year) - min(rating_year) + 1),
      rating_std_dev = sd(rating, na.rm = TRUE),
      trend = ifelse(n() > 1, coef(lm(rating ~ rating_year))[2], 0)
    ) %>%
    ungroup() %>%
    mutate(trend = ifelse(is.na(trend), 0, trend)) # Replace NA trends with 0
  
  # Step 3: Movie-specific features
  movie_features <- data %>%
    group_by(movieId) %>%
    summarise(
      movie_rating_count = n(),
      movie_avg_rating = mean(rating, na.rm = TRUE),
      movie_rating_std_dev = sd(rating, na.rm = TRUE)
    ) %>%
    ungroup() %>%
    mutate(movie_rating_std_dev = ifelse(is.na(movie_rating_std_dev), 0, movie_rating_std_dev)) # remove NAs 
  
  # Step 4: Combine all features into the main dataset
  data <- data %>%
    left_join(user_features, by = "userId") %>%
    mutate(
      user_movie_diff = rating - user_avg_rating,
      days_since_last_rating = as.numeric(difftime(as_datetime(timestamp), last_rating_date, units = "days"))
    ) %>%
    left_join(movie_features, by = "movieId") %>%
    mutate(years_since_release = rating_year - movie_year)
  
  # Rename columns to replace hyphens with underscores
  colnames(data) <- gsub("-", "_", colnames(data))
  
  return(data)
}

# Apply the updated function
edx_clean <- feature_engineering(edx)


# Feature Engineering Summary
  #Now that the feature engineering process has been applied, the following code shows a summary

# Extract feature namess
feature_names <- colnames(edx_clean)

# Create a data frame for display
feature_table <- data.frame(Feature_Names = feature_names)

# Print the feature table
print(feature_table)

################################################################################
# Feature Validation with Visualizations 
################################################################################

# This section checks and evaluates the features developed above for errors.

# 1. Check for missing values and in key features
cat("Missing values check for main features:\n")
summary(edx_clean %>% select(user_rating_count, user_avg_rating, rating_duration, recent_rating_count, movie_rating_count, movie_avg_rating))
# Table above shows no missing values, suggesting the main features are functional.

# View Distribution of user and movie-specific features
cat("\nVisualizing distributions of user and movie-specific features:\n")
edx_clean %>%
  select(user_rating_count, user_avg_rating, movie_rating_count, movie_avg_rating) %>%
  pivot_longer(cols = everything(), names_to = "Feature", values_to = "Value") %>%
  ggplot(aes(x = Value)) +
  geom_histogram(bins = 30, fill = "blue", alpha = 0.7) +
  facet_wrap(~ Feature, scales = "free_x") +
  labs(title = "Distributions of Key Features", x = "Value", y = "Frequency") +
  theme_minimal()
  #the movie_avg_rating and user_avg_rating histograms show normal distributions of ratings around 3.5/5. 
  #The movie_rating_count histrogram shows that most movies receive less than 5K ratings.
  #Lastly, the user_rating_count histogram shows most users provide less than 1K movie ratings.

# 2. Look at the head of all features, and check for 0s in all features
head(edx_clean)
# Check for columns with non-zero values in edx_clean
non_zero_summary <- edx_clean %>%
  summarise(across(everything(), ~ sum(. != 0, na.rm = TRUE)))
# Print the summary
print("Number of non-zero values per column:")
print(non_zero_summary)
print(ncol(edx_clean))
  #We can see we now have 26 columns.
  #This output suggests the processing code above has effectively built the desired features.

# 3. Check movie-specific feature distributions
cat("\nSummary of movie-specific features:\n")
summary(edx_clean %>% select(movie_rating_count, movie_avg_rating, movie_rating_std_dev))
  # Table above shows no NAs, and no ranges of 0, suggesting functional movie-specific features.


# 4. Verify sample user rating trends (e.g., user rating trends and average ratings per year)
cat("Head of user rating trends and average ratings per year:\n")
print(head(edx_clean %>% select(userId, avg_ratings_per_year, trend)))
  # The 'trend' feature, which measures the relationship between a user's ratings and the year of rating, shows 0 for this sample.
  # A trend value of 0 occurs when users have made only a single rating (or very few) or have given the same rating to all movies across years.
cat("\nSample of user rating trends and average ratings per year:\n")
print(
  edx_clean %>%
    select(userId, avg_ratings_per_year, trend) %>%
    {set.seed(1); sample_n(., 10)}
)
  # As seen in this next sample, the feature engineering function accounts for cases with insufficient data by replacing NA trends with 0.
  #This  sample of 'trend' shows several different values, suggesting the trend feature is functional. 
  #Additionally, a trend of .66 suggest a trend between variables and possible usefulness for ML.

# View Distribution of user trends
cat("\nVisualizing user rating trends:\n")
ggplot(edx_clean, aes(x = trend)) +
  geom_histogram(bins = 30, fill = "green", alpha = 0.7) +
  labs(title = "Distribution of User Rating Trends", x = "Trend", y = "Frequency") +
  theme_minimal()
  #the vast majority of trends are 0, but there are trends in both positive and negative directions.


# 5. Ensure correct decade distribution after feature engineering
cat("\nDecade distribution in dataset:\n")
decade_distribution <- table(edx_clean$decade)
print(decade_distribution)
# View Decade distribution
cat("\nVisualizing decade distribution:\n")
ggplot(as.data.frame(decade_distribution), aes(x = Var1, y = Freq)) +
  geom_bar(stat = "identity", fill = "purple", alpha = 0.7) +
  labs(title = "Decade Distribution of Movies", x = "Decade", y = "Count") +
  theme_minimal()
#The table and barchart above show the majority of movie ratings are for movies in the 90's, 00's, and 80's
#This finding aligns with the launching of movie lens in the late 90s, and demonstrates the successful development of the 
#decades featues. These features aim to find relationships with eras and nostalgic ratings.

# 6. Final dataset dimension check
cat("\nFinal dataset dimensions:\n")
cat("Rows:", nrow(edx_clean), "Columns:", ncol(edx_clean), "\n")
# The printout above confirms we've created 96 features for use in ML.

################################################################################
# Numeric Modeling Algorithm
################################################################################

# This script uses Gaussian regression to predict movie ratings as a continuous variable.
# Regression offers high accuracy for continuous targets but may predict arbitrary decimals 
# rather than adhering strictly to 0.5-point increments. 
# Nonessential non-numeric variables (e.g., titles) are removed, 
# while essential ones (e.g., genres) are encoded as numeric for modeling.

# Partition the edx_clean data into training and validation sets 
set.seed(1, sample.kind = "Rounding")  # Use "Rounding" if using R 3.6 or later
train_index <- createDataPartition(y = edx_clean$rating, times = 1, p = 0.8, list = FALSE)

# Create training and validation sets
train_set <- edx_clean[train_index, ]
validation_set <- edx_clean[-train_index, ]


# Check dimensions of the training and validation sets
cat("Training set dimensions: Rows:", nrow(train_set), "Columns:", ncol(train_set), "\n")
cat("Validation set dimensions: Rows:", nrow(validation_set), "Columns:", ncol(validation_set), "\n")

# Encode essential non-numeric variables as numeric
  # Identify numeric columns and essential categorical columns
essential_columns <- c("genres", "last_rating_date", "rating_day_of_week", "decade")
selected_columns <- union(names(train_set)[sapply(train_set, is.numeric)], essential_columns)

# Subset the training and validation datasets
train_set_numeric <- train_set[, intersect(colnames(train_set), selected_columns)]
validation_set_numeric <- validation_set[, intersect(colnames(validation_set), selected_columns)]

# Log the column names and dimensions
cat("Columns in train_set_numeric:\n")
print(colnames(train_set_numeric))
cat("Dimensions of train_set_numeric: Rows:", nrow(train_set_numeric), "Columns:", ncol(train_set_numeric), "\n")

cat("Columns in validation_set_numeric:\n")
print(colnames(validation_set_numeric))
cat("Dimensions of validation_set_numeric: Rows:", nrow(validation_set_numeric), "Columns:", ncol(validation_set_numeric), "\n")
# above we see we now have 25 varables, all numeric, as is essential for regressian analysis. 

# Prepare predictors (x) and target variable (y) (This might take a few minutes)
x <- as.matrix(train_set_numeric[, colnames(train_set_numeric) != "rating"])
y <- train_set_numeric$rating


# Perform elastic net regression (this will take a few minutes)
elastic_net_model <- cv.glmnet(x, y, alpha = 0.5, family = "gaussian")


# Calculate RMSE from cross-validation MSE
rmse_values <- sqrt(elastic_net_model$cvm)

# Plot RMSE vs. log(lambda)
plot(
  log(elastic_net_model$lambda), rmse_values,
  type = "b", 
  xlab = "log(Lambda)",
  ylab = "Root Mean Squared Error (RMSE)",
  main = "RMSE vs. log(Lambda)"
)

# Add vertical lines for lambda.min and lambda.1se
abline(v = log(elastic_net_model$lambda.min), col = "blue", lty = 2, lwd = 2)
abline(v = log(elastic_net_model$lambda.1se), col = "red", lty = 2, lwd = 2)

# Add a legend for the lines
legend("topright", legend = c("lambda.min", "lambda.1se"),
       col = c("blue", "red"), lty = 2, lwd = 2)

# Extract RMSE values for lambda.min and lambda.1se
lambda_min_rmse <- rmse_values[which(elastic_net_model$lambda == elastic_net_model$lambda.min)]
lambda_1se_rmse <- rmse_values[which(elastic_net_model$lambda == elastic_net_model$lambda.1se)]

# Print results
cat("RMSE for lambda.min:", lambda_min_rmse, "\n")
cat("RMSE for lambda.1se:", lambda_1se_rmse, "\n")

# Cross-validation plot shows the relationship between lambda and predictive performance.
# The x-axis represents log(lambda), and the y-axis shows RMSE. Smaller lambda values retain 
# more features, while larger lambda values shrink coefficients to zero, excluding predictors.
# The optimal lambda (lambda.min) minimizes RMSE and is used to build the final model, as it 
# aligns with lambda.1se in this case.


# Identify the lambda corresponding to the minimum RMSE
best_lambda_rmse <- elastic_net_model$lambda[which.min(rmse_values)]
print(paste("Best lambda (based on RMSE):", best_lambda_rmse))

# Fit the model with the best lambda based on RMSE
final_model <- glmnet(x, y, alpha = 0.5, lambda = best_lambda_rmse)

# Print coefficients of the final model
print(coef(final_model))


################################################################################
# Predict on the Validation Set
################################################################################

# Prepare the predictor matrix (x_val) and target variable (y_val)
x_val <- as.matrix(validation_set_numeric[, colnames(validation_set_numeric) != "rating"])
y_val <- validation_set_numeric$rating


# fit the final model using the RMSE-optimized lambda
final_model_rmse <- glmnet(x, y, alpha = 0.5, lambda = best_lambda_rmse)

# Predict ratings for the validation set using the RMSE-optimized final model
predictions <- predict(final_model_rmse, newx = x_val)

# Calculate RMSE on the validation set
rmse_validation <- sqrt(mean((predictions - y_val)^2))
cat("RMSE on Validation Set (Optimized for RMSE):", rmse_validation, "\n")


################################################################################
# visualize the predictions 
################################################################################

# check predicted vs actual
residuals <- y_val - predictions
hist(residuals, breaks = 50, main = "Residuals Distribution", xlab = "Residuals")

# The residuals, representing the difference between actual and predicted ratings, 
# are tightly clustered between -0.05 and +0.05, indicating that the model predicts 
# movie ratings with high accuracy and minimal error. 
# This narrow range suggests the model effectively captures the underlying patterns in the data, 
# with no significant bias or overestimation.

# With this solid performance, it's time for to test the model on the independent final holdout set.


################################################################################
# Final Holdout Test
################################################################################

# This section tests the trained model on the final holdout set.

# Apply the feature engineering function to the final holdout set
final_holdout_clean <- feature_engineering(final_holdout_test)

# Ensure only numeric columns and essential categorical columns are included
essential_columns <- c("genres", "last_rating_date", "rating_day_of_week", "decade")
selected_columns <- union(names(final_holdout_clean)[sapply(final_holdout_clean, is.numeric)], essential_columns)
final_holdout_numeric <- final_holdout_clean[, intersect(colnames(final_holdout_clean), selected_columns)]

# Remove unnecessary columns like 'title'
final_holdout_numeric <- final_holdout_numeric[, colnames(train_set_numeric)]

# Verify dimensions and alignment
cat("Number of columns in final holdout numeric dataset:", ncol(final_holdout_numeric), "\n")
cat("Expected number of predictors for the model:", length(coef(final_model)) - 1, "\n")

# Prepare the predictor matrix and target variable
x_holdout <- as.matrix(final_holdout_numeric[, colnames(final_holdout_numeric) != "rating"])
y_holdout <- final_holdout_numeric$rating

# Ensure dimensions match before predicting####see if WORKS WOTHOUT ELSE STATMENT
if (ncol(x_holdout) == length(coef(final_model)) - 1) {
  # Predict ratings for the final holdout set using the trained model
  predictions_holdout <- predict(final_model, newx = x_holdout)
  
  # Calculate RMSE on the final holdout set
  rmse_holdout <- sqrt(mean((predictions_holdout - y_holdout)^2))
  cat("RMSE on Final Holdout Set:", rmse_holdout, "\n")
}

# Final holdout RMSE (~0.03262) shows minimal increase from validation set RMSE (0.03147), 
# confirming the model's robustness in predicting movie ratings for unseen data.


################################################################################
# visualize the performance
################################################################################

# Calculate residuals for the final holdout set
residuals_holdout <- y_holdout - predictions_holdout

# Visualize the residuals
hist(
  residuals_holdout, 
  breaks = 50, 
  main = "Residuals Distribution for Final Holdout Set", 
  xlab = "Residuals",
  col = "skyblue",
  border = "white"
)
# Like the histogram of residuals on the validation set, most predictions fell within + or - .05. 
# Which is within 1 rating level- this is an accurate result.


# Top 10 largest errors

# The table below shows that our largest mistakes were .13 and these mistakes don't appear to be in any particular genre or decade, suggestion
# no absolutely necessary adjustments.

# Calculate residuals and absolute residuals
residuals_holdout <- y_holdout - predictions_holdout
absolute_residuals <- abs(residuals_holdout)

# Find the top 10 largest errors
largest_errors <- order(absolute_residuals, decreasing = TRUE)[1:10]

# Create a table with movie details and error metrics for the largest errors
largest_errors_table <- data.frame(
  final_holdout_clean[largest_errors, c("movieId", "title", "genres")], # Movie details
  Actual = y_holdout[largest_errors],                                  # Actual ratings
  Predicted = predictions_holdout[largest_errors],                    # Predicted ratings
  Residual = residuals_holdout[largest_errors]                        # Residuals
)

# Print the table
cat("Details of Movies with the Largest Prediction Errors:\n")
print(largest_errors_table)


################################################################################
# Discussion
################################################################################

#An RMSE of ~0.0326 on a scale of 0.5 to 5.0 ratings indicates very high accuracy. The predictions are extremely close to the actual ratings.
#This level of performance suggests the feature engineering, model selection (elastic net), and hyperparameter tuning were effective.


################################################################################
# References
################################################################################

#Durrani, A. (2024, August 15). Top streaming statistics in 2024. Forbes. https://www.forbes.com/home-improvement/internet/streaming-stats/ 
  
#F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

#Jesse Vig, Shilad Sen, and John Riedl. 2012. The Tag Genome: Encoding Community Knowledge to Support Novel Interaction. ACM Trans. Interact. Intell. Syst. 2, 3: 13:1–13:44. https://doi.org/10.1145/2362394.2362395

#Link to MovieLens dataset 10mil: https://grouplens.org/datasets/movielens/10m/