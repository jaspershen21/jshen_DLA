# Multimodal Low vs. High Situation Awareness Classification using Physiological Sensors

This repository contains the source code for the research paper, "Multimodal Low vs. High Situation Awareness Classification using Physiological Sensors". The project focuses on developing an ensemble machine learning model to predict an operator's sitation awareness (SA) from non-invasive physiological signals.

## About the Code

This repository consists of Jupyter Notebooks used for exploratory data analysis, data processing, model training, and evaluation as described in the paper. The notebooks are organized into various modeling approaches.

-   `Early Investigations` contains code for the exploratory data analysis and training baseline logistic regression models on an earlier iteration of the dataset. Here, we also perform correlation calculations for the features as well as attempts to perform feature selection and hyperparameter tuning.
-   `10 Fold Raw SA` contains code for exploring the dataset using raw SA, instead of adjusted SA from the original dataset. This also includes determining where to split the data to obtain various class balances. We also train and evaluate models using this raw SA dataset using 10-fold stratified cross-validation. Due to the class imbalance from the data splitting, we experiment with different strategies (different class weights, oversampling, undersampling, SMOTE) in order to reduce the impact of the class imbalance.
-   `Model by Sensor` contains code for the initial sensor fusion approach where we fit one model for each sensor then combine them using a simple unweighted average. We experiment with various regularization strategies (no regularization, LASSO, ridge regression) and compare them against each other to determine if there is a difference in performance. We ensure that data leakage is not a concern while maintaining the 10-fold stratified cross-validation using the adjusted SA dataset with.
-   `3 Moving Average with Stacking` contains code for training an ensemble model using the stacking ensemble approach with the average of 3 SA data. We only experiment with SA level 1 and do not perform cross-validation to determine the feasibility of this approach.
-   `3 Moving Average` contains code for exploratory data analysis with the average of 3 SA data and various ensembling approaches using the sensor fusion approach.
    -   `Model_By_Sensor_With_Intercept.ipynb` contains the finalized code for the ensemble modeling approach. We use the average of 3 SA data with 10-fold cross-validation to evaluate ensemble model performance while using 5-fold inner cross-validation for hyperparameter tuning of the single-sensor models. This also contains various ensemble approaches (majority voting, unweighted average, weighted average, bayesian decision) using this strategy, where we end up proceeding with a weighted average based on the training set F1 score.

## How to Use

NOTE: The dataset used in this project was obtained from the original authors, who have not made the dataset public as of yet.

1. **Clone the repository:**
    ```bash
        git clone https://github.com/jaspershen21/jshen_DLA.git
    ```
2. **Install dependencies:**
   It is recommended to create a virtual environment. Install the required packages using pip:
    ```bash
        pip install pandas scikit-learn numpy matplotlib scipy
    ```
3. **Run the Notebooks:**
   Launch Jupyter Notebook, JupyerLab, or an IDE that supports Jupyter Notebooks and run the various notebooks as you see fit.

## Dependencies

-   Python 3.x
-   pandas
-   scikit-learn
-   numpy
-   matplotlib
-   scipy
