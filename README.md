# California Housing Price Prediction 

An end-to-end Machine Learning pipeline utilizing Python, Scikit-Learn, and Pandas to predict median house values in California. This project demonstrates advanced data preprocessing, diagnostic visualization, debugging data leakage, and transitioning from linear to ensemble-based machine learning models.

## Project Overview & Architecture

The objective of this project is to predict the `median_house_value` of various California districts based on demographic, geographic, and structural features. 

The pipeline undergoes four major phases:
1. **Exploratory Data Analysis (EDA):** Visualizing distributions and feature-target relationships using custom grid scatter plots.
2. **Feature Engineering & Transformation:** Resolving heavily right-skewed data and stabilizing variance across features.
3. **Debugging & Validation:** Detecting and fixing a critical data leakage bug (~99% $R^2$ score).
4. **Model Evolution:** Transitioning from a Polynomial Ridge Regression baseline to a robust Random Forest Regressor.

---


### 1. Handling Skewness & Target Capping
Initial visualizations revealed heavy right-skewed tails across multiple numerical features (`population`, `total_rooms`, `total_bedrooms`, and `households`). Additionally, the data exhibited artificial maximum ceilings (e.g., a sharp data spike capping the target variable at $500,000$).

* **Transformation:** Instead of trimming 15–20% of the dataset to eliminate extreme outliers, a **Yeo-Johnson Power Transformation** (`PowerTransformer`) was implemented. This dynamically optimized $\lambda$ parameters to convert highly skewed curves into symmetrical normal distributions centered around 0.
* **Scaling:** A `StandardScaler` was applied to ensure uniform feature magnitude, protecting downstream distance-based optimization steps.

### 2. Debugging Data Leakage
During validation, an initial model trial returned an artificial, near-perfect $R^2$ score of **0.9978**. 

* **The Diagnostics:** Custom isolated subplots of the training dataframe revealed that the target column (`median_house_value`) had inadvertently remained inside the feature matrix ($X$) before passing into the `PolynomialFeatures` transformer. 
* **The Fix:** The target feature was explicitly purged from the feature space prior to splitting and transformation, restoring the pipeline's true mathematical integrity.

---

## Model Performance & Evolution

The project evaluates two distinct algorithmic paradigms to compare how linear models handle complex spatial data versus tree-based ensemble methods.

### 1. Baseline: Polynomial Ridge Regression
* **Approach:** Degree 2 polynomial features with an $L2$ regularization penalty ($\alpha = 1.0$) to control structural complexity. 
* **Results:** * **$R^2$ Score:** `0.65` (Explains 65% of data variance honestly)
  * **Mean Absolute Error (MAE):** `~$46,500`
* **Limitation:** Linear models struggle to natively map non-linear bimodal geographic features like `latitude` and `longitude` (e.g., coastal metropolitan pricing pockets).

### 2. Champion Model: Random Forest Regressor
* **Approach:** A non-linear ensemble model utilizing recursive data partitioning to handle localized geographic interactions and regional target capping safely.
* **Results:**
  * **$R^2$ Score:** **`0.82`**  *(20% performance jump over the linear baseline)*
* **Why it Won:** The tree structure isolates highly priced coastal coordinate boundaries effortlessly, preventing the artificial $500k target ceiling from warping global predictions.

