# Breast Cancer Prediction Project

## Purpose of the Study 🧪

The purpose of this study is to develop a predictive model for breast cancer outcomes. The primary objective is to explore the effectiveness of different machine learning classifiers in accurately predicting whether a given case is malignant or benign. 

## Description of the Dataset 🔬

### Data Source
The dataset used in this project is the Wisconsin Prognostic Breast Cancer (WPBC) dataset, sourced from real-world clinical data.
https://archive.ics.uci.edu/dataset/16/breast+cancer+wisconsin+prognostic

### Dataset Overview
- **Number of attributes:** 34 (ID, outcome, 32 real-valued input features)
- **Number of records:** The dataset contains a specific number of records, each corresponding to a distinct breast cancer case.
- **Number of instances:** 198

- **Attribute information**
1) ID number
2) Outcome (R = recur, N = nonrecur)
3) Time (recurrence time if field 2 = R, disease-free time if 
	field 2	= N)
4-33) Ten real-valued features are computed for each cell nucleus:

	a) radius (mean of distances from center to points on the perimeter)
	b) texture (standard deviation of gray-scale values)
	c) perimeter
	d) area
	e) smoothness (local variation in radius lengths)
	f) compactness (perimeter^2 / area - 1.0)
	g) concavity (severity of concave portions of the contour)
	h) concave points (number of concave portions of the contour)
	i) symmetry 
	j) fractal dimension ("coastline approximation" - 1)

34) Tumor size - diameter of the excised tumor in centimeters
35) Lymph node status - number of positive axillary lymph nodes
observed at time of surgery

**Missing attribute values** 
	Lymph node status is missing in 4 cases.

**Class distribution**
  151 nonrecur, 47 recur

## Classification 📊

### Libraries used
Pandas, Scikit-learn, Matplotlib, Seaborn, Imblearn

### Data Preprocessing
1. **Reading Data:** The project reads the WPBC dataset and assigns relevant column names.
2. **Handling Missing Values:** Rows with missing values in the 'Lymph node status' column are dropped.
3. **Data Encoding:** 'Outcome' column is encoded to numerical values ('R' and 'N' to 0 and 1).
4. **Oversampling:** SMOTE (Synthetic Minority Over-sampling Technique) is applied to address class imbalance.
5. **Data Splitting:** The dataset is split into training and testing sets.
6. **Standardization:** Features are standardized using StandardScaler.
7. **Correlation Analysis** Highly correlated features are identified and dropped during data preprocessing.


### Classifiers
**The choice of hyperparameters has been optimized based on experimentation and grid search.**

- The project evaluates the performance of several classifiers:
  - Neural Network
  - Naive Bayes
  - Support Vector Machine
  - Decision Tree
  - AdaBoost
  - Bagging

### Evaluation Metrics
- Performance is assessed using standard metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

### Summary
1. **Neural Network and Support Vector Machine (SVM)**:
   - They share the highest accuracy (0.8649), suggesting they are correctly identifying a high number of true positives and true negatives.
   - Neural Network has the highest F1 Score (0.8684), which is indicative of a balanced precision-recall trade-off.
   - Precision is very high for both, indicating a low number of false positives.
2. **Naive Bayes**:
   - Shows the lowest accuracy (0.6757) and the lowest F1 Score (0.6667), indicating that it might not be as effective as the others for this particular dataset or problem.
   - The recall is notably lower compared to other classifiers, which means it is missing a higher number of true positives.
3. **Decision Tree**:
   - Has good performance across all metrics, with a particularly high precision (0.9167), which suggests that it is reliable when it predicts a positive class.
4. **AdaBoost**:
   - Shows moderate performance in terms of accuracy (0.7973) and recall (0.6829), but with a high precision (0.9333), indicating fewer false positives.
   - The lower recall suggests it's missing out on identifying some true positives.
5. **Bagging**:
   - Demonstrates strong performance, particularly in precision (0.9394), indicating a very low false positive rate.
   - It has the lowest F1 Score (0.8378) among the top-performing classifiers, which suggests a slightly less balanced precision-recall trade-off compared to Neural Network and SVM.

## Usage 🐍
Running main.py creates each of the classifiers and checks their accuracy, precision, recall and F1 score. Those metrics are printed out to the console and to output_table.csv

## Authors ✍️
**Paweł Harasiuk:** https://github.com/PawelHarasiuk 

**Lena Marusik:** https://github.com/lenamiriam

**Maciej Lizuraj:** https://github.com/maciejlizuraj
