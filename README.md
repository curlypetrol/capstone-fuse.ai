### Start the project

Create a conda environment based on the environment file at the root of the project.
```
conda env create --file environment.yml
```

# Fetal Risk Classification using CTG Data: A Machine Learning Approach

![image](https://github.com/curlypetrol/capstone-fuse.ai/assets/114018163/2779e93b-7ade-4ba6-be5f-18e08bef9ac3)

**Authors:** Edith Gómez, José Elier Fajardo, Sarah Peña

## Objective:
The objective of this project is to develop and evaluate machine learning models for fetal risk classification using Cardiotocography (CTG) data. By leveraging various machine learning algorithms, the aim is to create a robust model that accurately predicts fetal health status, ultimately improving prenatal care.

## Table of Contents:
1. [**Introduction**](https://github.com/curlypetrol/capstone-fuse.ai?tab=readme-ov-file#1-introduction)<br>
2. [**Data Description**](https://github.com/curlypetrol/capstone-fuse.ai?tab=readme-ov-file#2-data-description)<br>
3. [**Data Preprocessing**](https://github.com/curlypetrol/capstone-fuse.ai?tab=readme-ov-file#3-data-preprocessing)<br>
4. [**Model Selection**](https://github.com/curlypetrol/capstone-fuse.ai/blob/main/README.md#5-model-selection)<br>
5. [**Model Evaluation**](https://github.com/curlypetrol/capstone-fuse.ai?tab=readme-ov-file#6-model-evaluation)<br>
**6. Results**<br>
**7. Conclusion**<br>
**8. Future Work**<br>
**9. References**<br>

## 1. Introduction
Reducing child mortality rates stands as a paramount objective within the global health agenda, intricately linked with the United Nations' Sustainable Development Goals. By 2030, nations aspire to eradicate preventable deaths among newborns and children under the age of five, striving to diminish under‑5 mortality to as low as 25 per 1,000 live births. Concurrently, maternal mortality remains a concerning issue, claiming 295,000 lives during and post-pregnancy in 2017, with the majority transpiring in low-resource settings and often preventable.

In response to these challenges, technologies such as Cardiotocography (CTGs) have emerged as accessible tools for assessing fetal health, empowering healthcare practitioners to preemptively address risks to both child and maternal well-being. Operating through the transmission and analysis of ultrasound pulses, CTGs provide critical insights into fetal heart rate (FHR), movements, uterine contractions, and more.

The convergence of health complications during gestation as a global concern underscores the urgency for innovative solutions. Machine learning (ML) algorithms offer promising avenues for predicting fetal health based on cardiotocographic (CTG) data, categorizing health states into normal, needing assurance, or indicative of pathology. 

By exploring various ML techniques, including support vector machines, random forests, and others, the project aims to identify key factors in CTG data that influence fetal health and improve prediction accuracy. This research has the potential to enhance fetal monitoring and ultimately improve pregnancy outcomes.

## 2. Data Description
The dataset utilized in this study comprises 2,126 records sourced from Cardiotocogram (CTG) exams, a pivotal tool in assessing fetal health during pregnancy. The data was meticulously collected and annotated, with a focus on feature extraction to aid in the classification process.

Each record in the dataset encompasses an array of features extracted from CTG exams, capturing essential parameters indicative of fetal well-being. These features have been meticulously curated to encompass a comprehensive understanding of fetal health dynamics.

To ensure the accuracy and reliability of the dataset, three expert obstetricians meticulously classified the records into three distinct classes:

- Normal: Records in this category denote instances where fetal health parameters fall within the expected range, indicating a healthy status.

- Suspect: This class encompasses records where certain fetal health indicators display deviations or abnormalities, suggesting potential concerns that warrant further medical attention or monitoring.

- Pathological: Records classified under this category indicate significant deviations from normal fetal health parameters, signaling potential pathological conditions that require urgent medical intervention or specialized care.

The dataset serves as a valuable resource for researchers, practitioners, and stakeholders involved in maternal-fetal healthcare, facilitating exploratory analysis, predictive modeling, and the development of decision support systems aimed at enhancing prenatal care and mitigating adverse pregnancy outcomes.

For further exploration and analysis, the dataset is publicly accessible on Kaggle via the following link: [Fetal Health Classification Dataset.](https://shorturl.at/fgjJN)

## 3. Data Preprocessing
In order to explore and verify the features and challenges within the dataset, the following steps were undertaken:

1. Importing Libraries: Necessary libraries such as pandas, matplotlib, numpy, seaborn, and sklearn were imported for data manipulation, visualization, and preprocessing.
2. Handling Duplicate Rows: Initially, the dataset containing 2126 rows and 22 columns was checked for duplicate rows. Thirteen duplicate rows were identified and subsequently removed, resulting in a dataset with 2113 rows and 22 columns.
3. Handling Missing Values: No missing data was found in the dataset, thus no imputation or handling of missing values was required.
4. Scaling: Due to the significant scale differences among variables, scaling techniques were applied to ensure uniformity before model training. This step is crucial for maintaining model performance.
5. Identifying Skewness: Examination of variable distributions revealed skewness, particularly right skewness, among certain variables. Skewness can impact model performance; hence, it was noted for further consideration.

### Exploratory Data Analysis (EDA):

1. **Histogram Visualization:** Histograms were created to visualize the distribution of variables. Skewness and normal-like distributions among features were observed.
2. **Class Imbalance Check:** Class imbalance within the target variable (fetal_health) was identified, with a substantial difference in instances among the "normal," "suspect," and "pathological" categories. Addressing this imbalance is crucial for model training.
3. **Boxplot Analysis:** Boxplots were utilized to compare variable distributions across different classes. Significant differences were observed, especially concerning histogram variables, accelerations, and mean_value_of_long_term_variability.
4. **Correlation Analysis:** Correlation between the target variable and other features was examined. Features such as acceleration, histogram mode and mean, prolonged decelerations, and abnormal short-term variability showed higher correlations with fetal_health. This analysis aids in feature selection for model development.
5. **Feature Range Observation:** Charts depicting the ranges of features were generated to identify scaling differences and variations in feature ranges. Notably, histogram variances exhibited major variability, while features related to accelerations, contractions, and fetal movement showed limited variation.
6. **Feature Correlation Analysis:** Correlation between sets of features was explored to assess the necessity of using all features for model training. Understanding feature correlations helps in optimizing model performance and reducing redundancy.
7. **Target Variable Transformation:** Labels of the fetal_health variable were mapped to meaningful categories (1: normal, 2: suspect, 3: pathological) for better interpretation and understanding of the target variable.

These preprocessing steps lay the foundation for building robust machine learning models for fetal health classification, ensuring data quality, uniformity, and relevance.

## 5. Model Selection

### Selected Metrics
Given the objective of building a sickness prediction model, priority is given to identifying cases representing sickness presence. Hence, special attention is paid to the recall metric of the classes representing or indicating sickness, namely 'suspect' and 'pathological'.

Balanced accuracy is crucial due to the issue of imbalanced classes. It calculates the average recall obtained on each class, serving as our main metric for model optimization.

The averaging method used for combining metrics across classes is Macro-averaging, assigning equal importance to all labels regardless of their distribution.

### Hyperparameter Optimization in ML models
Hyperparameters of ML models such as Logistic Regression and Random Classifier are optimized based on their balanced accuracy to maximize performance. The best model is saved for further rigorous evaluation.

Parameter grids are selected considering training time (around 20 minutes), feasible parameter combinations (less than 500), and metric maximization. The final selection of parameter grids is the culmination of research, recommendations from experts, and experimentation to ensure optimal performance.

## 6. Model Evaluation

### Evaluation Approach
Final evaluation of the performance of selected ML models is conducted through ten-fold cross-validation with the test data partition. This ensures precise model evaluation independent of random data partition configurations that may skew scores. These results are utilized for final model comparison. We outlined our model evaluation approach for fetal health assessment, covering data preprocessing, model training, evaluation, and model storage, the following steps were the roadmap undertaken to determine our baseline metrics using a Dummy Classifier:

- **Baseline Metric Estimation:**<br>
  We initiate the evaluation process by preparing the dataset for model training. Data preprocessing involves importing essential libraries and standardizing the features using StandardScaler. Additionally, we define metrics to assess model performance, considering the 
  imbalanced nature of the target variable.

- **Model Training:**<br>
  To optimize our models, hyperparameters are tuned using GridSearchCV, which performs an exhaustive search over specified parameter values while optimizing the specified metric, balanced accuracy. The best performing model is selected based on the evaluation results.

-  **Evaluation:**<br>
   We rigorously evaluate the selected model using cross-validation to ensure precise performance evaluation, independent of random data partitions that may influence metrics. Baseline metrics are established using a dummy classifier that predicts the most frequent 
   class.

- **Saving the Best Model:**<br>
  Finally, the best-performing model is saved using the pickle library for future use and deployment.

### Artificial Neural Network Architecture
The final model evaluated is an Artificial Neural Network (ANN) implemented with the TensorFlow library. It features an architecture comprising three hidden layers with a Rectified Linear Unit (ReLU) activation function. Several techniques are employed to achieve faster convergence and prevent overfitting:

- Batch normalization: Re-centers and re-scales inputs at each hidden layer, enhancing training speed and maintaining a stable input distribution across layers.
- Dropout layers: Two dropout layers with a rate of 20% are added to reduce overfitting.
- L2 regularization: Applied in each hidden layer to further mitigate overfitting.

This architecture was determined through extensive research on similar projects and expert recommendations. The model runs for a total of 120 epochs. Also, the model with the highest recall on the 'Pathological' class is saved using the callback ModelCheckpoint, ensuring the highest possible sickness detection.

## 7. Results

## 8. Conclusion

## 9. Future Work

## 10. References
