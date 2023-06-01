# IICOFDM
Code repository for paper &lt;IICOFDM: an interpretable ICO fraud detection model based on Random Forest and SHAP>  
# IDE  
pycharm and jupternotebook, Compiler Environment: python 3.9  
# Primary dependency libs or packages:
python3.9  
numpy 1.21.5  
pandas 1.4.4  
seaborn 0.11.2  
matplotlib 3.5.2  
scikit-learn 1.0.2  
LightGBM 3.3.4  
scikitplot 0.3.7  
shap 0.41.0  
# Data  
The dataset includes the original dataset (dataset_fraud_14.11_2020), the initial ICO fusion dataset constructed after data type conversion and feature filtering (data_1), the dataset after missing value filling (data_2 is), the training set after category imbalance processing and feature selection (data_train, 70%), and the test set (data_test, 30%).  
# Code 1-5:
Data pre-processing codes, including feature type conversion, missing value filling, One-Hot encoding, data analysis, sampling, feature selection, and other pre-processing work.  
# Code 6：
The performance of the model is compared for different combinations of three sampling methods and three feature selection methods, and the joint preprocessing method of SMOTE+Tomek_Link and RFE is finally adopted.  
# Code 7:
Performing hyperparameter tuning and plot the validation curve of the parameters.  
# Code 8:
Plotting validation curves for multiple models
# Code 9:
Plotting confusion matrix, ROC curve, KS curve, Lift curve of the model
# Code 10：
Analyzing features to enhance model interpretability
