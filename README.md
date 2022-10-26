# IICOFDM
An interpretable ICO fraud detection model based on Random Forest and SHAP

environment：
python3.7
numpy 1.18.1
pandas 1.0.1
scikit-learn 0.22.1
shap 0.40.0

The files in the folder "Figures" are all the images used in the project, 
drawn by matplotlib, seaborn, scikit-plot and other tools, and then typeset by Visio.

The data in the folder "data" consists of raw data and pre-processed data. 
The "dataset_fraud_14.11_2020" is the raw data, "data_1", "data_2", 
and "data_3" are the raw data after feature type conversion, missing value padding, 
and One-Hot encoding, respectively, and "data_clean" is the dataset after normalization 
and feature selection and finally used for model training. When training the model, 
we save the divided training set and test set as "data_train" and "data_test" 
in advance so that they can be read at any time during the subsequent process.

The files in the folder "Modeling" are the code of the project, 
which we compiled using pycharm, where SHAP force plot needs to be run in Jupyter Notebook. 
The codes 1-4 perform feature engineering on the data, which are data type conversion, 
missing value filling, unique heat coding, and feature selection. 
The adaptive model selection is mainly performed in code 5, and Random Forest is finally determined as the base model of IICOFDM. 
Code 6 performs the model hyperparameter tuning, and codes 7-8 analyze the generalization capability of IICOFDM. 
Finally, code 9 enhances the interpretability of the model by introducing SHAP values, and IICOFDM modeling is concluded.
