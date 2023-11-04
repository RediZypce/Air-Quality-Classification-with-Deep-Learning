# Air Quality Classification With Deep Learning
#### Project Overview
This [Jupyter Notebook](Air_Quality_Prediction_DL.ipynb) project aims to build a classifier using deep learning to categorize air quality based on historical data. The goal is to classify air quality into one of six categories: 'Very Poor,' 'Poor,' 'Moderate,' 'Satisfactory,' 'Severe,' and 'Good.' This classification is based on various environmental features and air quality indices.
![image](https://github.com/RediZypce/Air-Quality-Classification-with-Deep-Learning/assets/109640560/9a314e82-c05c-4a39-88ef-35065ae84d05)

# Datasets
The project uses two datasets for training and testing the deep learning model:

* [__air_quality_train.csv:__](air_quality_train.csv) This dataset contains the training data, including feature columns and labels for air quality. The training dataset has 7782 entries.

* [__air_quality_test.csv:__](air_quality_test.csv) This dataset contains the testing data with similar feature columns and labels for air quality. The testing dataset is used to evaluate the model's performance.

# Data Features
The datasets consist of the following features:

* PM2.5 (Particulate Matter 2.5)
* PM10 (Particulate Matter 10)
* NO (Nitric Oxide)
* NO2 (Nitrogen Dioxide)
* NOx (Nitrogen Oxides)
* NH3 (Ammonia)
* CO (Carbon Monoxide)
* SO2 (Sulfur Dioxide)
* O3 (Ozone)
* Benzene
* Toluene
* Xylene
* AQI (Air Quality Index)
* Air_Quality (Label)

# Methodology
__Data Preprocessing:__ The project begins with loading and preprocessing the data. This includes handling missing values, encoding the labels, and splitting the data into training and testing sets.

__Model Architecture:__ A deep learning model is designed using the TensorFlow and Keras libraries. The model architecture includes an input layer, hidden layers, and an output layer. The architecture aims to learn the mapping between the input features and the air quality labels.

__Model Training:__ The model is trained on the training dataset using the specified architecture. The training process involves optimizing the model's parameters to minimize the loss function.

__Model Evaluation:__ After training, the model is evaluated using the testing dataset. Evaluation metrics like accuracy, precision, recall, and F1-score are computed to assess the model's performance.

__Confusion Matrix:__ A confusion matrix is generated to visualize the model's performance in classifying each category of air quality.

# Usage
* Ensure you have the necessary libraries installed (e.g., TensorFlow, Pandas, Scikit-learn, Matplotlib, Seaborn).
* Place the training dataset (air_quality_train.csv) and testing dataset (air_quality_test.csv) in the same directory as this Jupyter Notebook.
* Run the notebook cells sequentially to load the data, train the model, and visualize the results.

# Contact 
Feel free to contact me on [X](https://twitter.com/home)
