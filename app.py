import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import Vectors

# Initialize Spark session
spark = SparkSession.builder.appName("WineClassificationApp").getOrCreate()

# Load the trained model
model = PipelineModel.load("wine_classification_model")

# Function to make predictions
def predict(features):
    df = spark.createDataFrame([(Vectors.dense(features),)], ["features"])
    prediction = model.transform(df)
    return prediction.select("prediction").collect()[0][0]

# Streamlit user interface
st.title("Wine Classification Model")
st.write("Enter the features of the wine to get the class prediction:")

# Input fields for wine features
features = []
for i in range(1, 14):
    feature = st.number_input(f"Feature {i}", value=0.0)
    features.append(feature)

# Predict button
if st.button("Predict"):
    prediction = predict(features)
    st.write(f"The predicted class is: {prediction}")
