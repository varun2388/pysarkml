from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Initialize Spark session
spark = SparkSession.builder.appName("WineClassification").getOrCreate()

# Load the wine dataset
data = spark.read.csv("wine.csv", header=True, inferSchema=True)

# Data preprocessing
indexer = StringIndexer(inputCol="class", outputCol="label")
assembler = VectorAssembler(inputCols=data.columns[1:], outputCol="features")

# Split the data into training and test sets
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Initialize the classifier
rf = RandomForestClassifier(labelCol="label", featuresCol="features")

# Create a pipeline
pipeline = Pipeline(stages=[indexer, assembler, rf])

# Train the model
model = pipeline.fit(trainingData)

# Make predictions
predictions = model.transform(testData)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy = {accuracy}")

# Save the model
model.save("wine_classification_model")
