import pyspark
from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Vectors
from pyspark.sql import functions as f
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.feature import PCA
from pyspark.ml.feature import Normalizer
from pyspark.ml.tuning import CrossValidator
import pandas as pd

#create the spark context and disable warning messages
spark = SparkSession.builder.appName('wineQuality-ml').getOrCreate()
spark.sparkContext.setLogLevel("OFF")

# read the .csv file and convert to dataframe then split the data into testing and training sets
df = spark.read.csv('winequality-white.csv', header=True, inferSchema=True, sep=';')
trainingData, testingData = df.withColumn("binary", f.when(f.col("quality") > 5, 1).otherwise(0)).randomSplit([0.8,0.2])

# get a list of the feartures from the datafram for use in building a vecotor of features separate of labels 
featureCols = [cols for cols in df.columns if cols != "quality"]
assembler = VectorAssembler(inputCols=featureCols, outputCol="features")
normalizer = Normalizer(inputCol="features", outputCol="normFeatures")


# set up estimators to build the models/pipelines
lr = LinearRegression(maxIter=100, featuresCol="normFeatures", labelCol="quality")
rf = RandomForestRegressor(featuresCol="normFeatures", labelCol="quality", numTrees=100, maxBins=128, maxDepth=20, minInstancesPerNode=5, seed=33)
logr = LogisticRegression(featuresCol = "normFeatures", labelCol="quality", maxIter=100)
gbt = GBTClassifier(featuresCol="normFeatures", labelCol="binary")

# set up the pipelines
lrPipeline = Pipeline(stages=[assembler, normalizer, lr])
rfPipeline = Pipeline(stages=[assembler, normalizer, rf])
logrPipeline = Pipeline(stages=[assembler, normalizer, logr])
gbtPipeline = Pipeline(stages=[assembler, normalizer, gbt])

# set up the evaluators to check the quality of the models
rmseEval = RegressionEvaluator(labelCol="quality", predictionCol="prediction", metricName="rmse")
r2Eval = RegressionEvaluator(labelCol="quality", predictionCol="prediction", metricName="r2")
f1Eval = MulticlassClassificationEvaluator(labelCol="quality", metricName="f1")
preEval = MulticlassClassificationEvaluator(labelCol="quality", metricName="weightedPrecision")
gbtF1Eval = MulticlassClassificationEvaluator(labelCol="binary", metricName="f1")
gbtPreEval = MulticlassClassificationEvaluator(labelCol="binary", metricName="weightedPrecision")

# set up the linear regression model for cross validation against regParm and elasticNetParam
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.0, 0.3, 0.6, 0.9]).addGrid(lr.elasticNetParam, [0.2, 0.4, 0.6, 0.9]).build()
cvRmse = CrossValidator(estimator = lrPipeline, estimatorParamMaps = paramGrid, evaluator = rmseEval, numFolds = 5)
cvR2 = CrossValidator(estimator = lrPipeline, estimatorParamMaps = paramGrid, evaluator = r2Eval, numFolds = 4)

# set up the logistic regression model for cross validation against regParam and elasticNetParam
logParamGrid = ParamGridBuilder().addGrid(logr.regParam, [0.0, 0.3, 0.6, 0.9]).addGrid(logr.elasticNetParam, [0.2, 0.4, 0.6, 0.8]).build()
cvF1 = CrossValidator(estimator = logrPipeline, estimatorParamMaps = logParamGrid, evaluator = f1Eval, numFolds = 5)
cvPre = CrossValidator(estimator = logrPipeline, estimatorParamMaps = logParamGrid, evaluator = preEval,numFolds = 5)

# training the models
rmseModel = cvRmse.fit(trainingData)
r2Model = cvR2.fit(trainingData)
rfModel = rfPipeline.fit(trainingData)
f1Model = cvF1.fit(trainingData)
preModel = cvPre.fit(trainingData)
gbtModel = gbtPipeline.fit(trainingData)

# testing the models
rmsePredictions = rmseModel.transform(testingData)
r2Predictions = r2Model.transform(testingData)
rfPredictions = rfModel.transform(testingData)
f1Predictions = f1Model.transform(testingData)
prePredictions = preModel.transform(testingData)
gbtPredictions = gbtModel.transform(testingData)

# print the resuts to screen
print("\n\n\n\n") # adding space to separate results from everything before it. 
print("The rmse of the linear regression model with cross validation using the rmse evaluator: " + str(rmseEval.evaluate(rmsePredictions)))
print("\nThe r2 of the linear regression model with cross validation using the rmse evaluator: " + str(r2Eval.evaluate(rmsePredictions)))
print("\nThe rmse of the linear regression model with cross validation using the r2 evaluator: " + str(rmseEval.evaluate(r2Predictions)))
print("\nThe r2 of the linear regression model with cross validation using the r2 evaluator: " + str(r2Eval.evaluate(r2Predictions)))
print("\nThe rmse of the random forest model: " + str(rmseEval.evaluate(rfPredictions)))
print("\nThe r2 of the random forest model: " + str(r2Eval.evaluate(rfPredictions)))
print("\nThe f1 of the logistic regression model with cross validation using f1 evaluator: " + str(f1Eval.evaluate(f1Predictions)))
print("\nThe weighted precision of the logistic regression model with cross validation using weighted precision evaluator: " + str(preEval.evaluate(prePredictions)))
print("\nThe f1 of the gradient-boosted tree model using the f1 evaluator: " + str(gbtF1Eval.evaluate(gbtPredictions)))
print("\nThe weighted precision of the Gradient-Boosted Tree model using weighted precision evaluator: " + str(gbtPreEval.evaluate(gbtPredictions)))
