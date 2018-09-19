# Load library
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)

# Set working directory
setwd("C:/Users/david.LIZZY/Documents/GitHub/Human-Activity-Recognition-Analysis")

# Load csv
trainRaw <- read.csv("raw_data/pml-training.csv")
testRaw <- read.csv("raw_data/pml-training.csv")
dim(trainRaw)
dim(testRaw)

# Drop NA
trainRaw <- trainRaw[, colSums(is.na(trainRaw)) == 0] 
testRaw <- testRaw[, colSums(is.na(testRaw)) == 0]

# Clean data
classe <- trainRaw$classe
trainRemove <- grepl("^X|timestamp|window", names(trainRaw))
trainRaw <- trainRaw[, !trainRemove]
trainCleaned <- trainRaw[, sapply(trainRaw, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(testRaw))
testRaw <- testRaw[, !testRemove]
testCleaned <- testRaw[, sapply(testRaw, is.numeric)]

# Slice data
set.seed(22519) # For reproducibile purpose
inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
trainData <- trainCleaned[inTrain, ]
testData <- trainCleaned[-inTrain, ]

# Modeling
controlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=trainData, method="rf", trControl=controlRf, ntree=250)
modelRf

predictRf <- predict(modelRf, testData)
confusionMatrix(testData$classe, predictRf)

accuracy <- postResample(predictRf, testData$classe)
accuracy
oose <- 1 - as.numeric(confusionMatrix(testData$classe, predictRf)$overall[1])
oose

# Predicting
result <- predict(modelRf, testCleaned[, -length(names(testCleaned))])
result

# Correlation Matrix Visualization
corrPlot <- cor(trainData[, -length(names(trainData))])
corrplot(corrPlot, method="color")
         
# Decision Tree Visualization
treeModel <- rpart(classe ~ ., data=trainData, method="class")
prp(treeModel) # fast plot