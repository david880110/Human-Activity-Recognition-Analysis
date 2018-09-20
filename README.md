![built with R](https://img.shields.io/badge/built%20with-R-blue.svg)    

# Human-Activity-Recognition-Analysis

![alt text](https://raw.githubusercontent.com/david880110/Human-Activity-Recognition-Analysis/master/image/under-armour-connected-fitness-666x333.jpg)

<p align="center">
  • <a href="#findings">Findings</a>
  • <a href="#syntax-detail">Project Detail</a>
  • <a href="#technology-Used">Technology Used</a>
</p>

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. Devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement, aim to find patterns in individual's behavior. The goal of this project is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.
## Data Source

<img src="https://raw.githubusercontent.com/david880110/Human-Activity-Recognition-Analysis/master/image/wayback-machine-logo.jpg" width="240" height="80"/>

The [data](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har.) for this project come from "http://web.archive.org"， licensed under the Creative Commons license (CC BY-SA).
## Findings 

## Project Detail

### How the model was built

3 Machine Learning models was used to compared anc choose the highest output accuracy
* Decision Trees
* Random Forest
* Generalized Boosted Regression

### How cross validation is used
The cleaned training set is splitted into a pure training data set (60%) and a validation data set (40%).

```R
inTrain <- createDataPartition(training$classe, p=0.6, list=FALSE)
myTraining <- training[inTrain, ]
myTesting <- training[-inTrain, ]
dim(myTraining); dim(myTesting)
```



### What is the expected out of sample error
The expected out-of-sample error is 0.11% (1-99.89%)

### why the choices was made

The Random Forest model gave an accuracy of testing dataset of 99.89%, which was more accurate than that of from the Decision Trees or GBM.

|          | Decision Trees | Random Forest | Generalized Boosted Regression |
|:--------:|:--------------:|:-------------:|:------------------------------:|
| Accuracy |     0.8789     |     0.9986    |             0.9966             |

## Technology Used

<img src="https://raw.githubusercontent.com/david880110/tech-logo/master/R_logo.svg.png" width="180" height="90"/>

© David Gu 2018 All Rights reserved.
