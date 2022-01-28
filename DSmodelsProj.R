library(ggplot2)
library(boot)
library(dplyr)
library(randomForest)
library(MASS)
library(tree)
library(gbm)

#### Initialize data ----------------
data = read.csv("clean_kc_house_data.csv")
names(data1)
# Creating 
data["logPrice"] <- log(data["price"])
data["logsqft_lot"] <- log(data["sqft_lot"])
data["logsqft_lot15"] <- log(data["sqft_lot15"])
data["Age"]  = 2020 - data$yr_built
data["level"] = ifelse(data$price >= 645000, "High", "Low")
data$level = as.factor(data$level)
data1 = data[5:27]
#View(data)
#summary(data)
#write.csv(data, "clean_kc_house_data1.csv")

#### Linear Regression Model --------------
lm.fit1=lm(logPrice ~ bedrooms + bathrooms + logsqft_lot + sqft_living + floors + waterfront + 
             view + condition + grade + logsqft_lot15 + sqft_living15 + 
             sqft_living:sqft_living15 + sqft_lot:logsqft_lot15 + Age, data=data)
summary(lm.fit1)
mean(lm.fit1$residuals^2)

#This model was an R^2 of 0.6638 , and a MSE of 0.09

#### Classification Modeling -----------------------

#logarithmic model after doing backwards elimination
log.fit=glm(level ~ bedrooms + bathrooms + logsqft_lot + sqft_living + floors + waterfront + 
              view + condition + grade + logsqft_lot15 + sqft_living15 + 
              sqft_living:sqft_living15 + sqft_lot:logsqft_lot15 + Age, family = binomial, data=data)
summary(log.fit)

#Using the model to make predictions
predictLevel = predict(log.fit, type="response")
ConfusionMatrix = table(data$level, predictLevel >.7)
colnames(ConfusionMatrix) = c("Predicted Low Price","Predicted High Price")
rownames(ConfusionMatrix) = c("Actually Low Price","Actually High Price")
ConfusionMatrix
# We can also check the error of the model prediction by reversing the measure
1 - (4017+13897) / (4017+1800+959+13897)
# MCR =  0.1334591
#The model can accurately identify houses valued at a high price accuracy of 86% which is greater than our baseline linear model.

#### Validation set approach for linear regression (1) -------------------------------------------------------------------
  
# Create the train / test split of your data
# Note: you can determine how much data to be train and test
set.seed(1) 
train <- sample(1:nrow(data), 0.75*nrow(data))
test <- -train
trainingData = data[train,]
testingData = data[test,]
# Store the testing value in a separated column to measure the error rate and MSE
testing <- data$level[test] # for misclassification rate calculation
testing_outcome <- data$logPrice[test] # for MSE calculation
## Regression Problem: ##
# Train the training dataset with linear regression model
reg_fit <- lm(logPrice ~ bedrooms + bathrooms + logsqft_lot + sqft_living + floors + waterfront + 
                view + condition + grade + logsqft_lot15 + sqft_living15 + 
                sqft_living:sqft_living15 + sqft_lot:logsqft_lot15 + Age, data = trainingData)
# Predict the log price with the trained model using the testing dataset
reg_fit.pred <- predict(reg_fit, testingData)
# Calculate the testing MSE
mse <- mean((reg_fit.pred - testing_outcome)^2)
print(paste("The testing MSE from this model is", mse)) #0.0918

#### Cross Validation approach for logistic regression (2) -------------------------------------------------------------------

## Classification Problem: 
logit_fit <- glm(level ~ bedrooms + bathrooms + logsqft_lot + sqft_living + floors + waterfront + 
                   view + condition + grade + logsqft_lot15 + sqft_living15 + 
                   sqft_living:sqft_living15 + sqft_lot:logsqft_lot15 + Age,
                  family = binomial, data = trainingData)
# Predict the "high" or "low" class with the trained model using the testing dataset
probs <- predict(logit_fit, newdata = testingData, type = "response")
# Set the threshold for the classification
logit_fit.pred <- ifelse(probs > 0.50, "Low", "High")
# Create the confusion matrix
table(logit_fit.pred, testing)
# Calculate the misclassification error
mcr <- mean(logit_fit.pred != testing)
print(paste("The testing mis-classification rate from this model is", mcr)) #0.123

##  linear regression: ##
logit_fit <- glm(level ~ bedrooms + bathrooms + logsqft_lot + sqft_living + floors + waterfront + 
                   view + condition + grade + logsqft_lot15 + sqft_living15 + 
                   sqft_living:sqft_living15 + sqft_lot:logsqft_lot15 + Age,
                 family = binomial, data = trainingData)
# Predict the "high" or "low" class with the trained model using the testing dataset
probs <- predict(logit_fit, newdata = testingData, type = "response")
# Set the threshold for the classification
logit_fit.pred <- ifelse(probs > 0.50, "Low", "High")
# Create the confusion matrix
table(logit_fit.pred, testing)
# Calculate the misclassification error
mcr <- mean(logit_fit.pred != testing)
print(paste("The testing mis-classification rate from this model is", mcr)) #0.123


#### LOOCV for linear regression (2) -------------------------------------------------------------------

lm.fitCV <- lm(logPrice ~ bedrooms + bathrooms + logsqft_lot + sqft_living + floors + waterfront + 
                view + condition + grade + logsqft_lot15 + sqft_living15 + 
                sqft_living:sqft_living15 + sqft_lot:logsqft_lot15 + Age, data = data)


cv.glm(data,lm.fitCV)$delta 

loocv = function(fit){
  h=lm.influence(fit)$h
  mean((residuals(fit)/(1-h))^2)
}

# Now we use the loocv() function to perform the cross validation
cv = loocv(lm.fitCV)
cv
#### Decision Tree Model (Regression / Classification (5)) -------------------------------------------------------------------
# FOR CLASSIFICATION
n = nrow(data1)
n # that is 20673
tn = .7*n
train = sample(1:n, tn)
test = -train
trainData = data1[train,]
testData  = data1[test,]
testOutcome = data1$level[test]

# Fitting a Classification Tree Model
treeFit <- tree(level~ bedrooms + bathrooms + logsqft_lot + sqft_living + floors + waterfront + 
                  view + condition + grade + logsqft_lot15 + sqft_living15 + Age, data=trainData)
summary(treeFit)

# Plot the tree diagram
plot(treeFit)
text(treeFit, pretty=0)

# See each of the split from the tree model
treeFit

# Check the test error rate
treePred <- predict(treeFit, testData, type="class")
table(treePred,testOutcome)
mean(treePred != testOutcome)  # 0.143

# FOR REGRESSION
set.seed(1) 
n = nrow(data1)
n # that is 20673
tn = .7*n
train = sample(1:n, tn)
test = -train
trainData = data1[train,]
testData  = data1[test,]
testOutcome = data1$logPrice[test]

# Fitting a Regression Tree Model
treeFit <- tree(logPrice~ bedrooms + bathrooms + logsqft_lot + sqft_living + floors + waterfront + 
                  view + condition + grade + logsqft_lot15 + sqft_living15 + Age, data=trainData)
summary(treeFit)

# Plot the tree diagram
plot(treeFit)
text(treeFit)

# See each of the split from the tree model
treeFit

# Check the test error rate
treePred <- predict(treeFit, testData)
table(treePred,testOutcome)
mean((treePred - testOutcome)^2)  # 0.127


#### Bagging Tree Model (Regression / Classification) -------------------------------------------------------------------
dim(data1)
View(data1)
data3 = data1[-c(4, 10,11,12,13,14,15,16,18,23)]
View(testOutcome)
## Regression Problem: ##
set.seed(1) 
nrow(testOutcome)
n = nrow(data3)
n # that is 20673
tn = .7*n
train = sample(1:n, tn)
test = -train
trainData = data3[train,]
testData  = data3[test,]
testOutcome = data3$logPrice[test]
# Bagging is simply a special case of random forest with m = 12 (All predictors are used to grow trees)
bag.data = randomForest(logPrice~., data=trainData, mtry=12, importance=TRUE)
# mtry is the number of variables randomly sampled as candidates at each spit. 

bag.data
names(bag.data)
summary(bag.data)

# Using the importance() function to check the importance of each variable
importance(bag.data)
# Two measures are used to evaluate the importance of a variable,
# First, we can check the mean decrease of accuracy in prediction on the out of bag samples 
# when the variable is excluded from the model. (%IncMSE)
# Secondly, checking the total decrease in node impurity that result from splits over that variable


# We can plot these importance measures with the varImpPlot() function
varImpPlot(bag.data)
nrow(predict.bag)
# Make prediction with the trained model by passing in the testing dataset
predict.bag = predict(bag.data, newdata=testData)
View(predict.bag)
nrow(testOutcome)
# Plot the prediction and testing outcome
plot(predict.bag,testOutcome)
abline(0,1)

# Calculate the MSE for the bagging model
MSE.bag = mean((predict.bag - testOutcome )^2)
MSE.bag

#### Random Forest -------------------------------------------------------------------

# FOR REGRESSION
## Regression Problem: ##
set.seed(1) 
n = nrow(data1)
n # that is 20673
tn = .7*n
train = sample(1:n, tn)
test = -train
trainData = data1[train,]
testData  = data1[test,]
testOutcome = data1$logPrice[test]

# In random Forest we only need to change the mtry so that we get the minimum MSE, 
# so, let's try different size of mtry - from 1 to 22
dim(data1)
# Using a for loop to run different size of mtry (number of predictors to be consider for each split of the tree)
MSE.Rf=rep(0,22)
for(d in 1:22){ 
  rf.data = randomForest(logPrice~., data=trainData, mtry=d, importance=TRUE)
  predict.rf = predict(rf.data,newdata = testData)
  MSE.Rf[d] = mean((predict.rf- testOutcome )^2)
}
MTRY = c(1:22)

# Plot the MSE for each size of ntry
plot(MTRY,MSE.Rf,type="b",col="red")
min(MSE.Rf)
data.frame(MTRY, MSE.Rf)

# mtry=9 created the minimum error
rf.data = randomForest(~., data=trainData, mtry=9, importance=TRUE)

# Getting the prediction from the best model
predict.rf = predict(rf.data, newdata=testData)

# Calculate the MSE from the best model
MSE.Rf = mean((predict.rf- testOutcome )^2)
MSE.Rf

# Check the importance measures
importance(rf.data)

# Plot the importance measures
varImpPlot(rf.data)


#### Boosting -------------------------------------------------------------------
# If you are running regression problems then use distribution = "gaussian". If you are working on 
# The default value of Lambda is 0.001 (learning rate of the tree model)
boost.data3 = gbm(logPrice~., data=data3, distribution="gaussian", n.trees=5000, interaction.depth=4)
boost.data3
summary(boost.data3)
summary(data3$logPrice)


plot(boost.data3,i="grade")
plot(boost.data3,i="sqft_living")

# Get the prediction from the boosting model
predict.boost = predict(boost.data3, newdata=testData, n.trees=5000)

# Calculate the MSE of the boosting model
mean((predict.boost-testOutcome )^2)
plot(predict.boost,testOutcome)
abline(0,1)

# Lets test out some lambdas
Lambda = c(.00001,0.0001,0.001,.01,0.1,.15,.2)
Counter = 1
MSE.Boost = rep(0,7)

# Using a for loop to check different values for Lamda
for(d in Lambda){
  boost.data3 = gbm(logPrice~., data=trainData, distribution="gaussian", n.trees=5000, interaction.depth=4, shrinkage=d)
  predict.boost = predict(boost.data3, newdata=testData, n.trees=5000)
  MSE.Boost[Counter] = mean((predict.boost- testOutcome )^2)
  Counter = Counter + 1
}
MSE.Boost
# The min happened at  = 0.01
plot(Lambda,MSE.Boost,type="b",col="red") 

min(MSE.Boost) 
data.frame(Lambda, MSE.Boost)

# Now let's fix Lambda and change size of the tree
TreeSize = c(50,100,200,400,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000)
Counter = 1
MSE.Boost.Tree = rep(0,15)

# Create a for loop to check different tree size with the best Lamda selected (shrinkage=0.01)
for(d in TreeSize){
  boost.data3 = gbm(logPrice~., data=trainData, distribution="gaussian", n.trees=d, interaction.depth=4, shrinkage=0.01)
  predict.boost = predict(boost.data3, newdata=testData, n.trees=d)
  MSE.Boost.Tree[Counter] = mean((predict.boost- testOutcome )^2)
  Counter = Counter + 1
}

# The tree size reaches miminum at 10000
plot(TreeSize,MSE.Boost.Tree,type="b",col="red")

# it seems like 4000 was a very good choice
min(MSE.Boost.Tree) 
data.frame(TreeSize,MSE.Boost.Tree)


###
depthSize = c(1,2,3,4,5,10,15,20)
Counter = 1
MSE.Boost.Tree = rep(0,8)
#interation.depth
for(d in depthSize){
  boost.data3 = gbm(logPrice~., data=trainData, distribution="gaussian", n.trees=10000, interaction.depth=d, shrinkage=0.01)
  predict.boost = predict(boost.data3, newdata=testData, n.trees=d)
  MSE.Boost.Tree[Counter] = mean((predict.boost- testOutcome )^2)
  Counter = Counter + 1
}
# The tree size reaches miminum at 10000
plot(depthSize,MSE.Boost.Tree,type="b",col="red")

# it seems like 4000 was a very good choice
min(MSE.Boost.Tree) 
data.frame(TreeSize,MSE.Boost.Tree)
