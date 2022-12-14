
################################################################################
##############################  Events 2020  ###################################
################################################################################
################################################################################
### Nonlinear Classification
# 1 Nonlinear Discriminant Analysis
# 2 Neural Networks
# 3 Flexible Discriminant Analysis
# 4 Support Vector Machines
# 5 K-Nearest Neighbors
# 6 Naive Bayes

library(caret)
library(doMC)
library(kernlab)
library(klaR)
library(lattice)
library(latticeExtra)
library(MASS)
library(mda)
library(nnet)
library(pROC)


# Import the dataset
NHL_df_event = read.csv("C:/Users/m29336/Documents/NHL_df_event.csv", header=TRUE)
dim(NHL_df_event)        # 78582   88

NHL_df_event$event = as.factor(NHL_df_event$event) #change to factor
class(NHL_df_event$event)

#set Control
ctrl = trainControl(method = "cv", repeats = 5, summaryFunction=multiClassSummary, classProbs=TRUE )
ctrl1 <- trainControl(method = "repeatedcv", repeats = 5)
ctrl2 <- trainControl(method = "LGOCV", summaryFunction = multiClassSummary, classProbs = TRUE, savePredictions = TRUE)

## Split the data into training (80%) and test sets (20%)
set.seed(123)
inTrain_event <- createDataPartition(NHL_df_event$event, p = .8)[[1]]
Train.NHL_df_event <- NHL_df_event[ inTrain_event, ]
Test.NHL_df_event  <- NHL_df_event[-inTrain_event, ]


################################################################################
### 1 Nonlinear Discriminant Analysis

set.seed(123)
mdaFit <- train(event ~ .,
                data = Train.NHL_df_event,
                method = "mda",
                trControl = ctrl2)
mdaFit

# Mixture Discriminant Analysis
#
# 62866 samples
# 88 predictor
# 2 classes: 'Goal', 'NoGoal'
#
# No pre-processing
# Resampling: Cross-Validated (10 fold)
# Summary of sample sizes: 56579, 56579, 56580, 56580, 56580, 56579, ...
# Resampling results across tuning parameters:
#  
#   subclasses  ROC        Sens       Spec    
# 1           0.9957430  0.9995439  0.9739758
# 2           0.9954317  0.9995439  0.9739587
# 3           0.9952409  0.9993156  0.9739758
# 4           0.9944015  0.9979462  0.9733261
# 5           0.9936128  0.9984034  0.9703850
# 6           0.9921094  0.9977179  0.9675638
# 7           0.9911692  0.9970340  0.9644007
# 8           0.9893673  0.9906528  0.9680424
#
# ROC was used to select the optimal model using the largest value.
# The final value used for the model was subclasses = 1.

### Predict the test set
goal.Results_mdaFit <- data.frame(obs = Test.NHL_df$goal)
goal.Results_mdaFit$prob <- predict(mdaFit, Test.NHL_df, type = "prob")[, "Goal"]
goal.Results_mdaFit$pred <- predict(mdaFit, Test.NHL_df)
goal.Results_mdaFit$Label <- ifelse(goal.Results_mdaFit$obs == "Goal",
                                    "True Outcome: goal",
                                    "True Outcome: no goal")

goal.Results_mdaFit$obs <- as.factor(goal.Results_mdaFit$obs)

defaultSummary(goal.Results_mdaFit)

# Accuracy     Kappa
# 0.9736574 0.8269232

### Create the confusion matrix from the test set.
confusionMatrix(data = goal.Results_mdaFit$pred,
                reference = goal.Results_mdaFit$obs)

# Confusion Matrix and Statistics
#
# Reference
# Prediction  Goal NoGoal
# Goal    1094    413
# NoGoal     1  14208
#
# Accuracy : 0.9737        
# 95% CI : (0.971, 0.9761)
# No Information Rate : 0.9303        
# P-Value [Acc > NIR] : < 2.2e-16      
#
# Kappa : 0.8269        
#
# Mcnemar's Test P-Value : < 2.2e-16      
#                                          
#             Sensitivity : 0.99909        
#             Specificity : 0.97175        
#          Pos Pred Value : 0.72595        
#          Neg Pred Value : 0.99993        
#              Prevalence : 0.06967        
#          Detection Rate : 0.06961        
#    Detection Prevalence : 0.09589        
#       Balanced Accuracy : 0.98542        
#                                          
#        'Positive' Class : Goal        

### ROC curves:
goal.ROC_mdaFit <- roc(response = goal.Results_mdaFit$obs, predictor = goal.Results_mdaFit$prob, levels = levels(goal.Results_mdaFit$obs))
coords(goal.ROC_mdaFit, "all")[,1:3]

auc(goal.ROC_mdaFit) # Area under the curve: 0.9957
ci.auc(goal.ROC_mdaFit) # 95% CI: 0.9947-0.9967 (DeLong)

### Note the x-axis is reversed
plot(goal.ROC_mdaFit)

mda=list( classifier=mdaFit, roc=goal.ROC_mdaFit, auc=auc(goal.ROC_mdaFit) )
mda


# ################################################################################
# ### 2 Neural Networks
# 
# nnetGrid <- expand.grid(size = 1:10, decay = c(0, .1, 1, 2))
# maxSize <- max(nnetGrid$size)
# 
# set.seed(123)
# nnetFit <- train(goal ~ .,
#                  data = Train.NHL_df,
#                  method = "nnet",
#                  preProc=c("center","scale","spatialSign"),
#                  metric = "ROC",
#                  tuneGrid = nnetGrid,
#                  trace = FALSE,
#                  maxit = 2000,
#                  trControl = ctrl)
# nnetFit
# 
# # [insert output]
# #
# #
# #
# #
# 
# 
# ### Predict the test set
# goal.Results_nnetFit <- data.frame(obs = Test.NHL_df$goal)
# goal.Results_nnetFit$prob <- predict(nnetFit, Test.NHL_df, type = "prob")[, "Goal"]
# goal.Results_nnetFit$pred <- predict(nnetFit, Test.NHL_df)
# goal.Results_nnetFit$Label <- ifelse(goal.Results_nnetFit$obs == "Goal", 
#                                     "True Outcome: goal", 
#                                     "True Outcome: no goal")
# 
# goal.Results_nnetFit$obs <- as.factor(goal.Results_nnetFit$obs)
# 
# defaultSummary(goal.Results_nnetFit)
# 
# # [insert output]
# #
# #
# #
# #
# 
# ### Create the confusion matrix from the test set.
# confusionMatrix(data = goal.Results_nnetFit$pred, 
#                 reference = goal.Results_nnetFit$obs)
# 
# # [insert output]
# #
# #
# #
# #
# 
# ### ROC curves:
# goal.ROC_nnetFit <- roc(response = goal.Results_nnetFit$obs, predictor = goal.Results_nnetFit$prob, levels = levels(goal.Results_nnetFit$obs))
# coords(goal.ROC_nnetFit, "all")[,1:3]
# 
# auc(goal.ROC_nnetFit) # [insert output]
# ci.auc(goal.ROC_nnetFit) # [insert output]
# 
# ### Plot ROC
# plot(goal.ROC_nnetFit1, type = "s", legacy.axes = TRUE)
# 
# # [save plot]
# #
# #
# #
# #
# 
# 
# set.seed(123)
# nnetFit2 <- train(goal ~ .,
#                   data = Train.NHL_df,
#                   method = "avNNet",
#                   metric = "ROC",
#                   tuneGrid = nnetGrid,
#                   repeats = 10,
#                   trace = FALSE,
#                   maxit = 2000,
#                   allowParallel = FALSE, ## this will cause to many workers to be launched.
#                   trControl = ctrl)
# 
# nnetFit2
# 
# ### Predict the test set
# goal.Results_nnetFit2 <- data.frame(obs = Test.NHL_df$goal)
# goal.Results_nnetFit2$prob <- predict(nnetFit2, Test.NHL_df, type = "prob")[, "Goal"]
# goal.Results_nnetFit2$pred <- predict(nnetFit2, Test.NHL_df)
# goal.Results_nnetFit2$Label <- ifelse(goal.Results_nnetFit2$obs == "Goal", 
#                                      "True Outcome: goal", 
#                                      "True Outcome: no goal")
# 
# goal.Results_nnetFit2$obs <- as.factor(goal.Results_nnetFit2$obs)
# 
# defaultSummary(goal.Results_nnetFit2)
# 
# # [insert output]
# #
# #
# #
# #
# 
# ### Create the confusion matrix from the test set.
# confusionMatrix(data = goal.Results_nnetFit2$pred, 
#                 reference = goal.Results_nnetFit2$obs)
# 
# # [insert output]
# #
# #
# #
# #
# 
# ### ROC curves:
# goal.ROC_nnetFit2 <- roc(response = goal.Results_nnetFit2$obs, predictor = goal.Results_nnetFit2$prob, levels = levels(goal.Results_nnetFit2$obs))
# coords(goal.ROC_nnetFit2, "all")[,1:3]
# 
# auc(goal.ROC_nnetFit2) # [insert output]
# ci.auc(goal.ROC_nnetFit2) # [insert output]
# 
# ### Plot ROC
# plot(goal.ROC_nnetFit2, type = "s", legacy.axes = TRUE)
# 
# # [save plot]
# #
# #
# #
# #
# 
# nnet1 <- nnetFit$results
# nnet1$Transform <- "No Transformation"
# nnet1$Model <- "Single Model"
# 
# nnet2 <- nnetFit2$results
# nnet2$Transform <- "No Transformation"
# nnet2$Model <- "Model Averaging"
# nnet2$bag <- NULL
# 
# nnetResults <- rbind(nnet1,nnet2)
# nnetResults$Model <- factor(as.character(nnetResults$Model),
#                             levels = c("Single Model", "Model Averaging"))
# 
# useOuterStrips(
#   xyplot(ROC ~ size|Model*Transform,
#          data = nnetResults,
#          groups = decay,
#          as.table = TRUE,
#          type = c("p", "l", "g"),
#          lty = 1,
#          ylab = "ROC AUC",
#          xlab = "Number of Hidden Units",
#          auto.key = list(columns = 2, 
#                          title = "Weight Decay", 
#                          cex.title = 1)))
# 
# 
# nnet1=list( classifier=nnetFit, roc=goal.ROC_nnetFit, auc=auc(goal.ROC_nnetFit) )
# nnet1
# 
# nnet2=list( classifier=nnetFit2, roc=goal.ROC_nnetFit2, auc=auc(goal.ROC_nnetFit2) )
# nnet2


################################################################################
### 3 Flexible Discriminant Analysis

set.seed(123)
fdaFit <- train(event ~ .,
                data = Train.NHL_df_event,
                method = "fda",
                tuneGrid = expand.grid(degree = 1, nprune = 2:25),
                trControl = ctrl)
fdaFit

# Flexible Discriminant Analysis 
# 
# 62866 samples
# 88 predictor
# 2 classes: 'Goal', 'NoGoal' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold) 
# Summary of sample sizes: 56579, 56579, 56580, 56580, 56580, 56579, ... 
# Resampling results across tuning parameters:
#   
# #   nprune  ROC        Sens       Spec     
# 2      0.6993662  0.0000000  1.0000000
# 3      0.8680747  0.0000000  1.0000000
# 4      0.9584751  0.9995439  0.9173792
# 5      0.9868261  0.9995439  0.9739758
# 6      0.9924031  0.9995439  0.9739758
# 7      0.9954715  0.9995439  0.9739758***
# 8      0.9954715  0.9995439  0.9739758
# 9      0.9954715  0.9995439  0.9739758
# 10      0.9954715  0.9995439  0.9739758
# 11      0.9954715  0.9995439  0.9739758
# 12      0.9954715  0.9995439  0.9739758
# 13      0.9954715  0.9995439  0.9739758
# 14      0.9954715  0.9995439  0.9739758
# 15      0.9954715  0.9995439  0.9739758
# 16      0.9954715  0.9995439  0.9739758
# 17      0.9954715  0.9995439  0.9739758
# 18      0.9954715  0.9995439  0.9739758
# 19      0.9954715  0.9995439  0.9739758
# 20      0.9954715  0.9995439  0.9739758
# 21      0.9954715  0.9995439  0.9739758
# 22      0.9954715  0.9995439  0.9739758
# 23      0.9954715  0.9995439  0.9739758
# 24      0.9954715  0.9995439  0.9739758
# 25      0.9954715  0.9995439  0.9739758
# 
# Tuning parameter 'degree' was held constant at a value of 1
# ROC was used to select the optimal model using the largest value.
# The final values used for the model were degree = 1 and nprune = 7.


### Predict the test set
goal.Results_fdaFit <- data.frame(obs = Test.NHL_df$goal)
goal.Results_fdaFit$prob <- predict(fdaFit, Test.NHL_df, type = "prob")[, "Goal"]
goal.Results_fdaFit$pred <- predict(fdaFit, Test.NHL_df)
goal.Results_fdaFit$Label <- ifelse(goal.Results_fdaFit$obs == "Goal", 
                                     "True Outcome: goal", 
                                     "True Outcome: no goal")

goal.Results_fdaFit$obs <- as.factor(goal.Results_fdaFit$obs)

defaultSummary(goal.Results_fdaFit)

# Accuracy     Kappa 
# 0.9736574 0.8269232 

### Create the confusion matrix from the test set.
confusionMatrix(data = goal.Results_fdaFit$pred, 
                reference = goal.Results_fdaFit$obs)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  Goal NoGoal
# Goal    1094    413
# NoGoal     1  14208
# 
# Accuracy : 0.9737         
# 95% CI : (0.971, 0.9761)
# No Information Rate : 0.9303         
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.8269         
# 
# Mcnemar's Test P-Value : < 2.2e-16      
#                                          
#             Sensitivity : 0.99909        
#             Specificity : 0.97175        
#          Pos Pred Value : 0.72595        
#          Neg Pred Value : 0.99993        
#              Prevalence : 0.06967        
#          Detection Rate : 0.06961        
#    Detection Prevalence : 0.09589        
#       Balanced Accuracy : 0.98542        
#                                          
#        'Positive' Class : Goal 

### ROC curves:
goal.ROC_fdaFit <- roc(response = goal.Results_fdaFit$obs, predictor = goal.Results_fdaFit$prob, levels = levels(goal.Results_fdaFit$obs))
coords(goal.ROC_fdaFit, "all")[,1:3]

auc(goal.ROC_fdaFit) # Area under the curve: 0.9953
ci.auc(goal.ROC_fdaFit) # 95% CI: 0.9944-0.9961 (DeLong)

### Plot ROC
plot(goal.ROC_fdaFit, type = "s", legacy.axes = TRUE)

fda=list( classifier=fdaFit, roc=goal.ROC_fdaFit, auc=auc(goal.ROC_fdaFit) )
fda


# ################################################################################
# ### 4 Support Vector Machines
# 
# set.seed(123)
# sigmaRange <- sigest(as.matrix(Train.NHL_df))
# svmRGrid <- expand.grid(sigma = as.vector(sigmaRangeFull)[1], C = 2^(-3:4))
# 
# set.seed(123)
# svmRFit <- train(goal ~ .,
#                  data = Train.NHL_df,
#                  method = "svmRadial",
#                  metric = "ROC",
#                  tuneGrid = svmRGrid,
#                  #fit=FALSE,
#                  trControl = ctrl)
# #if error, try adding "fit=FALSE,"
# 
# svmRFit
# 
# # [insert output]
# #
# #
# #
# #
# 
# 
# ### Predict the test set
# goal.Results_svmRFit <- data.frame(obs = Test.NHL_df$goal)
# goal.Results_svmRFit$prob <- predict(svmRFit, Test.NHL_df, type = "prob")[, "Goal"]
# goal.Results_svmRFit$pred <- predict(svmRFit, Test.NHL_df)
# goal.Results_svmRFit$Label <- ifelse(goal.Results_svmRFit$obs == "Goal", 
#                                     "True Outcome: goal", 
#                                     "True Outcome: no goal")
# 
# goal.Results_svmRFit$obs <- as.factor(goal.Results_svmRFit$obs)
# 
# defaultSummary(goal.Results_svmRFit)
# 
# # [insert output]
# #
# #
# #
# #
# 
# ### Create the confusion matrix from the test set.
# confusionMatrix(data = goal.Results_svmRFit$pred, 
#                 reference = goal.Results_svmRFit$obs)
# 
# # [insert output]
# #
# #
# #
# #
# 
# ### ROC curves:
# goal.ROC_svmRFit <- roc(response = goal.Results_svmRFit$obs, predictor = goal.Results_svmRFit$prob, levels = levels(goal.Results_svmRFit$obs))
# coords(goal.ROC_svmRFit, "all")[,1:3]
# 
# auc(goal.ROC_svmRFit) # [insert output]
# ci.auc(goal.ROC_svmRFit) # [insert output]
# 
# ### Plot ROC
# plot(goal.ROC_svmRFit, type = "s", legacy.axes = TRUE)
# 
# # [save plot]
# #
# #
# #
# #
# 
# svm=list( classifier=svmFit, roc=goal.ROC_svmFit, auc=auc(goal.ROC_svmFit) )
# svm
# 
# ################################################################################
# ### 5 K-Nearest Neighbors
# 
# set.seed(123)
# knnFit <- train(event ~ .,
#                 data = Train.NHL_df_event,
#                 method = "knn",
#                 tuneLength=20,
#                 trControl = ctrl)
# knnFit
# 
# # [insert output]
# #
# #
# #
# #
# 
# 
# ### Predict the test set
# goal.Results_knnFit <- data.frame(obs = Test.NHL_df$goal)
# goal.Results_knnFit$prob <- predict(knnFit, Test.NHL_df, type = "prob")[, "Goal"]
# goal.Results_knnFit$pred <- predict(knnFit, Test.NHL_df)
# goal.Results_knnFit$Label <- ifelse(goal.Results_knnFit$obs == "Goal", 
#                                      "True Outcome: goal", 
#                                      "True Outcome: no goal")
# 
# goal.Results_knnFit$obs <- as.factor(goal.Results_knnFit$obs)
# 
# defaultSummary(goal.Results_knnFit)
# 
# # [insert output]
# #
# #
# #
# #
# 
# ### Create the confusion matrix from the test set.
# confusionMatrix(data = goal.Results_knnFit$pred, 
#                 reference = goal.Results_knnFit$obs)
# 
# # [insert output]
# #
# #
# #
# #
# 
# ### ROC curves:
# goal.ROC_knnFit <- roc(response = goal.Results_knnFit$obs, predictor = goal.Results_knnFit$prob, levels = levels(goal.Results_knnFit$obs))
# coords(goal.ROC_knnFit, "all")[,1:3]
# 
# auc(goal.ROC_knnFit) # [insert output]
# ci.auc(goal.ROC_knnFit) # [insert output]
# 
# ### Plot ROC
# plot(goal.ROC_knnFit, type = "s", legacy.axes = TRUE)
# 
# # [save plot]
# #
# #
# #
# #
# 
# knn=list( classifier=knnFit, roc=goal.ROC_knnFit, auc=auc(goal.ROC_knnFit) )
# knn

################################################################################
### 6 Naive Bayes

set.seed(123)
nBayesFit <- train(event ~ .,
                   data = Train.NHL_df_event,
                   method = "nb",
                   trControl = ctrl)
nBayesFit

# Naive Bayes 
# 
# 62867 samples
# 87 predictor
# 3 classes: 'GOAL', 'MISS', 'SHOT' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold) 
# Summary of sample sizes: 56580, 56580, 56580, 56580, 56582, 56580, ... 
# Resampling results across tuning parameters:
#   
#   usekernel  logLoss   AUC        prAUC      Accuracy   Kappa       Mean_F1    Mean_Sensitivity  Mean_Specificity  Mean_Pos_Pred_Value  Mean_Neg_Pred_Value  Mean_Precision  Mean_Recall  Mean_Detection_Rate
# FALSE           NaN        NaN        NaN        NaN         NaN        NaN        NaN               NaN               NaN                  NaN                  NaN             NaN          NaN          
# TRUE      2.065236  0.7213853  0.4950557  0.6592012  0.01297957  0.2891368  0.3440605         0.6695511         0.5089483            0.7356546            0.5089483       0.3440605    0.2197337          
# Mean_Balanced_Accuracy
# NaN             
# 0.5068058             
# 
# Tuning parameter 'fL' was held constant at a value of 0
# Tuning parameter 'adjust' was held constant at a value of 1
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were fL = 0, usekernel = TRUE and adjust = 1.

### Predict the test set
goal.Results_nBayesFit <- data.frame(obs = Test.NHL_df_event$event)
goal.Results_nBayesFit$prob <- predict(nBayesFit, Test.NHL_df_event, type = "prob")[, "GOAL"]
goal.Results_nBayesFit$pred <- predict(nBayesFit, Test.NHL_df_event)
goal.Results_nBayesFit$Label <- ifelse(goal.Results_nBayesFit$obs == "GOAL",
                                       "True Outcome: GOAL",
                                       "True Outcome: MISS",
                                       "True Outcome: SHOT")


goal.Results_nBayesFit$obs <- as.factor(goal.Results_nBayesFit$obs)

defaultSummary(goal.Results_nBayesFit)

# Accuracy      Kappa 
# 0.65943366 0.01480406 

### Create the confusion matrix from the test set.
confusionMatrix(data = goal.Results_nBayesFit$pred, 
                reference = goal.Results_nBayesFit$obs)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  GOAL  MISS  SHOT
# GOAL    38    13    32
# MISS     5    35    41
# SHOT  1052  4209 10290
# 
# Overall Statistics
# 
# Accuracy : 0.6594         
# 95% CI : (0.652, 0.6668)
# No Information Rate : 0.6594         
# P-Value [Acc > NIR] : 0.5037         
# 
# Kappa : 0.0148         
# 
# Mcnemar's Test P-Value : <2e-16         
# 
# Statistics by Class:
# 
#                      Class: GOAL Class: MISS Class: SHOT
# Sensitivity             0.034703    0.008222      0.9930
# Specificity             0.996922    0.995985      0.0170
# Pos Pred Value          0.457831    0.432099      0.6617
# Neg Pred Value          0.932382    0.729948      0.5549
# Prevalence              0.069679    0.270888      0.6594
# Detection Rate          0.002418    0.002227      0.6548
# Detection Prevalence    0.005282    0.005154      0.9896
# Balanced Accuracy       0.515813    0.502104      0.5050  

### ROC curves:
goal.ROC_nBayesFit <- multiclass.roc(response = goal.Results_nBayesFit$obs, predictor = goal.Results_nBayesFit$prob, levels = levels(goal.Results_nBayesFit$obs))

auc(goal.ROC_nBayesFit) # Area under the curve: 0.7518

### Plot ROC
plot(goal.ROC_nBayesFit, type = "s", legacy.axes = TRUE)

nBayes=list( classifier=nBayesFit, roc=goal.ROC_nBayesFit, auc=auc(goal.ROC_nBayesFit) )
nBayes


result.final = list( mda=mda, fda=fda, nBayes=nBayes )

#result.final = list( mda=mda, nnet1=nnet1, nnet2=nnet2, fda=fda, svm=svm, knn=knn, nBayes=nBayes )
