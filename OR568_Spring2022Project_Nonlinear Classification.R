#' ---
#' Team: Chang,Donde,Hassan,Karimi,Miller,Turissini    
#' title: "OR 568 - Spring 2022, Prof Xu"
#' Project
#' Due: May 10, 2022 11:59pm
#'


################################################################################
#########################  Goal-No Goal 2018-2020  #############################
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
NHL_df = read.csv("C:/Users/m29336/Documents/NHL_df3.csv", header=TRUE)
dim(NHL_df)        # 300359    91

NHL_df$goal = as.factor(NHL_df$goal) #change to factor
class(NHL_df$goal)

## Split the data into training (80%) and test sets (20%)
set.seed(123)
inTrain <- createDataPartition(NHL_df$goal, p = .8)[[1]]
Train.NHL_df <- NHL_df[ inTrain, ]
Test.NHL_df  <- NHL_df[-inTrain, ]

#set Control
ctrl = trainControl( summaryFunction=twoClassSummary, classProbs=TRUE )

################################################################################
### 1 Nonlinear Discriminant Analysis

set.seed(123)
mdaFit <- train(goal ~ .,
                data = Train.NHL_df,
                method = "mda",
                metric = "ROC",
                tuneGrid = expand.grid(subclasses = 1:8),
                trControl = ctrl)
mdaFit

#Mixture Discriminant Analysis 

#240288 samples
#90 predictor
#2 classes: 'Goal', 'NoGoal' 

#No pre-processing
#Resampling: Bootstrapped (25 reps) 
#Summary of sample sizes: 240288, 240288, 240288, 240288, 240288, 240288, ... 
#Resampling results across tuning parameters:

#  subclasses  ROC        Sens       Spec     
#1           0.9959637  0.9997626  0.9732429
#2           0.9960037  0.9997626  0.9732429
#3           0.9959581  0.9997626  0.9732429
#4           0.9953513  0.9997626  0.9731409
#5           0.9953391  0.9997626  0.9730908
#6           0.9945399  0.9997559  0.9722628
#7           0.9946743  0.9994343  0.9725876
#8           0.9944316  0.9988393  0.9722394

#ROC was used to select the optimal model using the largest value.
#The final value used for the model was subclasses = 2.


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
# 0.9741306 0.8278885 

### Create the confusion matrix from the test set.
confusionMatrix(data = goal.Results_mdaFit$pred, 
                reference = goal.Results_mdaFit$obs)

#Confusion Matrix and Statistics

#Reference
#Prediction  Goal NoGoal
#Goal    4128   1553
#NoGoal     1  54389

#Accuracy : 0.9741          
#95% CI : (0.9728, 0.9754)
#No Information Rate : 0.9313          
#P-Value [Acc > NIR] : < 2.2e-16       

#Kappa : 0.8279          

#Mcnemar's Test P-Value : < 2.2e-16       

#            Sensitivity : 0.99976         
#            Specificity : 0.97224         
#         Pos Pred Value : 0.72663         
#         Neg Pred Value : 0.99998         
#             Prevalence : 0.06874         
#         Detection Rate : 0.06872         
#   Detection Prevalence : 0.09457         
#      Balanced Accuracy : 0.98600         

#       'Positive' Class : Goal

### ROC curves:
goal.ROC_mdaFit <- roc(response = goal.Results_mdaFit$obs, predictor = goal.Results_mdaFit$prob, levels = levels(goal.Results_mdaFit$obs))
coords(goal.ROC_mdaFit, "all")[,1:3]

auc(goal.ROC_mdaFit) # Area under the curve: 0.9955
ci.auc(goal.ROC_mdaFit) # 95% CI: 0.9951-0.9959 (DeLong)

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
fdaFit <- train(goal ~ .,
                data = Train.NHL_df,
                method = "fda",
                metric = "ROC",
                tuneGrid = expand.grid(degree = 1, nprune = 2:25),
                trControl = ctrl)
fdaFit

#Flexible Discriminant Analysis 

#240288 samples
#90 predictor
#2 classes: 'Goal', 'NoGoal' 

#No pre-processing
#Resampling: Bootstrapped (25 reps) 
#Summary of sample sizes: 240288, 240288, 240288, 240288, 240288, 240288, ... 
#Resampling results across tuning parameters:

#  nprune  ROC        Sens       Spec     
#2      0.6993307  0.0000000  1.0000000
#3      0.8700318  0.0000000  1.0000000
#4      0.9575686  0.9997626  0.9151989
#5      0.9865975  0.9997626  0.9732429
#6      0.9924534  0.9997626  0.9732429
#7      0.9956139  0.9997626  0.9732429
#8      0.9956881  0.9997626  0.9732429 ***
#9      0.9956881  0.9997626  0.9732429
#10      0.9956881  0.9997626  0.9732429
#11      0.9956881  0.9997626  0.9732429
#12      0.9956881  0.9997626  0.9732429
#13      0.9956881  0.9997626  0.9732429
#14      0.9956881  0.9997626  0.9732429
#15      0.9956881  0.9997626  0.9732429
#16      0.9956881  0.9997626  0.9732429
#17      0.9956881  0.9997626  0.9732429
#18      0.9956881  0.9997626  0.9732429
#19      0.9956881  0.9997626  0.9732429
#20      0.9956881  0.9997626  0.9732429
#21      0.9956881  0.9997626  0.9732429
#22      0.9956881  0.9997626  0.9732429
#23      0.9956881  0.9997626  0.9732429
#24      0.9956881  0.9997626  0.9732429
#25      0.9956881  0.9997626  0.9732429

#Tuning parameter 'degree' was held constant at a value of 1
#ROC was used to select the optimal model using the largest value.
#The final values used for the model were degree = 1 and nprune = 8.




### Predict the test set
goal.Results_fdaFit <- data.frame(obs = Test.NHL_df$goal)
goal.Results_fdaFit$prob <- predict(fdaFit, Test.NHL_df, type = "prob")[, "Goal"]
goal.Results_fdaFit$pred <- predict(fdaFit, Test.NHL_df)
goal.Results_fdaFit$Label <- ifelse(goal.Results_fdaFit$obs == "Goal", 
                                    "True Outcome: goal", 
                                    "True Outcome: no goal")

goal.Results_fdaFit$obs <- as.factor(goal.Results_fdaFit$obs)

defaultSummary(goal.Results_fdaFit)

#Accuracy     Kappa 
#0.9741306 0.8278885

### Create the confusion matrix from the test set.
confusionMatrix(data = goal.Results_fdaFit$pred, 
                reference = goal.Results_fdaFit$obs)

#Confusion Matrix and Statistics

#Reference
#Prediction  Goal NoGoal
#Goal    4128   1553
#NoGoal     1  54389

#Accuracy : 0.9741          
#95% CI : (0.9728, 0.9754)
#No Information Rate : 0.9313          
#P-Value [Acc > NIR] : < 2.2e-16       

#Kappa : 0.8279          

#Mcnemar's Test P-Value : < 2.2e-16       

#            Sensitivity : 0.99976         
#            Specificity : 0.97224         
#         Pos Pred Value : 0.72663         
#         Neg Pred Value : 0.99998         
#             Prevalence : 0.06874         
#         Detection Rate : 0.06872         
#   Detection Prevalence : 0.09457         
#      Balanced Accuracy : 0.98600         

#       'Positive' Class : Goal

### ROC curves:
goal.ROC_fdaFit <- roc(response = goal.Results_fdaFit$obs, predictor = goal.Results_fdaFit$prob, levels = levels(goal.Results_fdaFit$obs))
coords(goal.ROC_fdaFit, "all")[,1:3]

auc(goal.ROC_fdaFit) # [insert output]
ci.auc(goal.ROC_fdaFit) # [insert output]

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
# knnFit <- train(goal ~ .,
#                 data = Train.NHL_df,
#                 method = "knn",
#                 metric = "ROC",
#                 preProc = c("center", "scale"),
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
nBayesFit <- train(goal ~ .,
                   data = Train.NHL_df,
                   method = "nb",
                   metric = "ROC",
                   trControl = ctrl)
nBayesFit

#Naive Bayes 

#240288 samples
#90 predictor
#2 classes: 'Goal', 'NoGoal' 

#No pre-processing
#Resampling: Bootstrapped (25 reps) 
#Summary of sample sizes: 240288, 240288, 240288, 240288, 240288, 240288, ... 
#Resampling results across tuning parameters:

#  usekernel  ROC        Sens         Spec     
#FALSE            NaN          NaN        NaN
#TRUE      0.8865865  0.001457407  0.9998688

#Tuning parameter 'fL' was held constant at a value of 0
#Tuning
#parameter 'adjust' was held constant at a value of 1
#ROC was used to select the optimal model using the largest value.
#The final values used for the model were fL = 0, usekernel = TRUE and
#adjust = 1.



### Predict the test set
goal.Results_nBayesFit <- data.frame(obs = Test.NHL_df$goal)
goal.Results_nBayesFit$prob <- predict(nBayesFit, Test.NHL_df, type = "prob")[, "Goal"]
goal.Results_nBayesFit$pred <- predict(nBayesFit, Test.NHL_df)
goal.Results_nBayesFit$Label <- ifelse(goal.Results_nBayesFit$obs == "Goal", 
                                       "True Outcome: goal", 
                                       "True Outcome: no goal")

goal.Results_nBayesFit$obs <- as.factor(goal.Results_nBayesFit$obs)

defaultSummary(goal.Results_nBayesFit)

#   Accuracy       Kappa 
# 0.931297964 0.002151946


### Create the confusion matrix from the test set.
confusionMatrix(data = goal.Results_nBayesFit$pred, 
                reference = goal.Results_nBayesFit$obs)

#Confusion Matrix and Statistics

#Reference
#Prediction  Goal NoGoal
#Goal       5      3
#NoGoal  4124  55939

#Accuracy : 0.9313          
#95% CI : (0.9292, 0.9333)
#No Information Rate : 0.9313          
#P-Value [Acc > NIR] : 0.4913          

#Kappa : 0.0022          

#Mcnemar's Test P-Value : <2e-16          

#            Sensitivity : 1.211e-03       
#            Specificity : 9.999e-01       
#         Pos Pred Value : 6.250e-01       
#         Neg Pred Value : 9.313e-01       
#             Prevalence : 6.874e-02       
#         Detection Rate : 8.323e-05       
#   Detection Prevalence : 1.332e-04       
#      Balanced Accuracy : 5.006e-01       

#       'Positive' Class : Goal


### ROC curves:
goal.ROC_nBayesFit <- roc(response = goal.Results_nBayesFit$obs, predictor = goal.Results_nBayesFit$prob, levels = levels(goal.Results_nBayesFit$obs))
coords(goal.ROC_nBayesFit, "all")[,1:3]

auc(goal.ROC_nBayesFit) # [insert output]
ci.auc(goal.ROC_nBayesFit) # [insert output]

### Plot ROC
plot(goal.ROC_nBayesFit, type = "s", legacy.axes = TRUE)

nBayes=list( classifier=nBayesFit, roc=goal.ROC_nBayesFit, auc=auc(goal.ROC_nBayesFit) )
nBayes


###############################
###############################

plot(goal.ROC_mdaFit, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(goal.ROC_nnetFit, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(goal.ROC_nnetFit2, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(goal.ROC_fdaFit, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(goal.ROC_svmFit, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(goal.ROC_knnFit, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(goal.ROC_nBayesFit, type = "s", add = TRUE, legacy.axes = TRUE)

###############################
###############################

goalResamples <- resamples(list(mda = mdaFit,
                                fda = fdaFit,
                                nBayes = nBayesFit))

bwplot(goalResamples, metric = "ROC")


result.final = list( mda=mda, fda=fda, nBayes=nBayes )

# goalResamples <- resamples(list(mda = mdaFit,
#                                 nnet1 = nnetFit,
#                                 nnet2 = nnetFit2,
#                                 fda = fdaFit,
#                                 svm = svmFit,
#                                 knn = knnFit,
#                                 nBayes = nBayesFit))
# 
# bwplot(goalResamples, metric = "ROC")
# 
# 
# result.final = list( mda=mda, nnet1=nnet1, nnet2=nnet2, fda=fda, svm=svm, knn=knn, nBayes=nBayes )

