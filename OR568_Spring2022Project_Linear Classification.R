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
### Linear Classification
# 1 Logistic Regression
# 2 Linear Discriminant Analysis
# 3 Partial Least Squares Discriminant Analysis
# 4 glmnet
# 5 Sparse logistic regression
# 6 Nearest Shrunken Centroids

library(caret)
library(doMC)
library(pROC)
library(glmnet)
library(lattice)
library(MASS)
library(pamr) 
library(pls)
library(sparseLDA)


# Import the dataset
NHL_df = read.csv("C:/Users/m29336/Documents/NHL_df3.csv", header=TRUE)
dim(NHL_df)        # 300359    91

NHL_df$goal = as.factor(NHL_df$goal) #change to factor
class(NHL_df$goal)

#set Control
ctrl = trainControl( summaryFunction=twoClassSummary, classProbs=TRUE )

## Split the data into training (80%) and test sets (20%)
set.seed(123)
inTrain <- createDataPartition(NHL_df$goal, p = .8)[[1]]
Train.NHL_df <- NHL_df[ inTrain, ]
Test.NHL_df  <- NHL_df[-inTrain, ]


################################################################################
### 1 Logistic Regression

set.seed(123)
logisticReg <- train(goal ~ .,
                     data = Train.NHL_df,
                     method = "glm",
                     trControl = ctrl)
logisticReg

#Generalized Linear Model
#
#240288 samples
#90 predictor
#2 classes: 'NoGoal', 'Goal'
#
#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 1 times)
#Summary of sample sizes: 216259, 216258, 216259, 216260, 216259, 216260, ...
#Resampling results:
#  
#  ROC        Sens       Spec    
#0.996292  0.9779004  0.9870448


logisticRegImp <- varImp(logisticReg, scale = FALSE)
logisticRegImp

#glm variable importance
#
#only 20 most important variables shown (out of 91)
#
#Overall
#shotGoalieFroze              21.590
#homePenalty1Length           12.060
#awayPenalty1Length            6.060
#arenaAdjustedYCordAbs         5.310
#offWing                       5.240
#arenaAdjustedShotDistance     4.505
#shotTypeBack                  4.333
#playerPositionThatDidEventD   4.310
#shotTypeWrist                 4.146
#defendingTeamForwardsOnIce    4.076
#shootingTeamDefencemenOnIce   3.639
#playerPositionThatDidEventC   3.629
#playerPositionThatDidEventL   3.620
#shotAngleAdjusted             3.357
#homeSkatersOnIce              3.356
#playerPositionThatDidEventR   3.268
#time                          3.204
#awaySkatersOnIce              3.195
#season                        2.681
#shotTypeSnap                  2.634

plot(logisticRegImp, top=15, scales = list(y = list(cex = .85)))

### Predict the test set
goal.Results_LR <- data.frame(obs = Test.NHL_df$goal)
goal.Results_LR$prob <- predict(logisticReg, Test.NHL_df, type = "prob")[, "Goal"]
goal.Results_LR$pred <- predict(logisticReg, Test.NHL_df)
goal.Results_LR$Label <- ifelse(goal.Results_LR$obs == "Goal",
                                "True Outcome: goal",
                                "True Outcome: no goal")


goal.Results_LR$obs <- as.factor(goal.Results_LR$obs)

defaultSummary(goal.Results_LR)
#Accuracy     Kappa
#0.9853174 0.8935715  

### Plot the probability
histogram(~prob|Label,
          data = goal.Results_LR,
          layout = c(2, 1),
          nint = 20,
          xlab = "Probability of goal",
          type = "count")

### Calculate and plot the calibration curve
goal.Calib_LR <- calibration(obs ~ prob, data = goal.Results_LR)
xyplot(goal.Calib_LR)

### Create the confusion matrix from the test set.
confusionMatrix(data = goal.Results_LR$pred,
                reference = goal.Results_LR$obs)

#Confusion Matrix and Statistics
#
#Reference
#Prediction  Goal NoGoal
#Goal    4034    787
#NoGoal    95  55155
#
#Accuracy : 0.9853          
#95% CI : (0.9843, 0.9863)
#No Information Rate : 0.9313          
#P-Value [Acc > NIR] : < 2.2e-16      
#
#Kappa : 0.8936          
#
#Mcnemar's Test P-Value : < 2.2e-16      
#                                          
#            Sensitivity : 0.97699        
#            Specificity : 0.98593        
#         Pos Pred Value : 0.83676        
#         Neg Pred Value : 0.99828        
#             Prevalence : 0.06874        
#         Detection Rate : 0.06715        
#   Detection Prevalence : 0.08026        
#      Balanced Accuracy : 0.98146        
#                                          
#       'Positive' Class : Goal                

### ROC curves:
goal.ROC_LR <- roc(response = goal.Results_LR$obs, predictor = goal.Results_LR$prob, levels = levels(goal.Results_LR$obs))
coords(goal.ROC_LR, "all")[,1:3]

auc(goal.ROC_LR) #Area under the curve: 0.9956

### Note the x-axis is reversed
plot(goal.ROC_LR)

### Lift charts
goal.Lift_LR <- lift(obs ~ prob, data = goal.Results_LR)
xyplot(goal.Lift_LR)

################################################################################
### 2 Linear Discriminant Analysis

set.seed(123)
ldaFit <- train(goal ~ .,
                data = Train.NHL_df,
                method = "lda",
                metric = "ROC",
                trControl = ctrl)
ldaFit

#Linear Discriminant Analysis
#
#240288 samples
#90 predictor
#2 classes: 'Goal', 'NoGoal'
#
#No pre-processing
#Resampling: Bootstrapped (25 reps)
#Summary of sample sizes: 240288, 240288, 240288, 240288, 240288, 240288, ...
#Resampling results:
#  
#  ROC        Sens       Spec    
#0.9959637  0.9997626  0.9732429

ldaImp <- varImp(ldaFit, scale = FALSE)
ldaImp

#ROC curve variable importance
#
#only 20 most important variables shown (out of 90)
#
#Importance
#arenaAdjustedShotDistance        0.7022
#shotPlayContinuedOutsideZone     0.6992
#arenaAdjustedYCordAbs            0.6734
#shotPlayContinuedInZone          0.6707
#arenaAdjustedXCordABS            0.6649
#shotWasOnGoal                    0.6466
#shotGoalieFroze                  0.5876
#timeSinceLastEvent               0.5857
#playerPositionThatDidEventD      0.5847
#speedFromLastEvent               0.5549
#shotRebound                      0.5521
#shotAngleAdjusted                0.5508
#playerPositionThatDidEventC      0.5481
#shotAnglePlusReboundSpeed        0.5442
#distanceFromLastEvent            0.5382
#shotAnglePlusRebound             0.5356
#time                             0.5353
#shootingTeamAverageTimeOnIce     0.5349
#shootingTeamMaxTimeOnIce         0.5343
#shootingTeamDefencemenOnIce      0.5323

plot(ldaImp, top=15, scales = list(y = list(cex = .85)))

### Predict the test set
goal.Results_ldaFit <- data.frame(obs = Test.NHL_df$goal)
goal.Results_ldaFit$prob <- predict(ldaFit, Test.NHL_df, type = "prob")[, "Goal"]
goal.Results_ldaFit$pred <- predict(ldaFit, Test.NHL_df)
goal.Results_ldaFit$Label <- ifelse(goal.Results_ldaFit$obs == "Goal",
                                    "True Outcome: goal",
                                    "True Outcome: no goal")

goal.Results_ldaFit$obs <- as.factor(goal.Results_ldaFit$obs)

defaultSummary(goal.Results_ldaFit)
#Accuracy     Kappa
#0.9741306 0.8278885

### Plot the probability of a goal
histogram(~prob|Label,
          data = goal.Results_ldaFit,
          layout = c(2, 1),
          nint = 20,
          xlab = "Probability of goal",
          type = "count")

### Calculate and plot the calibration curve
goal.Calib_ldaFit <- calibration(obs ~ prob, data = goal.Results_ldaFit)
xyplot(goal.Calib_ldaFit)

### Create the confusion matrix from the test set.
confusionMatrix(data = goal.Results_ldaFit$pred,
                reference = goal.Results_ldaFit$obs)

#Confusion Matrix and Statistics
#
#Reference
#Prediction  Goal NoGoal
#Goal    4128   1553
#NoGoal     1  54389
#
#Accuracy : 0.9741          
#95% CI : (0.9728, 0.9754)
#No Information Rate : 0.9313          
#P-Value [Acc > NIR] : < 2.2e-16      
#
#Kappa : 0.8279          
#
#Mcnemar's Test P-Value : < 2.2e-16      
#                                          
#            Sensitivity : 0.99976        
#            Specificity : 0.97224        
#         Pos Pred Value : 0.72663        
#         Neg Pred Value : 0.99998        
#             Prevalence : 0.06874        
#         Detection Rate : 0.06872        
#   Detection Prevalence : 0.09457        
#      Balanced Accuracy : 0.98600        
#                                          
#       'Positive' Class : Goal  

### ROC curves:
goal.ROC_ldaFit <- roc(response = goal.Results_ldaFit$obs, predictor = goal.Results_ldaFit$prob, levels = levels(goal.Results_ldaFit$obs))
coords(goal.ROC_ldaFit, "all")[,1:3]

auc(goal.ROC_ldaFit) #Area under the curve: 0.9953

### Note the x-axis is reversed
plot(goal.ROC_ldaFit)

### Lift charts
goal.Lift_ldaFit <- lift(obs ~ prob, data = goal.Results_ldaFit)
xyplot(goal.Lift_ldaFit)



################################################################################
### 3 Partial Least Squares Discriminant Analysis

set.seed(123)

plsFit <- train(goal ~ .,
                data = Train.NHL_df,
                method = "pls",
                tuneGrid = expand.grid(ncomp = 1:50),
                metric = "ROC",
                trControl = ctrl)
plsFit

#Partial Least Squares 
#
#240288 samples
#91 predictor
#2 classes: 'Goal', 'NoGoal' 
#
#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 5 times) 
#Summary of sample sizes: 216259, 216258, 216259, 216260, 216259, 216260, ... 
#Resampling results across tuning parameters:
#  
#  ncomp  ROC        Sens          Spec     
#1     0.5366997  0.0000000000  1.0000000
#2     0.6595268  0.0001816494  0.9999607
#3     0.7670039  0.0000000000  0.9999955
#4     0.7817446  0.0000000000  0.9999955
#5     0.7965744  0.0001816494  0.9999616
#6     0.8060205  0.0000000000  0.9999955
#7     0.8114653  0.0000000000  0.9999955
#8     0.8159241  0.0000000000  0.9999955
#9     0.8199835  0.0000000000  0.9999955
#10     0.8232254  0.0000000000  0.9999955
#
#ROC was used to select the optimal model using the largest value.
#The final value used for the model was ncomp = 10.

plsImp <- varImp(plsFit, scale = FALSE)
plsImp

#pls variable importance
#
#only 20 most important variables shown (out of 91)
#
#Overall
#timeUntilNextEvent                                5.550e-04
#arenaAdjustedShotDistance                         5.126e-04
#arenaAdjustedXCordABS                             4.059e-04
#arenaAdjustedYCordAbs                             2.851e-04
#homePenalty1Length                                2.832e-04
#shotAngleAdjusted                                 1.999e-04
#shotAnglePlusReboundSpeed                         1.493e-04
#timeSinceLastEvent                                1.300e-04
#speedFromLastEvent                                1.256e-04
#timeSinceFaceoff                                  9.155e-05
#awayPenalty1Length                                8.587e-05
#shotAnglePlusRebound                              8.586e-05
#defendingTeamMinTimeOnIceOfDefencemenSinceFaceoff 7.251e-05
#shootingTeamMinTimeOnIceOfDefencemenSinceFaceoff  7.002e-05
#playerNumThatDidEvent                             6.919e-05
#distanceFromLastEvent                             6.871e-05
#defendingTeamMinTimeOnIceOfForwardsSinceFaceoff   6.522e-05
#defendingTeamMinTimeOnIceOfForwards               6.316e-05
#defendingTeamMinTimeOnIce                         6.207e-05
#defendingTeamMinTimeOnIceOfDefencemen             5.876e-05

plot(plsImp, top=15, scales = list(y = list(cex = .85)))

### Predict the test set
goal.Results_plsFit <- data.frame(obs = Test.NHL_df$goal)
goal.Results_plsFit$prob <- predict(plsFit, Test.NHL_df, type = "prob")[, "Goal"]
goal.Results_plsFit$pred <- predict(plsFit, Test.NHL_df)
goal.Results_plsFit$Label <- ifelse(goal.Results_plsFit$obs == "Goal", 
                             "True Outcome: goal", 
                             "True Outcome: no goal")

goal.Results_plsFit$obs <- as.factor(goal.Results_plsFit$obs)

### Plot the probability
histogram(~prob|Label,
          data = goal.Results_plsFit,
          layout = c(2, 1),
          nint = 20,
          xlab = "Probability of goal",
          type = "count")

### Calculate and plot the calibration curve
goal.Calib_plsFit <- calibration(obs ~ prob, data = goal.Results_plsFit)
xyplot(goal.Calib_plsFit)

### Create the confusion matrix from the test set.
confusionMatrix(data = goal.Results_plsFit$pred, 
                reference = goal.Results_plsFit$obs)

#Confusion Matrix and Statistics
#
#           Reference
#Prediction  Goal NoGoal
#Goal           0      0
#NoGoal      4129  55942
#
#Accuracy : 0.9313          
#95% CI : (0.9292, 0.9333)
#No Information Rate : 0.9313          
#P-Value [Acc > NIR] : 0.5041          
#
#Kappa : 0               
#
#Mcnemar's Test P-Value : <2e-16          
#                                          
#            Sensitivity : 0.00000         
#            Specificity : 1.00000         
#         Pos Pred Value :     NaN         
#         Neg Pred Value : 0.93126         
#             Prevalence : 0.06874         
#         Detection Rate : 0.00000         
#   Detection Prevalence : 0.00000         
#      Balanced Accuracy : 0.50000         
#                                          
#       'Positive' Class : Goal    

### ROC curves:
goal.ROC_plsFit <- roc(response = goal.Results_plsFit$obs, predictor = goal.Results_plsFit$prob, levels = levels(goal.Results_plsFit$obs))
coords(goal.ROC_plsFit, "all")[,1:3]

auc(goal.ROC_plsFit) #Area under the curve: 0.7294

### Note the x-axis is reversed
plot(goal.ROC_plsFit)

### Lift charts
goal.Lift_plsFit <- lift(obs ~ prob, data = goal.Results_ldaFit)
xyplot(goal.Lift_plsFit)


################################################################################
### 4 glmnet

glmnGrid <- expand.grid(alpha = c(0,  .1,  .2, .4, .6, .8, 1), lambda = seq(.01, .2, length = 40))

set.seed(123)
glmnFit <- train(goal ~ .,
                 data = Train.NHL_df,
                 method = "glmnet",
                 tuneGrid = glmnGrid,
                 metric = "ROC",
                 trControl = ctrl)
glmnFit


# 0.1    0.01000000  0.9961510  7.530162e-01  0.9945540

glmnFitImp <- varImp(glmnFit, scale = FALSE)
glmnFitImp

#glmnet variable importance
#
#only 20 most important variables shown (out of 91)
#
#Overall
#shotGoalieFroze              3.462011
#shotPlayContinuedOutsideZone 3.453321
#shotGeneratedRebound         3.405815
#shotPlayContinuedInZone      3.405052
#shotWasOnGoal                1.629475
#shootingTeamDefencemenOnIce  0.214779
#shotRebound                  0.191206
#playerPositionThatDidEventD  0.176113
#homeSkatersOnIce             0.117162
#shotTypeSnap                 0.091845
#defendingTeamForwardsOnIce   0.072901
#shotTypeBack                 0.039083
#shotTypeSlap                 0.031242
#arenaAdjustedYCordAbs        0.017312


plot(glmnFitImp, top=15, scales = list(y = list(cex = .85)))

### Predict the test set
goal.Results_glmnFit <- data.frame(obs = Test.NHL_df$goal)
goal.Results_glmnFit$prob <- predict(glmnFit, Test.NHL_df, type = "prob")[, "Goal"]
goal.Results_glmnFit$pred <- predict(glmnFit, Test.NHL_df)
goal.Results_glmnFit$Label <- ifelse(goal.Results_glmnFit$obs == "Goal", 
                                    "True Outcome: goal", 
                                    "True Outcome: no goal")

goal.Results_glmnFit$obs <- as.factor(goal.Results_glmnFit$obs)

### Plot the probability 
histogram(~prob|Label,
          data = goal.Results_glmnFit,
          layout = c(2, 1),
          nint = 20,
          xlab = "Probability of goal",
          type = "count")

### Calculate and plot the calibration curve
goal.Calib_glmnFit <- calibration(obs ~ prob, data = goal.Results_glmnFit)
xyplot(goal.Calib_glmnFit)

### Create the confusion matrix from the test set.
confusionMatrix(data = goal.Results_glmnFit$pred, 
                reference = goal.Results_glmnFit$obs)


#Reference
#Prediction  Goal NoGoal
#Goal    3079    366
#NoGoal  1050  55576

#Accuracy : 0.9764          
#95% CI : (0.9752, 0.9776)
#No Information Rate : 0.9313          
#P-Value [Acc > NIR] : < 2.2e-16       

#Kappa : 0.8006          

#Mcnemar's Test P-Value : < 2.2e-16       

#           Sensitivity : 0.74570         
#          Specificity : 0.99346         
#       Pos Pred Value : 0.89376         
#         Neg Pred Value : 0.98146         
#             Prevalence : 0.06874         
#         Detection Rate : 0.05126         
#   Detection Prevalence : 0.05735         
#      Balanced Accuracy : 0.86958         

#       'Positive' Class : Goal 

### ROC curves:
goal.ROC_glmnFit <- roc(response = goal.Results_glmnFit$obs, predictor = goal.Results_glmnFit$prob, levels = levels(goal.Results_glmnFit$obs))
coords(goal.ROC_glmnFit, "all")[,1:3]

auc(goal.ROC_glmnFit) #Area under the curve: 0.9955

### Note the x-axis is reversed
plot(goal.ROC_glmnFit)

### Lift charts
goal.Lift_glmnFit <- lift(obs ~ prob, data = goal.Results_glmnFit)
xyplot(goal.Lift_glmnFit)


################################################################################
### 5 Sparse logistic regression

set.seed(476)
spLDAFit <- train(goal ~ .,
                  data = Train.NHL_df,
                  "sparseLDA",
                  tuneGrid = expand.grid(lambda = c(.1),
                                         NumVars = c(1, 5, 10, 15, 20, 50, 100, 250, 500, 1000)),
                  preProc = c("center", "scale"),
                  metric = "ROC",
                  trControl = ctrl)



spLDAFit


#spLDAFit

#NumVars  ROC        Sens       Spec     
#1     0.6990899  0.0000000  1.0000000
#5     0.9920791  0.9997426  0.9177753
#10     0.9957776  0.9997426  0.9731398
#15     0.9958832  0.9997426  0.9731398
#20     0.9959152  0.9997426  0.9731398
#50     0.9959527  0.9997426  0.9731398
#100     0.9958835  0.9997426  0.9731398
#250     0.9958835  0.9997426  0.9731398
#500     0.9958835  0.9997426  0.9731398
#1000     0.9958835  0.9997426  0.9731398

#Tuning parameter 'lambda' was held constant at a value of 0.1
#ROC was used to select the optimal model using the largest value.
#The final values used for the model were NumVars = 50 and lambda = 0.1.

### Predict the test set
goal.Results_spLDAFit <- data.frame(obs = Test.NHL_df$goal)
goal.Results_spLDAFit$prob <- predict(spLDAFit, Test.NHL_df, type = "prob")[, "Goal"]
goal.Results_spLDAFit$pred <- predict(spLDAFit, Test.NHL_df)
goal.Results_spLDAFit$Label <- ifelse(goal.Results_spLDAFit$obs == "Goal", 
                                     "True Outcome: goal", 
                                     "True Outcome: no goal")

goal.Results_spLDAFit$obs <- as.factor(goal.Results_spLDAFit$obs)


### Plot the probability 
histogram(~prob|Label,
          data = goal.Results_spLDAFit,
          layout = c(2, 1),
          nint = 20,
          xlab = "Probability of goal",
          type = "count")

### Calculate and plot the calibration curve
goal.Calib_glmnFit <- calibration(obs ~ prob, data = goal.Results_spLDAFit)
xyplot(goal.Calib_glmnFit)

### Create the confusion matrix from the test set.
confusionMatrix(data = goal.Results_spLDAFit$pred, 
                reference = goal.Results_spLDAFit$obs)

# Confusion Matrix and Statistics

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
goal.ROC_spLDAFit <- roc(response = goal.Results_spLDAFit$obs, predictor = goal.Results_spLDAFit$prob, levels = levels(goal.Results_spLDAFit$obs))
coords(goal.ROC_spLDAFit, "all")[,1:3]

auc(goal.ROC_spLDAFit) #Area under the curve: 0.9953

### Note the x-axis is reversed
plot(goal.ROC_spLDAFit)

### Lift charts
goal.Lift_spLDAFit <- lift(obs ~ prob, data = goal.Results_spLDAFit)
xyplot(goal.Lift_spLDAFit)



################################################################################
### 6 Nearest Shrunken Centroids

set.seed(123)
nscFit <- train(goal ~ .,
                data = Train.NHL_df,
                method = "pam",
                tuneGrid = data.frame(threshold = seq(0, 25, length = 30)),
                metric = "ROC",
                trControl = ctrl)
nscFit

#Nearest Shrunken Centroids 
#
#240288 samples
#90 predictor
#2 classes: 'Goal', 'NoGoal' 
#
#No pre-processing
#Resampling: Bootstrapped (25 reps) 
#Summary of sample sizes: 240288, 240288, 240288, 240288, 240288, 240288, ... 
#Resampling results across tuning parameters:
#  
#  threshold  ROC        Sens          Spec     
#0.000000  0.6510068  1.022796e-02  0.9960872
#0.862069  0.6587781  6.201288e-03  0.9976793
#1.724138  0.6679300  3.227503e-03  0.9987974
#2.586207  0.6782716  1.453038e-03  0.9995051
#3.448276  0.6893469  7.007475e-04  0.9998279
#4.310345  0.7002771  2.245739e-04  0.9999489
#5.172414  0.7091765  8.603171e-05  0.9999796
#6.034483  0.7144924  0.000000e+00  0.9999956
#6.896552  0.7165885  0.000000e+00  1.0000000 ***
#7.758621  0.7165699  0.000000e+00  1.0000000
#8.620690  0.7149019  0.000000e+00  1.0000000
#9.482759  0.7123265  0.000000e+00  1.0000000
#10.344828  0.7099949  0.000000e+00  1.0000000
#11.206897  0.7081698  0.000000e+00  1.0000000
#12.068966  0.7068230  0.000000e+00  1.0000000
#12.931034  0.7053319  0.000000e+00  1.0000000
#13.793103  0.7037497  0.000000e+00  1.0000000
#14.655172  0.7026413  0.000000e+00  1.0000000
#15.517241  0.7021480  0.000000e+00  1.0000000
#16.379310  0.7018332  0.000000e+00  1.0000000
#17.241379  0.7016031  0.000000e+00  1.0000000
#18.103448  0.7013536  0.000000e+00  1.0000000
#18.965517  0.7010842  0.000000e+00  1.0000000
#19.827586  0.7007848  0.000000e+00  1.0000000
#20.689655  0.7004280  0.000000e+00  1.0000000
#21.551724  0.7000259  0.000000e+00  1.0000000
#22.413793  0.6995552  0.000000e+00  1.0000000
#23.275862  0.6990050  0.000000e+00  1.0000000
#24.137931  0.6983711  0.000000e+00  1.0000000
#25.000000  0.6976086  0.000000e+00  1.0000000
#
#ROC was used to select the optimal model using the largest value.
#The final value used for the model was threshold = 6.896552.

nscFitImp <- varImp(nscFit, scale = FALSE)
nscFitImp

#pam variable importance
#
#only 20 most important variables shown (out of 90)
#
#Importance
#arenaAdjustedShotDistance                  -0.261962
#arenaAdjustedXCordABS                       0.202573
#arenaAdjustedYCordAbs                      -0.154887
#time                                        0.060883
#shotAnglePlusReboundSpeed                   0.055117
#homePenalty1Length                         -0.049186
#shotAngleAdjusted                          -0.030975
#shootingTeamMaxTimeOnIce                    0.028900
#timeSinceLastEvent                         -0.022184
#speedFromLastEvent                          0.021045
#shootingTeamMaxTimeOnIceOfForwards          0.020300
#defendingTeamMaxTimeOnIce                   0.018701
#shootingTeamMaxTimeOnIceOfDefencemen        0.017607
#shootingTeamAverageTimeOnIce                0.017152
#shootingTeamAverageTimeOnIceOfDefencemen    0.016675
#defendingTeamMaxTimeOnIceOfForwards         0.012642
#shootingTeamAverageTimeOnIceOfForwards      0.009534
#shotAnglePlusRebound                        0.009051
#defendingTeamMaxTimeOnIceOfDefencemen       0.007844
#defendingTeamAverageTimeOnIceOfDefencemen   0.006678

plot(nscFitImp, top=15, scales = list(y = list(cex = .85)))

### Predict the test set
goal.Results_nscFit <- data.frame(obs = Test.NHL_df$goal)
goal.Results_nscFit$prob <- predict(nscFit, Test.NHL_df, type = "prob")[, "Goal"]
goal.Results_nscFit$pred <- predict(nscFit, Test.NHL_df)
goal.Results_nscFit$Label <- ifelse(goal.Results_nscFit$obs == "Goal", 
                                      "True Outcome: goal", 
                                      "True Outcome: no goal")

goal.Results_nscFit$obs <- as.factor(goal.Results_nscFit$obs)

defaultSummary(goal.Results_plsFit)
#
# 

### Plot the probability 
histogram(~prob|Label,
          data = goal.Results_nscFit,
          layout = c(2, 1),
          nint = 20,
          xlab = "Probability of goal",
          type = "count")

### Calculate and plot the calibration curve
goal.Calib_glmnFit <- calibration(obs ~ prob, data = goal.Results_nscFit)
xyplot(goal.Calib_glmnFit)

### Create the confusion matrix from the test set.
confusionMatrix(data = goal.Results_nscFit$pred, 
                reference = goal.Results_nscFit$obs)

#Confusion Matrix and Statistics
#
#Reference
#Prediction  Goal NoGoal
#Goal       0      0
#NoGoal  4129  559423
#
#Accuracy : 0.9313          
#95% CI : (0.9292, 0.9333)
#No Information Rate : 0.9313          
#P-Value [Acc > NIR] : 0.5041          
#
#Kappa : 0               
#
#Mcnemar's Test P-Value : <2e-16          
#                                          
#            Sensitivity : 0.00000         
#            Specificity : 1.00000         
#         Pos Pred Value :     NaN         
#         Neg Pred Value : 0.93126         
#             Prevalence : 0.06874         
#         Detection Rate : 0.00000         
#   Detection Prevalence : 0.00000         
#      Balanced Accuracy : 0.50000         
#                                          
#       'Positive' Class : Goal  

### ROC curves:
goal.ROC_nscFit <- roc(response = goal.Results_nscFit$obs, predictor = goal.Results_nscFit$prob, levels = levels(goal.Results_nscFit$obs))
coords(goal.ROC_nscFit, "all")[,1:3]

auc(goal.ROC_nscFit) #Area under the curve: 0.7119

### Note the x-axis is reversed
plot(goal.ROC_nscFit)

### Lift charts
goal.Lift_nscFit <- lift(obs ~ prob, data = goal.Results_nscFit)
xyplot(goal.Lift_nscFit)



###############################
###############################

plot(goal.ROC_LR, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(goal.ROC_ldaFit, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(goal.ROC_plsFit, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(goal.ROC_glmnFit, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(goal.ROC_spLDAFit, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(goal.ROC_nscFit, type = "s", col = rgb(.2, .2, .2, .2), add = TRUE, legacy.axes = TRUE)

###############################
###############################

goalResamples <- resamples(list(glm = logisticReg,
                                lda = ldaFit,
                                pls = plsFit,
                                glmnet = glmnFit,
                                sparseLDA = spLDAFit,
                                pam = nscFit))

bwplot(goalResamples, metric = "ROC")

