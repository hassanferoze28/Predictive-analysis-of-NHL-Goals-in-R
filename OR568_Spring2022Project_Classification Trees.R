
################################################################################
#########################  Goal-No Goal 2018-2020  #############################
################################################################################
################################################################################
### Classification Trees
# 1 Basic Classification Tree
# 2 Bagged Tree
# 3 Random Forest
# 4 Boosting
# 5 C50

library(caret)
library(partykit)
library(pROC)
library(C50)
library(doMC)
library(gbm)
library(lattice)
library(randomForest)
library(reshape2)
library(rpart)
library(RWeka)


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
### 1 Basic Classification Tree

set.seed(123)
rpartFit <- train(goal ~ .,
                  data = Train.NHL_df,
                  method = "rpart",
                  tuneLength = 30,
                  metric = "ROC",
                  trControl = ctrl)
rpartFit

#CART 
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
#        cp            ROC        Sens       Spec     
#8.072980e-05  0.9899089  0.8988327  0.9897429
#8.325260e-05  0.9899212  0.8994901  0.9897303
#9.082102e-05  0.9899963  0.9013709  0.9897021
#9.687576e-05  0.9900576  0.9033500  0.9896748
#1.009122e-04  0.9900720  0.9045255  0.9896607
#1.059579e-04  0.9901386  0.9055339  0.9896535
#1.210947e-04  0.9906491  0.9097141  0.9896107
#1.332042e-04  0.9910676  0.9124799  0.9895630
#1.513684e-04  0.9914445  0.9180780  0.9894965
#1.614596e-04  0.9916737  0.9217280  0.9894600
#1.816420e-04  0.9923891  0.9272231  0.9893754
#2.018245e-04  0.9928677  0.9324563  0.9892611
#2.043473e-04  0.9929054  0.9333226  0.9892334
#2.270526e-04  0.9933846  0.9380027  0.9891347
#2.421894e-04  0.9937076  0.9403639  0.9890740
#2.623718e-04  0.9939512  0.9428078  0.9890039
#3.027367e-04  0.9944680  0.9494427  0.9888198
#3.632841e-04  0.9951198  0.9566220  0.9885836
#4.238314e-04  0.9951327  0.9616839  0.9883366
#4.541051e-04  0.9952088  0.9631186  0.9882471
#4.843788e-04  0.9951880  0.9643119  0.9881904
#5.449261e-04  0.9953442  0.9658972  0.9880864 ***
#5.751998e-04  0.9952238  0.9662899  0.9879986
#6.054735e-04  0.9952668  0.9659534  0.9880253
#7.265682e-04  0.9952750  0.9680994  0.9878601
#7.871155e-04  0.9952954  0.9689556  0.9878116
#1.029305e-03  0.9949941  0.9777037  0.9871262
#1.049487e-03  0.9949932  0.9771757  0.9871578
#1.256357e-03  0.9945229  0.9821651  0.9866895
#1.587551e-01  0.6965154  0.3998807  0.9930537
#
#ROC was used to select the optimal model using the largest value.
#The final value used for the model was cp = 0.0005449261.

plot(as.party(rpartFit$finalModel))

rpartFitImp <- varImp(rpartFit, scale = FALSE)
rpartFitImp

#rpart variable importance
#
#only 20 most important variables shown (out of 90)
#
#Overall
#shotGeneratedRebound                  10352.33
#shotGoalieFroze                        9337.47
#shotWasOnGoal                          7624.04
#arenaAdjustedShotDistance              5894.82
#shotPlayContinuedInZone                4748.09
#arenaAdjustedXCordABS                  3737.24
#arenaAdjustedYCordAbs                  3533.37
#shotPlayContinuedOutsideZone           1339.11
#yCord                                   488.79
#defendingTeamForwardsOnIce              230.96
#yCordAdjusted                           154.53
#shootingTeamDefencemenOnIce             116.65
#shootingTeamForwardsOnIce                94.65
#awaySkatersOnIce                         84.38
#time                                     81.15
#homePenalty1Length                       63.15
#shotAngleAdjusted                        50.30
#homeSkatersOnIce                         23.95
#defendingTeamMaxTimeOnIceOfDefencemen    20.55
#lastEventyCord_adjusted                  18.72

plot(rpartFitImp, top=15, scales = list(y = list(cex = .85)))

### Predict the test set
goal.Results_rpart <- data.frame(obs = Test.NHL_df$goal)
goal.Results_rpart$prob <- predict(rpartFit, Test.NHL_df, type = "prob")[, "Goal"]
goal.Results_rpart$pred <- predict(rpartFit, Test.NHL_df)
goal.Results_rpart$Label <- ifelse(goal.Results_rpart$obs == "Goal", 
                             "True Outcome: goal", 
                             "True Outcome: no goal")

goal.Results_rpart$obs <- as.factor(goal.Results_rpart$obs)

defaultSummary(goal.Results_rpart)
#Accuracy     Kappa 
#0.9854506 0.8937628 

### Plot the probability
histogram(~prob|Label,
          data = goal.Results_rpart,
          layout = c(2, 1),
          nint = 20,
          xlab = "Probability of goal",
          type = "count")

### Calculate and plot the calibration curve
goal.Calib_rpart <- calibration(obs ~ prob, data = goal.Results_rpart)
xyplot(goal.Calib_rpart)


### Create the confusion matrix from the test set.
confusionMatrix(data = goal.Results_rpart$pred, 
                reference = goal.Results_rpart$obs)

#Confusion Matrix and Statistics
#
#Reference
#Prediction  Goal NoGoal
#Goal    4003    748
#NoGoal   126  55194
#
#Accuracy : 0.9855          
#95% CI : (0.9845, 0.9864)
#No Information Rate : 0.9313          
#P-Value [Acc > NIR] : < 2.2e-16       
#
#Kappa : 0.8938          
#
#Mcnemar's Test P-Value : < 2.2e-16       
#                                          
#            Sensitivity : 0.96948         
#            Specificity : 0.98663         
#         Pos Pred Value : 0.84256         
#         Neg Pred Value : 0.99772         
#             Prevalence : 0.06874         
#         Detection Rate : 0.06664         
#   Detection Prevalence : 0.07909         
#      Balanced Accuracy : 0.97806         
#                                          
#       'Positive' Class : Goal             

### ROC curves:
goal.ROC_rpart <- roc(response = goal.Results_rpart$obs, predictor = goal.Results_rpart$prob, levels = levels(goal.Results_rpart$obs))
coords(goal.ROC_rpart, "all")[,1:3]

auc(goal.ROC_rpart) #Area under the curve: 0.996
ci.auc(goal.ROC_rpart) #95% CI: 0.9944-0.9953 (DeLong)

### Note the x-axis is reversed
plot(goal.ROC_rpart)

### Lift charts
goal.Lift_rpart <- lift(obs ~ prob, data = goal.Results_rpart)
xyplot(goal.Lift_rpart)

rpart=list( classifier=rpartFit, roc=goal.ROC_rpart, auc=auc(goal.ROC_rpart) )
rpart

################################################################################
### 2 Bagged Tree

set.seed(123)
treebagFit <- train(goal ~ .,
                    data = Train.NHL_df,
                    method = "treebag",
                    nbagg = 25,
                    metric = "ROC",
                    trControl = ctrl)
treebagFit

#Bagged CART 
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
#0.9962032  0.9581926  0.9887134


treebagFitImp <- varImp(treebagFit, scale = FALSE)
treebagFitImp

#treebag variable importance
#
#only 20 most important variables shown (out of 90)
#
#Overall
#shotGeneratedRebound         10377.0
#shotGoalieFroze               9354.6
#shotWasOnGoal                 7646.5
#arenaAdjustedShotDistance     6186.9
#shotPlayContinuedInZone       4758.2
#arenaAdjustedXCordABS         4239.5
#arenaAdjustedYCordAbs         3623.2
#shotPlayContinuedOutsideZone  1344.0
#yCord                          878.4
#time                           847.7
#yCordAdjusted                  783.8
#shotAngleAdjusted              566.3
#speedFromLastEvent             558.4
#timeSinceLastEvent             557.7
#shotAngle                      518.4
#distanceFromLastEvent          501.4
#xCord                          493.7
#playerNumThatDidEvent          453.3
#lastEventxCord                 430.1
#lastEventyCord                 399.2

plot(treebagFitImp, top=15, scales = list(y = list(cex = .85)))

### Predict the test set
goal.Results_treebag <- data.frame(obs = Test.NHL_df$goal)
goal.Results_treebag$prob <- predict(treebagFit, Test.NHL_df, type = "prob")[, "Goal"]
goal.Results_treebag$pred <- predict(treebagFit, Test.NHL_df)
goal.Results_treebag$Label <- ifelse(goal.Results_treebag$obs == "Goal", 
                                   "True Outcome: goal", 
                                   "True Outcome: no goal")

goal.Results_treebag$obs <- as.factor(goal.Results_treebag$obs)

defaultSummary(goal.Results_treebag)

#Accuracy     Kappa 
#0.9855837 0.8942474 

### Plot the probability
histogram(~prob|Label,
          data = goal.Results_treebag,
          layout = c(2, 1),
          nint = 20,
          xlab = "Probability of goal",
          type = "count")


### Calculate and plot the calibration curve
goal.Calib_treebag <- calibration(obs ~ prob, data = goal.Results_treebag)
xyplot(goal.Calib_treebag)

### Create the confusion matrix from the test set.
confusionMatrix(data = goal.Results_treebag$pred, 
                reference = goal.Results_treebag$obs)

#Confusion Matrix and Statistics
#
#Reference
#Prediction  Goal NoGoal
#Goal    3985    722
#NoGoal   144  55220
#
#Accuracy : 0.9856          
#95% CI : (0.9846, 0.9865)
#No Information Rate : 0.9313          
#P-Value [Acc > NIR] : < 2.2e-16       
#
#Kappa : 0.8942          
#
#Mcnemar's Test P-Value : < 2.2e-16       
#                                          
#            Sensitivity : 0.96512         
#            Specificity : 0.98709         
#         Pos Pred Value : 0.84661         
#         Neg Pred Value : 0.99740         
#             Prevalence : 0.06874         
#         Detection Rate : 0.06634         
#   Detection Prevalence : 0.07836         
#      Balanced Accuracy : 0.97611         
#                                          
#       'Positive' Class : Goal 

### ROC curves:
goal.ROC_treebag <- roc(response = goal.Results_treebag$obs, predictor = goal.Results_treebag$prob, levels = levels(goal.Results_treebag$obs))
coords(goal.ROC_treebag, "all")[,1:3]

auc(goal.ROC_treebag) # Area under the curve: 0.9956
ci.auc(goal.ROC_treebag) #95% CI: 0.9952-0.9961 (DeLong)

### Note the x-axis is reversed
plot(goal.ROC_treebag)


### Lift charts
goal.Lift_treebag <- lift(obs ~ prob, data = goal.Results_treebag)
xyplot(goal.Lift_treebag)

treebag=list( classifier=treebagFit, roc=goal.ROC_treebag, auc=auc(goal.ROC_treebag) )
treebag


# ################################################################################
# ### 3 Random Forest
# 
# mtryValues <- c(5, 10, 20, 32, 50, 100, 250, 500, 1000)
# set.seed(123)
# rfFit <- train(goal ~ .,
#                data = Train.NHL_df,
#                method = "rf",
#                ntree = 500,
#                tuneGrid = data.frame(mtry = mtryValues),
#                importance = TRUE,
#                metric = "ROC",
#                trControl = ctrl)
# rfFit
# 
# # [insert output]
# #
# #
# #
# #
# 
# 
# plot(as.party(rfFit$finalModel))
# 
# # [save plot]
# #
# #
# #
# #
# 
# rfFitImp <- varImp(rfFit, scale = FALSE)
# rfFitImp
# 
# # [insert output]
# #
# #
# #
# #
# 
# plot(treebagFitImp, top=15, scales = list(y = list(cex = .85)))
# 
# # [save plot]
# #
# #
# #
# #
# 
# ### Predict the test set
# goal.Results_rfFit <- data.frame(obs = Test.NHL_df$goal)
# goal.Results_rfFit$prob <- predict(rfFit, Test.NHL_df, type = "prob")[, "Goal"]
# goal.Results_rfFit$pred <- predict(rfFit, Test.NHL_df)
# goal.Results_rfFit$Label <- ifelse(goal.Results_rfFit$obs == "Goal", 
#                                      "True Outcome: goal", 
#                                      "True Outcome: no goal")
# 
# goal.Results_rfFit$obs <- as.factor(goal.Results_rfFit$obs)
# 
# defaultSummary(goal.Results_rfFit)
# 
# # [insert output]
# #
# #
# #
# #
# 
# ### Plot the probability
# histogram(~prob|Label,
#           data = goal.Results_rfFit,
#           layout = c(2, 1),
#           nint = 20,
#           xlab = "Probability of goal",
#           type = "count")
# 
# # [save plot]
# #
# #
# #
# #
# 
# ### Calculate and plot the calibration curve
# goal.Calib_rfFit <- calibration(obs ~ prob, data = goal.Results_rfFit)
# xyplot(goal.Calib_rfFit)
# 
# # [save plot]
# #
# #
# #
# #
# 
# ### Create the confusion matrix from the test set.
# confusionMatrix(data = goal.Results_rfFit$pred, 
#                 reference = goal.Results_rfFit$obs)
# 
# # [insert output]
# #
# #
# #
# #
# 
# ### ROC curves:
# goal.ROC_rfFit <- roc(response = goal.Results_rfFit$obs, predictor = goal.Results_rfFit$prob, levels = levels(goal.Results_rfFit$obs))
# coords(goal.ROC_rfFit, "all")[,1:3]
# 
# auc(goal.ROC_rfFit) # [insert output]
# ci.auc(goal.ROC_rfFit) # [insert output]
# 
# ### Note the x-axis is reversed
# plot(goal.ROC_rfFit)
# 
# # [save plot]
# #
# #
# #
# #
# 
# ### Lift charts
# goal.Lift_rfFit <- lift(obs ~ prob, data = goal.Results_rfFit)
# xyplot(goal.Lift_rfFit)
# 
# # [save plot]
# #
# #
# #
# #
# 
# rf=list( classifier=rfFit, roc=goal.ROC_rfFit, auc=auc(goal.ROC_rfFit) )
# rf
# 
# ################################################################################
# ### 4 Boosting
# 
# gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
#                        n.trees = (1:20)*100,
#                        shrinkage = c(.01, .1))
# 
# set.seed(123)
# gbmFit <- train(goal ~ .,
#                 data = Train.NHL_df,
#                 method = "gbm",
#                 tuneGrid = gbmGrid,
#                 metric = "ROC",
#                 verbose = FALSE,
#                 trControl = ctrl)
# gbmFit
# 
# # [insert output]
# #
# #
# #
# #
# 
# 
# plot(as.party(gbmFit$finalModel))
# 
# # [save plot]
# #
# #
# #
# #
# 
# gbmFitImp <- varImp(gbmFit, scale = FALSE)
# gbmFitImp
# 
# # [insert output]
# #
# #
# #
# #
# 
# plot(gbmFitImp, top=15, scales = list(y = list(cex = .85)))
# 
# # [save plot]
# #
# #
# #
# #
# 
# ### Predict the test set
# goal.Results_gbmFit <- data.frame(obs = Test.NHL_df$goal)
# goal.Results_gbmFit$prob <- predict(gbmFit, Test.NHL_df, type = "prob")[, "Goal"]
# goal.Results_gbmFit$pred <- predict(gbmFit, Test.NHL_df)
# goal.Results_gbmFit$Label <- ifelse(goal.Results_gbmFit$obs == "Goal", 
#                                    "True Outcome: goal", 
#                                    "True Outcome: no goal")
# 
# goal.Results_gbmFit$obs <- as.factor(goal.Results_gbmFit$obs)
# 
# defaultSummary(goal.Results_gbmFit)
# 
# # [insert output]
# #
# #
# #
# #
# 
# ### Plot the probability
# histogram(~prob|Label,
#           data = goal.Results_gbmFit,
#           layout = c(2, 1),
#           nint = 20,
#           xlab = "Probability of goal",
#           type = "count")
# 
# # [save plot]
# #
# #
# #
# #
# 
# ### Calculate and plot the calibration curve
# goal.Calib_gbmFit <- calibration(obs ~ prob, data = goal.Results_gbmFit)
# xyplot(goal.Calib_gbmFit)
# 
# # [save plot]
# #
# #
# #
# #
# 
# ### Create the confusion matrix from the test set.
# confusionMatrix(data = goal.Results_gbmFit$pred, 
#                 reference = goal.Results_gbmFit$obs)
# 
# # [insert output]
# #
# #
# #
# #
# 
# ### ROC curves:
# goal.ROC_gbmFit <- roc(response = goal.Results_gbmFit$obs, predictor = goal.Results_gbmFit$prob, levels = levels(goal.Results_gbmFit$obs))
# coords(goal.ROC_gbmFit, "all")[,1:3]
# 
# auc(goal.ROC_gbmFit) # [insert output]
# ci.auc(goal.ROC_gbmFit) # [insert output]
# 
# ### Note the x-axis is reversed
# plot(goal.ROC_gbmFit)
# 
# # [save plot]
# #
# #
# #
# #
# 
# ### Lift charts
# goal.Lift_gbmFit <- lift(obs ~ prob, data = goal.Results_gbmFit)
# xyplot(goal.Lift_gbmFit)
# 
# # [save plot]
# #
# #
# #
# #
# 
# gbm=list( classifier=gbmFit, roc=goal.ROC_gbmFit, auc=auc(goal.ROC_gbmFit) )
# gbm
# 
# 
# ################################################################################
# ### 5 C50
# 
# c50Grid <- expand.grid(trials = c(1:9, (1:10)*10),
#                        model = c("tree", "rules"),
#                        winnow = c(TRUE, FALSE))
# set.seed(476)
# c50Fit <- train(goal ~ .,
#                 data = Train.NHL_df,
#                 method = "C5.0",
#                 tuneGrid = c50Grid,
#                 verbose = FALSE,
#                 metric = "ROC",
#                 trControl = ctrl)
# c50Fit
# 
# # [insert output]
# #
# #
# #
# #
# 
# 
# plot(as.party(c50Fit$finalModel))
# 
# # [save plot]
# #
# #
# #
# #
# 
# c50FitImp <- varImp(c50Fit, scale = FALSE)
# c50FitImp
# 
# # [insert output]
# #
# #
# #
# #
# 
# plot(c50FitImp, top=15, scales = list(y = list(cex = .85)))
# 
# # [save plot]
# #
# #
# #
# #
# 
# ### Predict the test set
# goal.Results_c50Fit <- data.frame(obs = Test.NHL_df$goal)
# goal.Results_c50Fit$prob <- predict(c50Fit, Test.NHL_df, type = "prob")[, "Goal"]
# goal.Results_c50Fit$pred <- predict(c50Fit, Test.NHL_df)
# goal.Results_c50Fit$Label <- ifelse(goal.Results_c50Fit$obs == "Goal", 
#                                     "True Outcome: goal", 
#                                     "True Outcome: no goal")
# 
# goal.Results_c50Fit$obs <- as.factor(goal.Results_c50Fit$obs)
# 
# defaultSummary(goal.Results_c50Fit)
# 
# # [insert output]
# #
# #
# #
# #
# 
# ### Plot the probability
# histogram(~prob|Label,
#           data = goal.Results_c50Fit,
#           layout = c(2, 1),
#           nint = 20,
#           xlab = "Probability of goal",
#           type = "count")
# 
# # [save plot]
# #
# #
# #
# #
# 
# ### Calculate and plot the calibration curve
# goal.Calib_c50Fit <- calibration(obs ~ prob, data = goal.Results_c50Fit)
# xyplot(goal.Calib_c50Fit)
# 
# # [save plot]
# #
# #
# #
# #
# 
# ### Create the confusion matrix from the test set.
# confusionMatrix(data = goal.Results_c50Fit$pred, 
#                 reference = goal.Results_c50Fit$obs)
# 
# # [insert output]
# #
# #
# #
# #
# 
# ### ROC curves:
# goal.ROC_c50Fit <- roc(response = goal.Results_c50Fit$obs, predictor = goal.Results_c50Fit$prob, levels = levels(goal.Results_c50Fit$obs))
# coords(goal.ROC_c50Fit, "all")[,1:3]
# 
# auc(goal.ROC_c50Fit) # [insert output]
# ci.auc(goal.ROC_c50Fit) # [insert output]
# 
# ### Note the x-axis is reversed
# plot(goal.ROC_c50Fit)
# 
# # [save plot]
# #
# #
# #
# #
# 
# ### Lift charts
# goal.Lift_c50Fit <- lift(obs ~ prob, data = goal.Results_c50Fit)
# xyplot(goal.Lift_c50Fit)
# 
# # [save plot]
# #
# #
# #
# #
# 
# c50=list( classifier=c50Fit, roc=goal.ROC_c50Fit, auc=auc(goal.ROC_c50Fit) )
# c50
# 


###############################
###############################

plot.roc(goal.ROC_rpart, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot.roc(goal.ROC_treebag, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
#plot.roc(goal.ROC_rf, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
#plot.roc(goal.ROC_gbm, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
#plot.roc(goal.ROC_C50, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
#plot.roc(goal.ROC_j48, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
#plot.roc(goal.ROC_part, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)

###############################
###############################

goalResamples <- resamples(list(rpart = rpartFit,
                                treebag = treebagFit))

bwplot(goalResamples, metric = "ROC")

result.final = list( rpart=rpart, treebag=treebag)

#result.final = list( rpart=rpart, treebag=treebag, rf=rf, gbm=gbm, C50=C50, j48=j48, part=part )
