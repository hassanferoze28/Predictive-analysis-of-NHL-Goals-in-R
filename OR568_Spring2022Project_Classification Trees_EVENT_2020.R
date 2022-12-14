################################################################################
##############################  Events 2020  ###################################
################################################################################
################################################################################
### Classification Trees
# 1 Basic Classification Tree
# 2 Bagged Tree
# 3 Random Forest
# 4 Boosting

install.packages('rpart.plot')

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
library(rpart.plot)
library(RWeka)


# Import the dataset
NHL_df_event = read.csv("C:/Users/m29336/Documents/NHL_df_event.csv", header=TRUE)
dim(NHL_df_event)        # 78582   88

NHL_df_event$event = as.factor(NHL_df_event$event) #change to factor
class(NHL_df_event$event)

#set Control
ctrl = trainControl(method = "repeatedcv", repeats = 5, summaryFunction=multiClassSummary, classProbs=TRUE )

## Split the data into training (80%) and test sets (20%)
set.seed(123)
inTrain_event <- createDataPartition(NHL_df_event$event, p = .8)[[1]]
Train.NHL_df_event <- NHL_df_event[ inTrain_event, ]
Test.NHL_df_event  <- NHL_df_event[-inTrain_event, ]

################################################################################
### 1 Basic Classification Tree

set.seed(123)
rpartFit <- train(event ~ .,
                  data = Train.NHL_df_event,
                  method = "rpart",
                  tuneLength = 30,
                  trControl = ctrl)
rpartFit

# CART 
# 
# 62867 samples
# 87 predictor
# 3 classes: 'GOAL', 'MISS', 'SHOT' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold) 
# Summary of sample sizes: 56580, 56580, 56581, 56579, 56579, 56581, ... 
# Resampling results across tuning parameters:
#   
#   cp            logLoss    AUC        prAUC      Accuracy   Kappa      Mean_F1    Mean_Sensitivity  Mean_Specificity  Mean_Pos_Pred_Value  Mean_Neg_Pred_Value
# 0.0002335248  0.6001541  0.8167646  0.6732462  0.7127270  0.3316026  0.6412268  0.6817868         0.7576557         0.6635876            0.8078145          
# 0.0002428658  0.5854470  0.8176626  0.6729208  0.7144448  0.3317160  0.6391327  0.6809483         0.7572136         0.6655850            0.8109950          
# 0.0002452011  0.5854470  0.8176626  0.6729208  0.7144448  0.3317160  0.6391327  0.6809483         0.7572136         0.6655850            0.8109950          
# 0.0002490931  0.5852153  0.8178175  0.6718163  0.7146198  0.3318219  0.6390489  0.6810910         0.7571941         0.6658593            0.8112916          
# 0.0002568773  0.5852365  0.8176358  0.6447970  0.7144289  0.3317243  0.6382677  0.6812322         0.7571970         0.6648216            0.8111471          
# 0.0002646615  0.5776724  0.8183985  0.6452160  0.7162582  0.3314903  0.6353273  0.6803746         0.7565640         0.6673536            0.8150647          
# 0.0002685535  0.5775854  0.8184107  0.6451952  0.7162423  0.3311341  0.6349471  0.6801705         0.7564157         0.6673618            0.8152454          
# 0.0002802298  0.5730822  0.8183292  0.6466766  0.7167831  0.3308924  0.6334399  0.6801002         0.7561338         0.6677337            0.8166367          
# 0.0002957981  0.5697492  0.8185541  0.6345875  0.7178170  0.3312889  0.6318805  0.6799830         0.7560199         0.6698537            0.8192870          
# 0.0003035823  0.5682577  0.8174694  0.6339705  0.7184691  0.3291602  0.6273859  0.6787924         0.7549613         0.6715950            0.8233710          
# 0.0003113664  0.5656684  0.8175991  0.6338086  0.7187872  0.3278095  0.6256428  0.6776672         0.7542728         0.6730156            0.8248128          
# 0.0003152585  0.5629230  0.8177335  0.6354691  0.7186600  0.3277207  0.6257582  0.6777285         0.7542660         0.6722835            0.8243626          
# 0.0003175938  0.5629230  0.8177335  0.6354691  0.7186600  0.3277207  0.6257582  0.6777285         0.7542660         0.6722835            0.8243626          
# 0.0003269348  0.5621518  0.8179617  0.6084832  0.7194075  0.3289584  0.6252753  0.6787230         0.7545194         0.6749323            0.8259227          
# 0.0003456167  0.5586422  0.8173167  0.5971354  0.7201550  0.3277704  0.6222752  0.6777697         0.7538045         0.6762477            0.8286498          
# 0.0003469512  0.5584141  0.8165170  0.5888216  0.7197893  0.3258581  0.6206180  0.6767901         0.7531185         0.6755011            0.8289835          
# 0.0003619635  0.5577228  0.8166116  0.5892318  0.7199165  0.3257702  0.6203287  0.6766699         0.7530440         0.6757675            0.8292983          
# 0.0003736397  0.5536027  0.8155847  0.5246094  0.7210300  0.3238071  0.6148008  0.6759642         0.7518393         0.6812720            0.8345655          
# 0.0004670496  0.5530843  0.8079929  0.3839264  0.7220798  0.3211285  0.6044932  0.6767301         0.7504160         0.6878200            0.8421496          
# 0.0004826180  0.5530255  0.8078926  0.3836558  0.7221116  0.3203812  0.6033974  0.6762965         0.7501070         0.6892277            0.8430800          
# 0.0005137546  0.5528516  0.8080743  0.3836298  0.7222707  0.3207515  0.6035816  0.6764230         0.7502129         0.6895300            0.8432866          
# 0.0005293229  0.5520273  0.8071669  0.3455739  0.7227957  0.3217384  0.6029351  0.6777592         0.7504477         0.6926368            0.8447338  ***        
# 0.0008873943  0.5530071  0.8006230  0.3419939  0.7225730  0.3186210  0.5989580  0.6761242         0.7492506         0.6954265            0.8471060          
# 0.0009340993  0.5530433  0.8005990  0.3419377  0.7225412  0.3185635  0.5989455  0.6761081         0.7492361         0.6951979            0.8470338          
# 0.0010041567  0.5530202  0.8006013  0.3420366  0.7225253  0.3185108  0.5989000  0.6760885         0.7492206         0.6950166            0.8470547          
# 0.0021017234  0.5530985  0.8006809  0.3414707  0.7221753  0.3179534  0.5988504  0.6759579         0.7490939         0.6924673            0.8462516          
# 0.0037831021  0.5556150  0.7960728  0.3365679  0.7194233  0.3082701  0.5905861  0.6720061         0.7459934         0.6860723            0.8459053          
# 0.0042034468  0.5562207  0.7952662  0.3348869  0.7183894  0.3071568  0.5913290  0.6718642         0.7457950         0.6782956            0.8429530          
# 0.0076596142  0.5577411  0.7927755  0.3027644  0.7164170  0.2909034  0.5825905  0.6640743         0.7400071         0.6635576            0.8538105          
# 0.0413222176  0.7129831  0.6156700  0.1087853  0.6811220  0.1115199        NaN  0.4634787         0.6943710               NaN            0.8560732          
# Mean_Precision  Mean_Recall  Mean_Detection_Rate  Mean_Balanced_Accuracy
# 0.6635876       0.6817868    0.2375757            0.7197213             
# 0.6655850       0.6809483    0.2381483            0.7190810             
# 0.6655850       0.6809483    0.2381483            0.7190810             
# 0.6658593       0.6810910    0.2382066            0.7191425             
# 0.6648216       0.6812322    0.2381430            0.7192146             
# 0.6673536       0.6803746    0.2387527            0.7184693             
# 0.6673618       0.6801705    0.2387474            0.7182931             
# 0.6677337       0.6801002    0.2389277            0.7181170             
# 0.6698537       0.6799830    0.2392723            0.7180015             
# 0.6715950       0.6787924    0.2394897            0.7168769             
# 0.6730156       0.6776672    0.2395957            0.7159700             
# 0.6722835       0.6777285    0.2395533            0.7159972             
# 0.6722835       0.6777285    0.2395533            0.7159972             
# 0.6749323       0.6787230    0.2398025            0.7166212             
# 0.6762477       0.6777697    0.2400517            0.7157871             
# 0.6755011       0.6767901    0.2399298            0.7149543             
# 0.6757675       0.6766699    0.2399722            0.7148569             
# 0.6812720       0.6759642    0.2403433            0.7139017             
# 0.6878200       0.6767301    0.2406933            0.7135731             
# 0.6892277       0.6762965    0.2407039            0.7132017             
# 0.6895300       0.6764230    0.2407569            0.7133179             
# 0.6926368       0.6777592    0.2409319            0.7141035             
# 0.6954265       0.6761242    0.2408577            0.7126874             
# 0.6951979       0.6761081    0.2408471            0.7126721             
# 0.6950166       0.6760885    0.2408418            0.7126546             
# 0.6924673       0.6759579    0.2407251            0.7125259             
# 0.6860723       0.6720061    0.2398078            0.7089998             
# 0.6782956       0.6718642    0.2394631            0.7088296             
# 0.6635576       0.6640743    0.2388057            0.7020407             
# NaN       0.4634787    0.2270407            0.5789249             
# 
# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was cp = 0.0005293229.


plot(as.party(rpartFit$finalModel))


rpartFitImp <- varImp(rpartFit, scale = FALSE)
rpartFitImp

# rpart variable importance
# 
# only 20 most important variables shown (out of 87)
# 
# Overall
# shotGeneratedRebound                              2909.860
# shotPlayContinuedInZone                           2855.306
# shotGoalieFroze                                   2186.767
# shotDistance                                      1184.159
# shotPlayContinuedOutsideZone                       940.654
# shotTypeTip                                        757.981
# arenaAdjustedXCordABS                              633.001
# arenaAdjustedYCordAbs                              352.115
# shotTypeWrist                                      141.195
# playerPositionThatDidEventD                        104.324
# shootingTeamForwardsOnIce                           70.466
# shotAngleAdjusted                                   34.232
# xCord                                               31.350
# defendingTeamMaxTimeOnIceOfDefencemen               15.870
# defendingTeamAverageTimeOnIceOfForwards             13.653
# shotRebound                                          7.914
# yCordAdjusted                                        7.019
# shotAnglePlusReboundSpeed                            6.470
# defendingTeamMaxTimeOnIceOfDefencemenSinceFaceoff    6.278
# defendingTeamAverageTimeOnIceSinceFaceoff            5.965

plot(rpartFitImp, top=15, scales = list(y = list(cex = .85)))

### Predict the test set
goal.Results_rpart <- data.frame(obs = Test.NHL_df_event$event)
goal.Results_rpart$prob <- predict(rpartFit, Test.NHL_df_event, type = "prob")[, "GOAL"]
goal.Results_rpart$pred <- predict(rpartFit, Test.NHL_df_event)
goal.Results_rpart$Label <- ifelse(goal.Results_rpart$obs == "GOAL",
                                "True Outcome: GOAL",
                                "True Outcome: MISS",
                                "True Outcome: SHOT")


goal.Results_rpart$obs <- as.factor(goal.Results_rpart$obs)

defaultSummary(goal.Results_rpart)
# Accuracy     Kappa 
# 0.7203309 0.3238141 


### Create the confusion matrix from the test set.
confusionMatrix(data = goal.Results_rpart$pred, 
                reference = goal.Results_rpart$obs)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction GOAL MISS SHOT
# GOAL 1094  175  233
# MISS    0  385  289
# SHOT    1 3697 9841
# 
# Overall Statistics
# 
# Accuracy : 0.7203          
# 95% CI : (0.7132, 0.7273)
# No Information Rate : 0.6594          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.3238          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
#                      Class: GOAL Class: MISS Class: SHOT
# Sensitivity              0.99909     0.09044      0.9496
# Specificity              0.97209     0.97478      0.3090
# Pos Pred Value           0.72836     0.57122      0.7269
# Neg Pred Value           0.99993     0.74257      0.7601
# Prevalence               0.06968     0.27089      0.6594
# Detection Rate           0.06962     0.02450      0.6262
# Detection Prevalence     0.09558     0.04289      0.8615
# Balanced Accuracy        0.98559     0.53261      0.6293           

### ROC curves:
goal.ROC_rpart <- multiclass.roc(response = goal.Results_rpart$obs, predictor = goal.Results_rpart$prob, levels = levels(goal.Results_rpart$obs))

auc(goal.ROC_rpart) # Multi-class area under the curve: 0.8563

rpart=list( classifier=rpartFit, roc=goal.ROC_rpart, auc=auc(goal.ROC_rpart) )
rpart


################################################################################
### 2 Bagged Tree

set.seed(123)
treebagFit <- train(event ~ .,
                    data = Train.NHL_df_event,
                    method = "treebag",
                    nbagg = 25,
                    trControl = ctrl)
treebagFit

# Bagged CART 
# 
# 62867 samples
# 87 predictor
# 3 classes: 'GOAL', 'MISS', 'SHOT' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold) 
# Summary of sample sizes: 56580, 56580, 56580, 56580, 56582, 56580, ... 
# Resampling results:
#   
#   logLoss    AUC        prAUC      Accuracy   Kappa      Mean_F1    Mean_Sensitivity  Mean_Specificity  Mean_Pos_Pred_Value  Mean_Neg_Pred_Value  Mean_Precision  Mean_Recall  Mean_Detection_Rate
# 0.6893578  0.8154696  0.6326446  0.7070803  0.3300302  0.6479685  0.6811186         0.7586761         0.6585105            0.7972378            0.6585105       0.6811186    0.2356934          
# Mean_Balanced_Accuracy
# 0.7198973   

treebagFitImp <- varImp(treebagFit, scale = FALSE)
treebagFitImp

# treebag variable importance
# 
# only 20 most important variables shown (out of 87)
# 
# Overall
# shotDistance               4341
# time                       3642
# shotGeneratedRebound       2888
# shotPlayContinuedInZone    2839
# shotAngle                  2748
# speedFromLastEvent         2735
# shotAngleAdjusted          2659
# xCord                      2659
# timeSinceLastEvent         2610
# distanceFromLastEvent      2515
# yCordAdjusted              2495
# arenaAdjustedXCordABS      2342
# playerNumThatDidEvent      2229
# shotGoalieFroze            2189
# lastEventxCord             1973
# lastEventyCord             1916
# arenaAdjustedYCordAbs      1820
# timeSinceFaceoff           1809
# lastEventxCord_adjusted    1794
# lastEventyCord_adjusted    1727

plot(treebagFitImp, top=15, scales = list(y = list(cex = .85)))

### Predict the test set
goal.Results_treebag <- data.frame(obs = Test.NHL_df_event$event)
goal.Results_treebag$prob <- predict(treebagFit, Test.NHL_df_event, type = "prob")[, "GOAL"]
goal.Results_treebag$pred <- predict(treebagFit, Test.NHL_df_event)
goal.Results_treebag$Label <- ifelse(goal.Results_treebag$obs == "GOAL",
                                     "True Outcome: GOAL",
                                     "True Outcome: MISS",
                                     "True Outcome: SHOT")


goal.Results_treebag$obs <- as.factor(goal.Results_treebag$obs)

defaultSummary(goal.Results_treebag)

# Accuracy     Kappa 
# 0.7049952 0.3267469 


### Create the confusion matrix from the test set.
confusionMatrix(data = goal.Results_treebag$pred, 
                reference = goal.Results_treebag$obs)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction GOAL MISS SHOT
# GOAL 1047  138  186
# MISS   16  853  998
# SHOT   32 3266 9179
# 
# Overall Statistics
# 
# Accuracy : 0.705           
# 95% CI : (0.6978, 0.7121)
# No Information Rate : 0.6594          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.3267          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
#                      Class: GOAL Class: MISS Class: SHOT
# Sensitivity              0.95616     0.20038      0.8857
# Specificity              0.97784     0.91150      0.3838
# Pos Pred Value           0.76368     0.45688      0.7357
# Neg Pred Value           0.99665     0.75419      0.6343
# Prevalence               0.06968     0.27089      0.6594
# Detection Rate           0.06662     0.05428      0.5841
# Detection Prevalence     0.08724     0.11880      0.7940
# Balanced Accuracy        0.96700     0.55594      0.6348 



### ROC curves:
goal.ROC_treebag <- multiclass.roc(response = goal.Results_treebag$obs, predictor = goal.Results_treebag$prob, levels = levels(goal.Results_treebag$obs))

auc(goal.ROC_treebag) # Multi-class area under the curve: 0.8249

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
#                ntree = 100,
#                tuneGrid = data.frame(mtry = mtryValues),
#                importance = TRUE,
#                metric = "ROC",
#                trControl = ctrl)
#  rfFit
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
#                        n.minobsinnode = 10,
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

###############################
###############################

plot.roc(goal.ROC_rpart[['rocs']][[1]], type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot.roc(goal.ROC_treebag[['rocs']][[1]], type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)

goalResamples <- resamples(list(rpart = rpartFit,
                                treebag = treebagFit))

bwplot(goalResamples, metric = "Accuracy")
