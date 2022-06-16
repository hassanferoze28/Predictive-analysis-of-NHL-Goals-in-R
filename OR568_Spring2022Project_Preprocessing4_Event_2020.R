#' ---
#' author: "Daniel Turissini"
#' title: "OR 568 - Spring 2022, Prof Xu"
#' Project
#' Due: May 5, 2022 11:59pm
#'

################################################################################
##############################  Events 2020  ###################################
################################################################################
################################################################################
##### Data Preprocessing #####

library(dplyr)
library(caret)
library(e1071)
library(corrplot)

# Import the dataset
df = read.csv("C:/Users/m29336/Downloads/shots_2018-2020.csv", header=TRUE)
#df=shots_2018.2020
dim(df) # 300405    124

df2020 <- subset(df, season!="2018" & season!="2019")
dim(df2020) # 78611    124

### Taking care of missing data ###
MissingData <- unlist(lapply(df, function(x) any(is.na(x))))
MissingData <- names(MissingData)[MissingData]
head(MissingData)

# (delete NA because they are player IDs and are a fairly small percentage)
sum(is.na(df2020)) # 29
summary(is.na(df2020)) # shooterPlayerId TRUE:29
dfrmNA = df2020[complete.cases(df2020), ]
dim(dfrmNA)    # 78582   124

#drop season, only 2020
dfrmNA$season = NULL

#drop calculated probability predictor
dfrmNA$xGoal = NULL
dfrmNA$xFroze = NULL
dfrmNA$xRebound = NULL
dfrmNA$xPlayContinuedInZone = NULL
dfrmNA$xPlayContinuedOutsideZone = NULL
dfrmNA$xPlayStopped = NULL
dfrmNA$xShotWasOnGoal = NULL

#drop,predictor always 0 for goal
dfrmNA$timeUntilNextEvent = NULL

### Encoding categorical data using dummy variables ###
dfrmNA$teamHome = ifelse(dfrmNA$team == 'HOME', 1, 0)
# dfrmNA$teamAway = ifelse(dfrmNA$team == 'AWAY', 1, 0)

dfrmNA$locationHomezone = ifelse(dfrmNA$location == 'HOMEZONE', 1, 0)
# dfrmNA$locationAwayzone = ifelse(dfrmNA$location == 'AWAYZONE', 1, 0)

#remove goal, since proxy for event (class)
dfrmNA$goal = NULL
dfrmNA$shotWasOnGoal = NULL

dfrmNA$lastEventTeamHome = ifelse(dfrmNA$lastEventTeam == 'HOME', 1, 0)
# dfrmNA$lastEventTeamAway = ifelse(dfrmNA$lastEventTeam == 'AWAY', 1, 0)

dfrmNA$shotTypeBack = ifelse(dfrmNA$shotType == 'BACK', 1, 0)
dfrmNA$shotTypeDefl = ifelse(dfrmNA$shotType == 'DEFL', 1, 0)
dfrmNA$shotTypeSlap = ifelse(dfrmNA$shotType == 'SLAP', 1, 0)
dfrmNA$shotTypeSnap = ifelse(dfrmNA$shotType == 'SNAP', 1, 0)
dfrmNA$shotTypeTip =  ifelse(dfrmNA$shotType == 'TIP', 1, 0)
dfrmNA$shotTypeWrap = ifelse(dfrmNA$shotType == 'WRAP', 1, 0)
dfrmNA$shotTypeWrist =ifelse(dfrmNA$shotType == 'WRIST', 1, 0)

dfrmNA$shooterL = ifelse(dfrmNA$shooterLeftRight == 'L', 1, 0)
# dfrmNA$shooterR = ifelse(dfrmNA$shooterLeftRight == 'R', 1, 0)

dfrmNA$playerPositionThatDidEventC = ifelse(dfrmNA$playerPositionThatDidEvent == 'C', 1, 0)
dfrmNA$playerPositionThatDidEventD = ifelse(dfrmNA$playerPositionThatDidEvent == 'D', 1, 0)
dfrmNA$playerPositionThatDidEventG = ifelse(dfrmNA$playerPositionThatDidEvent == 'G', 1, 0)
dfrmNA$playerPositionThatDidEventL = ifelse(dfrmNA$playerPositionThatDidEvent == 'L', 1, 0)
dfrmNA$playerPositionThatDidEventR = ifelse(dfrmNA$playerPositionThatDidEvent == 'R', 1, 0)

# Convert IDs to categorical data
catCols = c("shotID", "game_id", "id", "goalieIdForShot", "shooterPlayerId")
dfrmNA[catCols] = lapply(dfrmNA[catCols], factor)

dim(dfrmNA) # 78582  128

dfrmNA_num1 = subset(dfrmNA,select= -event)
event = subset(dfrmNA,select="event")
dim(dfrmNA_num1) # 78582  128

dfrmNA_num = unlist(lapply(dfrmNA_num1, is.numeric))
dfrmNA_num = dfrmNA_num1[, dfrmNA_num]
dim(dfrmNA_num) # 78582  111

##### TEST MODELS WITH AND WITHOUT nearZeroVar#############
# Use the nearZeroVar function to filter out predictors with low frequencies
zeroVarCols = nearZeroVar(dfrmNA_num, names = TRUE)
zeroVarCols #15

allCols = names(dfrmNA_num)
NHL_filter = dfrmNA_num[, setdiff(allCols, zeroVarCols)]
dim(NHL_filter) # 78582    96


### 1 Address Skew ### FILTERED ###
# Distributions of the remaining predictors (random sampling)
set.seed(123)
sample_NHL_filter <- NHL_filter[, sample(1:ncol(NHL_filter), 8)]
names(sample_NHL_filter)
#[1] "playerNumThatDidEvent"                               
#[2] "arenaAdjustedYCord"                                  
#[3] "shootingTeamMaxTimeOnIceSinceFaceoff"                
#[4] "xCordAdjusted"                                       
#[5] "defendingTeamAverageTimeOnIceSinceFaceoff"           
#[6] "shootingTeamMaxTimeOnIce"                            
#[7] "shootingTeamAverageTimeOnIceOfDefencemenSinceFaceoff"
#[8] "shootingTeamMaxTimeOnIceOfForwards"  

#Look at skew (SAMPLE)
skew_df_sample1 <- apply(sample_NHL_filter, 2, skewness)
summary(skew_df_sample1)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#-0.5059  0.5308  0.8559  3.0031  1.5884 18.8681 

### 1 Transform ### FILTERED ###
#center and scale the data, then apply the transformation (SAMPLE)
centerScale_sample1 <- preProcess(sample_NHL_filter, method = c("center", "scale"))
df_transform_sample1 <- predict(centerScale_sample1, newdata = sample_NHL_filter)
df_SS_sample1 <- spatialSign(df_transform_sample1)
#splom(~df_SS_sample1, pch = 16, col = rgb(.2, .2, .2, .4), cex = .7)

#Look at skew full data
skew_df1 <- apply(NHL_filter, 2, skewness)
summary(skew_df1)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -5.0670  0.0858  1.0721  2.7948  3.3564 18.8681 

#center and scale the data, then apply the transformation
centerScale1 <- preProcess(NHL_filter, method = c("center", "scale"))
NHL_transform1 <- predict(centerScale1, newdata = NHL_filter)
df_SS <- spatialSign(NHL_transform1)

### 2 Address Skew ### UNFILTERED ###
# Distributions of the remaining predictors (random sampling)
set.seed(123)
sample_dfrmNA <- dfrmNA_num[, sample(1:ncol(dfrmNA_num), 8)]
names(sample_dfrmNA)
#[1] "lastEventShotDistance"                              
#[2] "defendingTeamAverageTimeOnIceOfForwardsSinceFaceoff"
#[3] "shootingTeamAverageTimeOnIceOfDefencemen"           
#[4] "yCord"                                              
#[5] "defendingTeamForwardsOnIce"                         
#[6] "lastEventxCord_adjusted"                            
#[7] "shootingTeamAverageTimeOnIceOfForwards"             
#[8] "lastEventyCord_adjusted"  

#Look at skew (SAMPLE)
skew_df_sample2 <- apply(sample_dfrmNA, 2, skewness)
summary(skew_df_sample2)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#-0.8935 -0.2089  1.1756  3.5855  4.3640 16.3960 


### 2 Transform ### UNFILTERED ###
#center and scale the data, then apply the transformation (SAMPLE)
centerScale_sample2 <- preProcess(sample_dfrmNA, method = c("center", "scale"))
df_transform_sample2 <- predict(centerScale_sample2, newdata = sample_dfrmNA)
df_SS_sample2 <- spatialSign(df_transform_sample2)
#splom(~df_SS_sample2, pch = 16, col = rgb(.2, .2, .2, .4), cex = .7)

#Look at skew full data
skew_df2 <- apply(dfrmNA_num, 2, skewness)
summary(skew_df2)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -5.0670  0.1268  1.2010  4.0576  4.2592 93.4238 

#center and scale the data, then apply the transformation
centerScale2 <- preProcess(dfrmNA_num, method = c("center", "scale"))
NHL_transform2 <- predict(centerScale2, newdata = dfrmNA_num)
df_SS <- spatialSign(NHL_transform2)

###Correlation Matrices###

filterCorr = cor(NHL_filter)
dim(filterCorr) #96

filtertransCorr = cor(NHL_transform1)
dim(filtertransCorr) #96

transCorr <- cor(NHL_transform2)
dim(transCorr) #111

ssData = spatialSign(scale(NHL_filter))
ssCorr = cor(ssData)

## plot the matrix with no labels or grid
corrplot(filterCorr, order = "hclust", addgrid.col = NA, tl.pos = "n")
corrplot(filtertransCorr, order = "hclust", addgrid.col = NA, tl.pos = "n")
ssData = spatialSign(scale(NHL_filter))
ssCorr = cor(ssData)
corrplot(ssCorr, order = "hclust", addgrid.col = NA, tl.pos = "n")

corrplot(filterCorr, order = "hclust", addgrid.col = rgb(.2, .2, .2, 0), tl.pos = "n")
corrplot(filtertransCorr, order = "hclust", addgrid.col = rgb(.2, .2, .2, 0), tl.pos = "n")
corrplot(ssCorr, order = "hclust", addgrid.col = rgb(.2, .2, .2, 0), tl.pos = "n")
corrplot(transCorr, order = "hclust", addgrid.col = rgb(.2, .2, .2, 0), tl.pos = "n")

corrInfo <- function(x) summary(x[upper.tri(x)])

corrInfo(filterCorr)
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# -0.953422 -0.007973  0.005600  0.070879  0.053451  1.000000 

corrInfo(filtertransCorr)
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# -0.953422 -0.007973  0.005600  0.070879  0.053451  1.000000 

corrInfo(ssCorr)
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# -0.953519 -0.007345  0.004874  0.074645  0.049637  1.000000 

corrInfo(transCorr)
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# -0.953422 -0.007232  0.004202  0.056391  0.044485  1.000000 

# dfrmNA_num = raw, no filter, no tranformation
# NHL_filter = filter (NZV), no tranformation
# NHL_transform1 = filter (NZV), tranformation (CS)
# NHL_transform2 = no filter, tranformation (CS)***

NHL_highCorr1 = findCorrelation(filtertransCorr, cutoff = .95)
NHL_highCorrN = names(NHL_transform1)[NHL_highCorr1]
length(NHL_highCorr1) #9
NHL_highCorr1
NHL_highCorrN1 = names(NHL_transform1)[NHL_highCorr1]
NHL_highCorrN1
#[1] "defendingTeamAverageTimeOnIceOfDefencemenSinceFaceoff"
#[2] "shootingTeamAverageTimeOnIceOfDefencemen"             
#[3] "shootingTeamMinTimeOnIceOfDefencemen"                 
#[4] "shootingTeamAverageTimeOnIceOfDefencemenSinceFaceoff" 
#[5] "arenaAdjustedShotDistance"                            
#[6] "xCordAdjusted"                                        
#[7] "teamHome"                                             
#[8] "arenaAdjustedXCord"                                   
#[9] "yCord" 

NHL_filterCorr = NHL_transform1[, -NHL_highCorr1]


NHL_df_event = cbind(NHL_filterCorr, event)
NHL_df_event$event = as.factor(NHL_df_event$event)
dim(NHL_df_event) #78582   88

write.csv(NHL_df_event, "NHL_df_event.csv", row.names = FALSE)
