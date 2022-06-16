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
##### Data Preprocessing #####

library(dplyr)
library(caret)
library(e1071)
library(corrplot)

# Import the dataset
df = read.csv("C:/Users/m29336/Downloads/shots_2018-2020.csv", header=TRUE)
#df=shots_2018.2020
dim(df)        # 300405    124

### Taking care of missing data ###
MissingData <- unlist(lapply(df, function(x) any(is.na(x))))
MissingData <- names(MissingData)[MissingData]
head(MissingData)

byPredByClass <- apply(df[, MissingData], 2,
                       function(x, y) {
                         tab <- table(is.na(x), y)
                         tab[2,]/apply(tab, 2, sum)
                       },
                       y = df$goal)

byPredByClass <- byPredByClass[apply(byPredByClass, 1, sum) > 0,]
byPredByClass <- byPredByClass[, apply(byPredByClass, 2, sum) > 0]
## now print:
t(byPredByClass)

# (delete NA because they are player IDs and are a fairly small percentage)
sum(is.na(df)) # 46
summary(is.na(df)) # goalieIdForShot TRUE:16, shooterPlayerId TRUE:30
dfrmNA = df[complete.cases(df), ]
dim(dfrmNA)    # 300359    124

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

dfrmNA$eventShot = ifelse(dfrmNA$event == 'SHOT', 1, 0)
dfrmNA$eventGoal = ifelse(dfrmNA$event == 'GOAL', 1, 0)
dfrmNA$eventMiss = ifelse(dfrmNA$event == 'MISS', 1, 0)

#remove event, since proxy for goal (class)
dfrmNA$eventShot = NULL
dfrmNA$eventGoal = NULL
dfrmNA$eventMiss = NULL

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

dim(dfrmNA) # 300359  132

dfrmNA_num1 = unlist(lapply(dfrmNA, is.numeric))
dfrmNA_num1 = dfrmNA[, dfrmNA_num1]
dim(dfrmNA_num1) # 300359  114

dfrmNA_num = subset(dfrmNA_num1,select= -goal)
goal = subset(dfrmNA_num1,select="goal")
dim(dfrmNA_num) # 300359  113

##### TEST MODELS WITH AND WITHOUT nearZeroVar#############
# Use the nearZeroVar function to filter out predictors with low frequencies
zeroVarCols = nearZeroVar(dfrmNA_num, names = TRUE)
zeroVarCols

allCols = names(dfrmNA_num)
NHL_filter = dfrmNA_num[, setdiff(allCols, zeroVarCols)]
dim(NHL_filter) # 300359    98


### 1 Address Skew ### FILTERED ###
# Distributions of the remaining predictors (random sampling)
set.seed(123)
sample_NHL_filter <- NHL_filter[, sample(1:ncol(NHL_filter), 8)]
names(sample_NHL_filter)
#[1] "homePenalty1Length"                                  
#[2] "arenaAdjustedXCord"                                  
#[3] "shootingTeamAverageTimeOnIceOfDefencemenSinceFaceoff"
#[4] "yCord"                                              
#[5] "defendingTeamMinTimeOnIceOfDefencemen"              
#[6] "shootingTeamAverageTimeOnIceOfDefencemen"            
#[7] "shootingTeamAverageTimeOnIceOfForwardsSinceFaceoff"  
#[8] "shootingTeamMaxTimeOnIce"

#Look at skew (SAMPLE)
skew_df_sample1 <- apply(sample_NHL_filter, 2, skewness)
summary(skew_df_sample1)
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max.
#0.008171  1.126302  7.192151  8.691822 16.589519 20.184656

### 1 Transform ### FILTERED ###
#center and scale the data, then apply the transformation (SAMPLE)
centerScale_sample1 <- preProcess(sample_NHL_filter, method = c("center", "scale"))
df_transform_sample1 <- predict(centerScale_sample1, newdata = sample_NHL_filter)
df_SS_sample1 <- spatialSign(df_transform_sample1)
#splom(~df_SS_sample1, pch = 16, col = rgb(.2, .2, .2, .4), cex = .7)

#Look at skew full data
skew_df1 <- apply(NHL_filter, 2, skewness)
summary(skew_df1)
#Min.  1st Qu.   Median     Mean  3rd Qu.     Max.
#-7.43810  0.09434  1.08102  4.48498  4.32068 23.08241

#center and scale the data, then apply the transformation
centerScale1 <- preProcess(NHL_filter, method = c("center", "scale"))
NHL_transform1 <- predict(centerScale1, newdata = NHL_filter)
df_SS <- spatialSign(NHL_transform1)

### 2 Address Skew ### UNFILTERED ###
# Distributions of the remaining predictors (random sampling)
set.seed(123)
sample_dfrmNA <- dfrmNA_num[, sample(1:ncol(dfrmNA_num), 8)]
names(sample_dfrmNA)
#[1] "lastEventShotAngle"                              
#[2] "defendingTeamAverageTimeOnIceSinceFaceoff"      
#[3] "shootingTeamAverageTimeOnIceOfForwards"          
#[4] "xCord"                                          
#[5] "shootingTeamMinTimeOnIceOfDefencemenSinceFaceoff"
#[6] "playerNumThatDidLastEvent"                      
#[7] "shootingTeamAverageTimeOnIce"                    
#[8] "lastEventxCord_adjusted"  

#Look at skew (SAMPLE)
skew_df_sample2 <- apply(sample_dfrmNA, 2, skewness)
summary(skew_df_sample2)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
#-0.7741 -0.0334  1.2011  3.8013  3.5597 19.4126


### 2 Transform ### UNFILTERED ###
#center and scale the data, then apply the transformation (SAMPLE)
centerScale_sample2 <- preProcess(sample_dfrmNA, method = c("center", "scale"))
df_transform_sample2 <- predict(centerScale_sample2, newdata = sample_dfrmNA)
df_SS_sample2 <- spatialSign(df_transform_sample2)
#splom(~df_SS_sample2, pch = 16, col = rgb(.2, .2, .2, .4), cex = .7)

#Look at skew full data
skew_df2 <- apply(dfrmNA_num, 2, skewness)
summary(skew_df2)
#Min.  1st Qu.   Median     Mean  3rd Qu.     Max.
#-7.4381   0.1048   1.2135   5.7356   6.9851 122.5349

#center and scale the data, then apply the transformation
centerScale2 <- preProcess(dfrmNA_num, method = c("center", "scale"))
NHL_transform2 <- predict(centerScale2, newdata = dfrmNA_num)
df_SS <- spatialSign(NHL_transform2)


# dfrmNA_num = raw, no filter, no tranformation
# NHL_filter = filter (NZV), no tranformation
# NHL_transform1 = filter (NZV), tranformation (CS)
# NHL_transform2 = no filter, tranformation (CS)

###Correlation Matrices###

filterCorr = cor(NHL_filter)
dim(filterCorr) #98

filtertransCorr = cor(NHL_transform1)
dim(filtertransCorr) #98

transCorr <- cor(NHL_transform2)
dim(transCorr) #113

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
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max.
#-0.952909 -0.007096  0.003963  0.065124  0.046562  1.000000

corrInfo(filtertransCorr)
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max.
#-0.952909 -0.007096  0.003963  0.065124  0.046562  1.000000

corrInfo(ssCorr)
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max.
#-0.952958 -0.006370  0.003553  0.072715  0.045118  1.000000

corrInfo(transCorr)
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max.
#-0.952909 -0.006325  0.002947  0.051854  0.038100  1.000000

NHL_highCorr1 = findCorrelation(filterCorr, cutoff = .95)
NHL_highCorrN = names(NHL_filter)[NHL_highCorr1]
length(NHL_highCorr1) #8
NHL_highCorr1
NHL_highCorrN1 = names(NHL_filter)[NHL_highCorr1]
NHL_highCorrN1
#[1] "defendingTeamMinTimeOnIceSinceFaceoff" "shootingTeamMinTimeOnIceSinceFaceoff"
#[3] "shootingTeamMinTimeOnIceOfDefencemen"  "shotDistance"                        
#[5] "xCordAdjusted"                         "teamHome"                            
#[7] "arenaAdjustedYCord"                    "arenaAdjustedXCord"

NHL_filterCorr = NHL_filter[, -NHL_highCorr1]

###No change
#NHL_highCorr2 = findCorrelation(filterCorr, cutoff = .98)
#NHL_highCorrN = names(NHL_filter)[NHL_highCorr2]
#length(NHL_highCorr2) #5
#NHL_highCorr2
#NHL_highCorrN2 = names(NHL_filter)[NHL_highCorr2]
#NHL_highCorrN2  
#NHL_filterCorr2 = NHL_filter[, -NHL_highCorr1]


#write.csv(NHL_filterCorr, "NHL_filterCorr.csv", row.names = FALSE)  
#write.csv(goal, "goal.csv", row.names = FALSE)

NHL_df = cbind(NHL_filterCorr, goal)
NHL_df$goal = as.factor(NHL_df$goal)
levels(NHL_df$goal) <- c("NoGoal", "Goal")
dim(NHL_df) #300359     91

write.csv(NHL_df, "NHL_df3.csv", row.names = FALSE)
