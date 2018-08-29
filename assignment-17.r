#assignment-17

library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(lattice)
library(rattle)

trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
valUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
data <- read.csv(url(trainUrl), header=TRUE, sep=",", na.strings=c("NA","#DIV/0!",""))
validation <- read.csv(url(valUrl), header=TRUE, sep=",", na.strings=c("NA","#DIV/0!",""))

summary(data)
summary(validation)
dim(data)
dim(validation)
# set last (classe) and prior (- classe) column index
last <- as.numeric(ncol(data))
prior <- last - 1

# set variables to numerics for correlation check, except the "classe"
for (i in 1:prior) {
  data[,i] <- as.numeric(data[,i])
  validation[,i] <- as.numeric(validation[,i])
}

# check the correlations
cor.check <- cor(data[, -c(last)])
diag(cor.check) <- 0 
plot( levelplot(cor.check, 
                main ="Correlation matrix for all WLE features in training set",
                scales=list(x=list(rot=90), cex=1.0),))

# find the highly correlated variables
highly.cor <- findCorrelation(cor(data[, -c(last)]), cutoff=0.9)

# remove highly correlated variables
data <- data[, -highly.cor]
validation <- validation[, -highly.cor]

# pre process variables
preObj <-preProcess(data[,1:prior],method=c('knnImpute', 'center', 'scale'))
dataPrep <- predict(preObj, data[,1:prior])
dataPrep$classe <- data$classe

valPrep <-predict(preObj,validation[,1:prior])
valPrep$problem_id <- validation$problem_id

# split dataset into training and test set
inTrain <- createDataPartition(y=dataPrep$classe, p=0.7, list=FALSE )
training <- dataPrep[inTrain,]
testing <- dataPrep[-inTrain,]

# set seed for reproducibility
set.seed(12345)

# get the best mtry
bestmtry <- tuneRF(training[-last],training$classe, ntreeTry=100, 
                   stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE, dobest=FALSE)

mtry <- bestmtry[as.numeric(which.min(bestmtry[,"OOBError"])),"mtry"]

# Model 1: RandomForest
wle.rf <-randomForest(classe~.,data=training, mtry=mtry, ntree=501, 
                      keep.forest=TRUE, proximity=TRUE, 
                      importance=TRUE,test=testing)

# plot the Out of bag error estimates
layout(matrix(c(1,2),nrow=1), width=c(4,1)) 
par(mar=c(5,4,4,0)) #No margin on the right side
plot(wle.rf, log="y", main ="Out-of-bag (OOB) error estimate per Number of Trees")
par(mar=c(5,0,4,2)) #No margin on the left side
plot(c(0,1),type="n", axes=F, xlab="", ylab="")
legend("top", colnames(wle.rf$err.rate),col=1:6,cex=0.8,fill=1:6)