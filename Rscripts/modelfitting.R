packages <- c("randomForest","xgboost","e1071","caret","glmnet",
              "MASS","neuralnet","pROC","stringr","doParallel","foreach",
              "RankAggreg","class","adabag","rpart","plsgenomics",
              "penalized","measures","pROC","UBL","keras","mlr")
lapply(packages, require, character.only = TRUE)

WD <- "/path/to/abundance/table/and/source/scripts"
setwd(WD)

df <- read.delim("abundanceTable.txt", header = TRUE, sep = "\t", dec = ".")
df$class <- factor(df$class)  # class labels


##################################################################################################
## Begin training Models
## Complete data
##################################################################################################
num.class <- length(levels(df$class))

# row indices
idx <- 1:nrow(df)

## train the classifier
source("ensemble.R")

Result1 <- list()
Result2 <- list()

reps <- 5     # number of replications

for(r in 1:reps){
  
  repeat{
    ## repeat partitioning of the data into train and test set until all all classes are present in both test and train set
    inTraining <- createDataPartition(df$class,p = 0.9,list = FALSE)
    
    shuf <- sample(inTraining[,1],replace = FALSE)      # for train data
    shufT <- sample(idx[which(!idx %in% inTraining[,1])], replace = FALSE)  # for test data
    
    # partitions the dataset
    dat.train <- df[shuf,]
    dat.test <- df[shufT,]
    
    if(length(unique(dat.train$class))==num.class & length(unique(dat.test$class))==num.class){
      break
    }
  }
  
  ## Train set
  y <- dat.train$class
  y <- as.factor(as.numeric(y)-1)   # Factor levels should begin from 0
  x <- data.matrix(dat.train[,!(names(dat.train) %in% c("class"))])
  
  ## Test set
  yTest <- dat.test$class
  yTest <- as.factor(as.numeric(yTest)-1)   # Factor levels should begin from 0
  xTest <- data.matrix(dat.test[,!(names(dat.test) %in% c("class"))])
  
  cat("Started Replication: ",r," of ",reps,"\n ")
  ens <- ensembleClassifier(x, y, M=10, ncomp=30,
                            train = dat.train, test = dat.test,
                            levsChar =as.character(levels(dat.train$class)))

  
  ## prints the best algorithms
  cat("The best algorithms for iteration : ",r," of ",reps,"\n ")
  print(ens$bestAlg)
  
  ## predict using the test data
  pred <- predictEns(ens, xTest, yTest, test = dat.test,
                     dlEnsPath = "dl_ens_time.h5", dlIndPath = "dl_ind_time.h5")
  
  # Prints ensemble performance
  cat("Ensemble Performance for iteration: ",r," of ",reps,"\n ")
  print(pred$ensemblePerf)
  
  # Prints individual classifier performance
  cat("Individual classifier Performance for iteration: ",r," of ",reps,"\n ")
  print(pred$indPerf)  
  
  # Saves the results
  Result1[[r]] <- pred$ensemblePerf
  Result2[[r]] <- pred$indPerf
  
  ## predicted class
  yPred <- pred$yhat
  
  ## confusion matrix
  conf.mat.rf <- confusionMatrix(yTest,yPred)
  
  cat("Confusion matrix for iteration: ",r," of ",reps,"\n ")
  print(conf.mat.rf)
  
  dfPred <- data.frame(truth=yTest,pred$pred)
  
  # convert numeric factors to character factors
  dfPred <- apply(dfPred,2, function(x) num2charFac(x,char.levs = as.character(levels(dat.train$class))))
  
  cat("Predictions for the best individual models for iteration: ",r," of ",reps,"\n ")
  print(dfPred)
  
  cat("Best models for : ",r," of ",reps,"\n ")
  print(ens$models)
  
  cat("Completed Replication: ",r," of ",reps,"\n ")
  
}

cat("Begin Computation of mean accuracy ", "\n ")
mean.acc <- as.numeric()
for(rr in 1:reps){
  A <- Result1[[rr]]
  A <- as.vector(A[,1])
  
  BB <- Result2[[rr]]
  B <- as.vector(BB[,1])
  mean.acc <- c(mean.acc,A,B)
}

cat("All mean accuracy as vector", "\n ")
print(mean.acc)

all.acc <- matrix(mean.acc,nrow = length(B)+1,ncol = rr, byrow = FALSE)
row.names(all.acc) <- c("ensemble",rownames(BB))

cat("prints the accuracies for the models across all replications", "\n")
print(all.acc)

cat("Obtains the mean accuracies for the models across all replications.", "\n")
apply(all.acc, 1, function(x) mean(x, na.rm=TRUE))


warnings()
