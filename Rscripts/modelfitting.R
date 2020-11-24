packages <- c("randomForest","ranger","xgboost","e1071","caret","glmnet",
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
num.class <- length(levels(df$class))
idx <- 1:nrow(df)   # row indices

## loads the ensemble function
source("ensemble.R")

Result1 <- list()
Result2 <- list()
bestAlg <- list()
confMat <- list()
reps <- 10     # number of replications
set.seed(2021)

for(r in 1:reps){
  repeat{
    ## repeat partitioning of the data into train and test set until all all classes are present in both test and train set
    inTraining <- createDataPartition(df$class,p = 0.9,list = FALSE)
    shuf <- sample(inTraining[,1],replace = FALSE)      # for train data
    shufT <- sample(idx[which(!idx %in% inTraining[,1])], replace = FALSE)  # for test data
    # partitions the dataset
    dat.train <- df[shuf,]
    dat.test <- df[shufT,]
    if(all(table(dat.train$class) >= 3) & all(table(dat.test$class) >= 3)){
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
                            algorithms=c("svm","rang",
                                         "pls_rf", "pca_rf", "rpart", "pls_rpart",
                                         "xgb","pls_xgb","mlp"), 
                            levsChar =as.character(levels(dat.train$class)))
  # the names of the best local classifiers
  bestAlg[[r]] <- ens$bestAlg
  ## predict using the test data
  pred <- predictEns(ens, xTest, yTest, test = dat.test,
                     dlEnsPath = "dl_ens_time.h5", dlIndPath = "dl_ind_time.h5")
  # Saves the results
  Result1[[r]] <- pred$ensemblePerf
  Result2[[r]] <- pred$indPerf
  ## predicted class
  yPred <- pred$yhat
  ## confusion matrix
  confMat[[r]] <- caret::confusionMatrix(yPred,yTest)
  # displays the truth and predictions for each of the "best" algorithms
  dfPred <- data.frame(truth=yTest, ensemble=yPred, pred$pred)
  dfPred <- as.list(dfPred)
  # convert numeric factors to character factors
  dfPred <- lapply(dfPred,function(x) 
    as.character(num2charFac(x,char.levs = 
                as.character(levels(dat.train$class)))))
  names(dfPred) <- c("truth","ensemble",ens$bestAlg)
  dfPred <- as.data.frame(dfPred)
  cat("Predictions for the best individual models for iteration: ",r," of ",reps,"\n ")
  print(dfPred)
  cat("Completed Replication: ",r," of ",reps,"\n ")
}

# save performance results
saveRDS(Result1,"ensClassifPerf.RDS")
saveRDS(Result2,"indClassifPerf.RDS")
saveRDS(bestAlg,"bestAlg.RDS")
saveRDS(confMat,"confMat.RDS")

warnings()
