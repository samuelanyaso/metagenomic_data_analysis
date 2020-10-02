packages <- c("randomForest","xgboost","e1071","caret","glmnet",
              "MASS","neuralnet","pROC","stringr","doParallel","foreach",
              "RankAggreg","class","adabag","rpart","plsgenomics",
              "penalized","measures","pROC","UBL","keras","mlr")
lapply(packages, require, character.only = TRUE)


source("validation.R")

##################################################################################################
## Construction of an ensemble of standard classifiers.
##################################################################################################

ensembleClassifier <- function(x, y, M=51, fit.individual=TRUE, varimp=FALSE, rfs=FALSE, nf=ceiling(sqrt(ncol(x))),
                               levsChar, train=NULL, test=NULL,
                               algorithms=c("svm", "rf","rf001","rf002","rf003","rang",
                                            "pls_rf", "pca_rf", "rpart", "pls_rpart",
                                            "adaboost","pls_adaboost","xgb","pls_xgb","mlp"), 
                               validation=c("accuracy", "kappa", "sensitivity"), 
                               ncomp=5, nunits=3, lambda1=5, kernel="radial",num.trees = 1500,
                               distance="Spearman", weighted=TRUE, verbose=TRUE, seed=NULL, ...){

  rownames(x) <- NULL # to suppress the warning message about duplicate rownames
  
  if(!is.null(seed)){
    set.seed(seed)
  }
  
  if(length(algorithms) < 2){
    stop("Ensemble classifier needs at least 2 classification algorithms")
  }
  
  n <- length(y)
  ncla <- length(unique(y))
  nalg <- length(algorithms)
  nvm <- length(validation)
  ly <- levels(y)
  
  if(length(ly) == 23 && all(ly != c("0","1","2","3","4","5","6","7","8","9","10","11","12",
                                     "13","14","15","16","17","18","19","20","21","22"))){
    stop("For multiclass classification, levels in y must be from 0 to 22")
  }
  
  fittedModels <- list() #to keep the fitted algorithms
  varImportance <- NULL
  
  for(k in 1:M){
    
    repeat{
      # obtain bootstrap resamples. To make sure each class is represented at least once
      s <- sample(n,replace = TRUE)
      if(length(table(y[unique(s)])) == ncla & length(table(y[-s])) == ncla)
        break
    }
    
    if(rfs){
      # perform feature selection?
      fs <- sample(1:ncol(x), nf)
    }else{
      fs <- 1:ncol(x)
    }
    
    training <- x[s, fs]
    testing <- x[-unique(s), fs]
    truth <- y[-unique(s)]
    trainY <- y[s]
    trainYChar <- num2charFac(trainY,char.levs = levsChar)
    truthChar <- num2charFac(truth,char.levs = levsChar)
    
    ## For MLP::Tranform labels to One-Hot Encoded labels
    trainLabels <- to_categorical(trainY, num_classes = ncla)
    testLabels <- to_categorical(truth, num_classes = ncla)
    colnames(trainLabels) <- ly
    colnames(testLabels) <- ly

    # construct PLS latent variables (store in plsX) if one of the PLS methods is used
    # this method is taken form library CMA (bioconductor)
    if("pls_lda" %in% algorithms || "pls_xgb" %in% algorithms || "pls_rf" %in% algorithms ||
       "pls_lr" %in% algorithms){
      if(ncomp >= ncol(x))
        stop("Decrease ncomp for the PLS models; must be smaller than ncol(x)")
      fpr <- pls.regression(training, transformy(trainY), ncomp = ncomp,unit.weights = TRUE)
      plsX <- scale(training, scale = FALSE, center = fpr$meanX)%*%fpr$R
      plsTestX <- scale(testing, scale = FALSE, center=fpr$meanX) %*% fpr$R 
    }
    
    # construct PCA latent variables (store in pcaX)
    if("pca_lda" %in% algorithms || "pca_qda" %in% algorithms || "pca_rf" %in% algorithms ||
       "pca_lr" %in% algorithms){
      if(ncomp >= ncol(x))
        stop("Decrease ncomp for the PLS models; must be smaller than ncol(x)")
      pcaX <- prcomp(training)$x[,1:ncomp]
      pcaTestX <- prcomp(testing)$x[,1:ncomp]
    }
    
    # Construct mlr tasks
    dat.train <- data.frame(class=trainYChar, training)
    dat.test <- data.frame(class=truthChar, testing)
    
    #convert characters to factors
    fact_col <- colnames(dat.train)[sapply(dat.train,is.character)]
    
    for(i in fact_col) set(dat.train,j=i,value = factor(dat.train[[i]]))
    for (i in fact_col) set(dat.test,j=i,value = factor(dat.test[[i]]))
    
    #create tasks
    traintask <- makeClassifTask (data = dat.train,target = "class")
    testtask <- makeClassifTask (data = dat.test,target = "class")
    
    #do one hot encoding
    traintask <- createDummyFeatures (obj = traintask)
    testtask <- createDummyFeatures (obj = testtask)
    

    #########################################################################
    ## Creates the xgBoost learner with MLR package
    
    xgblrn <- makeLearner("classif.xgboost",predict.type = "prob")
    xgblrn$par.vals <- list(objective="multi:softprob",
                            verbose = 0, booster = "gbtree",
                            eval_metric="mlogloss", nrounds=100L, eta=0.1,
                            max_depth=6, min_child_weight=4.23,
                            subsample=0.779, colsample_bytree=0.59)
    xgb_lrn_tune <- xgblrn
    #########################################################################
    
    ## Function to fit the multi-layer perceptron model
    MLP <- 
      function(x,training,trainLabels,wghts.nn,resamp.case.wghts,file_name){
        ## Initialize a sequential model
        keras.mod <- keras_model_sequential()  %>% 
          # Add layers to the model
          layer_dense(units = 200, activation = "relu", input_shape = c(ncol(x))) %>% 
          layer_dense(units = 23, activation = "softmax")
        
        # compile the model
        keras.mod %>% compile(loss = 'categorical_crossentropy',
                              optimizer = "adam",
                              metrics = "accuracy")
        
        # fits the model
        keras.mod %>% fit(training,
                          trainLabels,
                          epochs = 20,
                          batch_size = 3,
                          validation_split = 0.1,
                          class_weight = wghts.nn,
                          sample_weight = resamp.case.wghts,
                          verbose=0)
        
        # saves the model
        keras.mod %>% save_model_hdf5(file_name)
        
        return(keras.mod)
      }
    
    
    #############################################################################
    #### Begins parallel computation
    #############################################################################
    
    cores <- detectCores()
    cl <- makeCluster(cores - 1)
    registerDoParallel(cl)
    
    Res <- list()
    oper <- 
      foreach(j = 1:(nalg), .combine='comb', .multicombine=TRUE, .init=list(list(), list()),.export = c("xgb_lrn_tune","traintask","xgb_params","brierSummary"),
              .packages = c("randomForest","ranger","xgboost","e1071","caret","glmnet","MASS","pROC","stringr","RankAggreg","class",
                            "adabag","rpart","plsgenomics","penalized","measures","plyr","recipes","dplyr","keras"))  %dopar% {
                              res <- switch(algorithms[j],
                                            "svm" = suppressWarnings(svm(training, trainY, probability = T, kernel = kernel)),
                                            "rf" = ranger(x=training,y=trainY),
                                            "rf001" = ranger(x=training,y=trainY,num.trees = 1500, mtry = 34),
                                            "rf002" = ranger(x=training,y=trainY,num.trees = 500, mtry = 37),
                                            "rf003" = ranger(x=training,y=trainY,num.trees = 1000, mtry = 41),
                                            "rang" = ranger(x=training,y=trainY,num.trees = 1500, mtry = 46),
                                            "pls_rf" = ranger(x=as.data.frame(plsX), y=trainY),
                                            "pca_rf" = ranger(x=as.data.frame(pcaX), y=trainY),
                                            "rpart" = rpart(y~., data=data.frame(y=trainY,training),method = "class"),
                                            "pls_rpart" = rpart(y~., data=data.frame(y=trainY,plsX),method = "class"), 
                                            "adaboost" = boosting(y ~., data=data.frame(y=trainY,training)),
                                            "pls_adaboost" = boosting(y ~., data=data.frame(y=trainY,plsX)),
                                            "xgb" = mlr::train(learner = xgb_lrn_tune,task = traintask),
                                            "pls_xgb" = xgb.train(params = xgb_params, data = xgb.DMatrix(plsX, label=c(as.numeric(trainY)-1)),
                                                                  nrounds = 50),
                                            "mlp" = MLP(x = x,training = training,trainLabels = trainLabels,
                                                        wghts.nn = NULL,resamp.case.wghts = NULL,file_name="dl_ens_time.h5")
                              )
                              list(res, algorithms[j])
                            }
    stopCluster(cl)
    
    # Calls the saved & fitted MLP model
    keras.mod <- load_model_hdf5("dl_ens_time.h5")
    
    Res <- oper[[1]]
    
    for(j in 1:nalg){
      attr(Res[[j]], "algorithm") <- algorithms[j]
      if("pls_lda" == algorithms[j] || "pls_qda" == algorithms[j] || "pls_rf" == algorithms[j] ||
         "pls_xgb" == algorithms[j] || 
         "pls_adaboost" == algorithms[j] || "pls_rpart" == algorithms[j]){
        attr(Res[[j]], "meanX") <- fpr$meanX
        attr(Res[[j]], "R") <- fpr$R
      }
      if("pca_lda" == algorithms[j] || "pca_qda" == algorithms[j] || 
         "pca_rf" == algorithms[j]){
        attr(Res[[j]], "ncomp") <- ncomp
      }
    }
    
    # predict using fitted models on oob 
    predicted <- list()
    probs <- list()
    for(j in 1:nalg){
      switch(algorithms[j],
             "svm" = {pred <-  predict(Res[[j]], testing, prob=TRUE)
             predicted[[j]] <- pred},
             "rf" = {predicted[[j]] <- predict(Res[[j]], testing)$predictions},
             "rf001" = {predicted[[j]] <-predict(Res[[j]], testing)$predictions},
             "rf002" = {predicted[[j]] <- predict(Res[[j]], testing)$predictions},
             "rf003" = {predicted[[j]] <- predict(Res[[j]], testing)$predictions},
             "rang" = {predicted[[j]] <- predict(Res[[j]], testing)$predictions},
             "pls_rf" = {predicted[[j]] <- predict(Res[[j]], as.data.frame(plsTestX))$predictions},
             "pca_rf" = {predicted[[j]] <- predict(Res[[j]], as.data.frame(pcaTestX))$predictions},
             "rpart" = {predicted[[j]] <- predict(Res[[j]], newdata=data.frame(testing), type="class")},
             "pls_rpart" = {predicted[[j]] <- predict(Res[[j]], newdata=data.frame(plsTestX), type="class")}, 
             "adaboost" = {predicted[[j]] <- as.numeric(predict(Res[[j]], newdata=data.frame(testing))$class)},
             "pls_adaboost" = {predicted[[j]] <- as.numeric(predict(Res[[j]], newdata=data.frame(plsTestX))$class)},
             "xgb" = {xgpred <- predict(Res[[j]],testtask)
             predicted[[j]] <- factor(as.numeric(xgpred$data$response)-1, levels=ly)},
             "pls_xgb" = {pred <- predict(Res[[j]], newdata=xgb.DMatrix(plsTestX))
             temp <- data.frame(t(matrix(pred, nrow = ncla, ncol=length(pred)/ncla)))
             names(temp) <- ly
             predicted[[j]] <- factor(as.numeric(apply(temp, 1, which.max))-1,levels=ly)},
             "mlp" = {predicted[[j]] <- factor(keras.mod %>% predict_classes(testing, batch_size = 128), levels=ly)}
      )} 
    
    # compute validation measures
    scores <- matrix(0, nalg, nvm)
    rownames(scores) <- algorithms
    colnames(scores) <- validation
    
    for(i in 1:nalg){
      for(j in 1:nvm){
        scores[i,j] <- switch(validation[j],
                              "accuracy" = accuracy(truth, factor(predicted[[i]], levels = ly)),
                              "kappa" = KAPPA(truth, factor(predicted[[i]], levels = ly)),
                              "sensitivity" = sensitivity(truth, factor(predicted[[i]], levels = ly)),
                              "auc" = as.numeric(multiclass.roc(as.numeric(truth), 
                                                                as.numeric(factor(predicted[[i]], levels = ly)),quiet=T)$auc)
        )
      }
    }
    
    # perform rank aggregation
    convScores <- convertScores(scores)
    if(nvm > 1 && nalg <= 6){
      if(weighted){
        fittedModels[[k]] <- Res[[which(algorithms == BruteAggreg(convScores$ranks,
                                                                  nalg, convScores$weights, distance=distance)$top.list[1])]]
      }else{
        fittedModels[[k]] <- Res[[which(algorithms == BruteAggreg(convScores$ranks, nalg,
                                                                  distance=distance)$top.list[1])]]
      }
    }else if(nvm > 1 && nalg > 6){
      if(weighted){
        fittedModels[[k]] <- Res[[which(algorithms == RankAggreg(convScores$ranks,
                                                                 nalg, convScores$weights, distance=distance, verbose=FALSE)$top.list[1])]]
      }else{
        fittedModels[[k]] <- Res[[which(algorithms == RankAggreg(convScores$ranks, nalg,
                                                                 distance=distance, verbose=FALSE)$top.list[1])]]
      }
    }else{
      fittedModels[[k]] <- Res[[which.max(scores[,1])]]
    }
    
    ###############################################################################################
    # variable importance as in Random Forest
    if(varimp){
      predicted <- matrix(0, nrow(testing), ncol(testing)+1) # one extra for untouched data
      kalg <- attr(fittedModels[[k]], "algorithm")
      nts <- nrow(testing) # number of testing samples
      
      if(!kalg %in% c("pls_lda", "pls_qda", "pls_rf","pca_lda", "pca_qda", "pca_rf","pls_rpart","pls_adaboost","pls_xgb")){
        testingX <- testing
        testingY <- as.numeric(truth)
        
        for(i in 1:(ncol(testing)+1)){
          if(i != ncol(testing)+1){
            rsample <- sample(nts,nts)
            testing <- testingX
            testing[,i] <- testingX[rsample, i]
          }
          else{
            testing <- testingX
          }
          colnames(testing) <- colnames(testingX)
          
          switch(kalg,
                 "svm"       = predicted[,i] <- as.character(predict(fittedModels[[k]], testing, prob=TRUE)),
                 "rf"        = predicted[,i] <- as.character(predict(fittedModels[[k]], testing)$prediction),
                 "rf001"     = predicted[,i] <- as.character(predict(fittedModels[[k]], testing)$prediction),
                 "rf002"    = predicted[,i] <- as.character(predict(fittedModels[[k]], testing)$prediction),
                 "rf003"    = predicted[,i] <- as.character(predict(fittedModels[[k]], testing)$prediction),
                 "rang"     = predicted[,i] <- as.character(predict(fittedModels[[k]], testing)$prediction),
                 "rpart"     = predicted[,i] <- as.character(predict(fittedModels[[k]], newdata=data.frame(testing), type="class")),
                 "adaboost"  = predicted[,i] <- as.character(as.numeric(predict(fittedModels[[k]], newdata=data.frame(testing))$class)),
                 "xgb" = {xgpred <- predict(fittedModels[[k]],testtask)
                 predicted[,i] <- as.character(factor(as.numeric(xgpred$data$response)-1, levels=ly))},
                 "pls"       = predicted[,i] <- as.character(predict.pls(fittedModels[[k]], newdata=testing)$class),
                 "mlp" = predicted[,i] <- as.character(factor(keras.mod %>% predict_classes(testing, batch_size = 128), levels=ly))
          )
        }
        
        predicted <- apply(predicted,2,as.numeric)
        predicted <- predicted 
        
        untouchedAccuracy <- sum(predicted[,ncol(predicted)] == testingY)/length(testingY)
        tempImp <- rep(0, ncol(testing))
        for(i in 1:ncol(testing))
          tempImp[i] <- untouchedAccuracy - sum(predicted[,i] == testingY)/length(testingY)
        varImportance <- rbind(varImportance, tempImp)
      }
    }
    ##############################################################################################################

    # some output
    if(verbose)
      cat("Iter ", k, "\n")
  } # ends loop 1:M
  
  # how many times each algorithm was the best
  bestAlg <- unlist(sapply(fittedModels, FUN = function(x) attr(x, "algorithm")))
  rawImportance <- varImportance
  if(!is.null(varImportance)){
    varImportance <- matrix(c(apply(varImportance, 2, mean), apply(varImportance, 2, mean)/
                                (apply(varImportance, 2, sd)/sqrt(nrow(varImportance)))), ncol=2)
    rownames(varImportance) <- colnames(x)
    colnames(varImportance) <- c("MeanDecreaseAcc", "StdMeanDecreaseAcc")
  }
  
  
  
  ###########################################################################
  # train all classifiers individually on all training data
  ###########################################################################
  
  training <- x #train using all data
  trainY <- y
  trainYChar <- num2charFac(y,char.levs = levsChar)
  
  ## Tranform labels to One-Hot Encoded labels
  trainLabels <- to_categorical(trainY, num_classes = ncla)
  colnames(trainLabels) <- ly
  
  ## MLR setup
  dat.train <- train
  dat.test <- test
  
  #convert characters to factors
  fact_col <- colnames(dat.train)[sapply(dat.train,is.character)]
  
  for(i in fact_col) set(dat.train,j=i,value = factor(dat.train[[i]]))
  for (i in fact_col) set(dat.test,j=i,value = factor(dat.test[[i]]))
  
  #create tasks
  traintask <- makeClassifTask (data = dat.train,target = "class")
  testtask <- makeClassifTask (data = dat.test,target = "class")
  
  #do one hot encoding 
  traintask <- createDummyFeatures (obj = traintask)
  testtask <- createDummyFeatures (obj = testtask)
  
  
  if(fit.individual){
    # construct PLS latent variables (store in plsX) if one of the PLS methods used
    # this method is taken from library CMA (bioconductor)
    if("pls_lda" %in% algorithms || "pls_qda" %in% algorithms || "pls_rf" %in% algorithms ||
       "pls_adaboost" %in% algorithms || "pls_xgb" %in% algorithms){
      if(ncomp >= ncol(x))
        stop("Decrease ncomp for the PLS models; must be smaller than ncol(x)")
      fpr <- pls.regression(x, transformy(y), ncomp=ncomp)
      plsX <- scale(x, scale=FALSE, center=fpr$meanX)%*%fpr$R
    }
    
    # construct PCA latent variables (store in pcaX)
    if("pca_lda" %in% algorithms || "pca_qda" %in% algorithms || "pca_rf" %in% algorithms){
      if(ncomp >= ncol(x))
        stop("Decrease ncomp for the PLS models; must be smaller than ncol(x)")
      pcaX <- prcomp(x)$x[,1:ncomp]
    }  
    
    #############################################################################
    #### Begins parallel computation
    #############################################################################
    Res <- list()
    
    cores <- detectCores()
    cl <- makeCluster(cores - 1)
    registerDoParallel(cl)
    
    temp <- 
      foreach(j = 1:nalg, .combine='comb', .multicombine=TRUE, .init=list(list(), list()),.export = c("xgb_lrn_tune","traintask","xgb_params","brierSummary"),
              .packages = c("randomForest","ranger","xgboost","e1071","caret","glmnet","MASS","pROC","stringr","RankAggreg","class",
                            "adabag","rpart","plsgenomics","penalized","measures","plyr","recipes","dplyr","keras"))  %dopar% {
                              res <- switch(algorithms[j],
                                            "svm" = suppressWarnings(svm(training, trainY, probability = T, kernel = kernel)),
                                            "rf" = ranger(x=training,y=trainY),
                                            "rf001" = ranger(x=training,y=trainY,num.trees = 1500, mtry = 34),
                                            "rf002" = ranger(x=training,y=trainY,num.trees = 500, mtry = 37),
                                            "rf003" = ranger(x=training,y=trainY,num.trees = 1000, mtry = 41),
                                            "rang" = ranger(x=training,y=trainY,num.trees = 1500, mtry = 46),
                                            "pls_rf" = ranger(x=as.data.frame(plsX), y=trainY),
                                            "pca_rf" = ranger(x=as.data.frame(pcaX), y=trainY),
                                            "rpart" = rpart(y~., data=data.frame(y=trainY,training),method = "class"),
                                            "pls_rpart" = rpart(y~., data=data.frame(y=trainY,plsX),method = "class"), 
                                            "adaboost" = boosting(y ~., data=data.frame(y=trainY,training)),
                                            "pls_adaboost" = boosting(y ~., data=data.frame(y=trainY,plsX)),
                                            "xgb" = mlr::train(learner = xgb_lrn_tune,task = traintask),
                                            "pls_xgb" = xgb.train(params = xgb_params, data = xgb.DMatrix(plsX, label=c(as.numeric(trainY)-1)),
                                                                  nrounds = 50),
                                            "mlp" = MLP(x = x,training = training,trainLabels = trainLabels,
                                                        wghts.nn = NULL,resamp.case.wghts = NULL,file_name="dl_ind_time.h5")
                              )
                              list(res, algorithms[j])
                            }
    stopCluster(cl)
    
    ## Stores the result
    Res <- temp[[1]]
    
    for(j in 1:nalg){
      attr(Res[[j]], "algorithm") <- algorithms[j]
      if("pls_lda" == algorithms[j] || "pls_qda" == algorithms[j] || "pls_rf" == algorithms[j] ||
         "pls_xgb" == algorithms[j] || 
         "pls_adaboost" == algorithms[j] || "pls_rpart" == algorithms[j]){
        attr(Res[[j]], "meanX") <- fpr$meanX
        attr(Res[[j]], "R") <- fpr$R
      }
      if("pca_lda" == algorithms[j] || "pca_qda" == algorithms[j] || 
         "pca_rf" == algorithms[j]){
        attr(Res[[j]], "ncomp") <- ncomp
      }
    }
  }
  
  newFittedModels <- list()
  for(i in 1:M){
    newFittedModels[[i]] <- switch(bestAlg[i],
                                   "svm" = Res$svm,
                                   "rf" = Res$rf,
                                   "rf001" = Res$rf001,
                                   "rf002" = Res$rf002,
                                   "rf003" = Res$rf003,
                                   "rang" = Res$rang,
                                   "pls_rf" = Res$pls_rf,
                                   "pca_rf" = Res$pca_rf,
                                   "rpart" = Res$rpart,
                                   "pls_rpart" = Res$pls_rpart,
                                   "adaboost" = Res$adaboost,
                                   "pls_adaboost" = Res$pls_adaboost,
                                   "xgb" = Res$xgb,
                                   "pls_xgb" =  Res$pls_xgb,
                                   "mlp" = Res$mlp
    )
  }
  
  for(i in 1:M){
    attr(newFittedModels[[i]], "algorithm") <- bestAlg[i]
    if("pls_lda" == bestAlg[i] || "pls_qda" == bestAlg[i] || "pls_rf" == bestAlg[i] ||
       "pls_xgb" == bestAlg[i] || "pls_adaboost" == bestAlg[i] ||
       "pls_rpart" == bestAlg[i]){
      attr(newFittedModels[[i]], "meanX") <- fpr$meanX
      attr(newFittedModels[[i]], "R") <- fpr$R
    }
    if("pca_lda" == bestAlg[i] || "pca_qda" == bestAlg[i] ||
       "pca_rf" == bestAlg[i])
      attr(newFittedModels[[i]], "ncomp") <- ncomp
  }
  
  res <- list(models = newFittedModels, indModels = Res, rawImportance = rawImportance, M = M,
              bestAlg = bestAlg, levels = ly, importance = varImportance, convScores = convScores)
  class(res) <- "ensemble"
  res
}



##################################################################################################
## Predict test test and evaluate model performance
##################################################################################################

predictEns <- function(EnsObject, newdata, y=NULL, test=NULL, dlEnsPath, dlIndPath, plot=TRUE){
  ly <- EnsObject$levels
  M <- EnsObject$M
  n <- nrow(newdata)
  predicted <- matrix(0, n, M)
  
  keras.ens <- load_model_hdf5(dlEnsPath)
  keras.ind <- load_model_hdf5(dlIndPath)
  
  for(i in 1:M){
    # construct components for MLR setup
    testing <- newdata
    
    if(!is.null(y)){
      dat.test <- test
    } else {
      dat.test <- data.frame(class=factor(rep("MYS",nrow(testing))),testing)
    }
    
    #convert characters to factors
    fact_col <- colnames(dat.test)[sapply(dat.test,is.character)]
    
    for (iii in fact_col) set(dat.test,j=iii,value = factor(dat.test[[iii]]))
    
    #create tasks
    testtask <- makeClassifTask (data = dat.test,target = "class")
    
    #do one hot encoding`<br/> 
    testtask <- createDummyFeatures (obj = testtask)
    
    # construct components for PLS and PCA
    
    if(attr(EnsObject$models[[i]], "algorithm") %in% c("pls_lda", "pls_qda", "pls_rf", "pls_rpart","pls_adaboost","pls_xgb")){
      R <- attr(EnsObject$models[[i]], "R")
      meanX <- attr(EnsObject$models[[i]], "meanX")
      plsTestX <- scale(testing, scale=FALSE, center=meanX)%*%R
    }
    if(attr(EnsObject$models[[i]], "algorithm") %in% c("pca_lda", "pca_qda", "pca_rf")){
      pcaTestX <- prcomp(testing)$x[,1:as.numeric(attr(EnsObject$models[[i]], "ncomp"))]
    }
    
    switch(attr(EnsObject$models[[i]], "algorithm"),
           "svm" = predicted[,i] <-  as.character(predict(EnsObject$models[[i]], testing, prob=TRUE)),
           "rf" = predicted[,i] <- as.character(predict(EnsObject$models[[i]], testing)$prediction),
           "rf001" = predicted[,i] <- as.character(predict(EnsObject$models[[i]], testing)$prediction),
           "rf002" = predicted[,i] <- as.character(predict(EnsObject$models[[i]], testing)$prediction),
           "rf003" = predicted[,i] <- as.character(predict(EnsObject$models[[i]], testing)$prediction),
           "rang"     = predicted[,i] <- as.character(predict(EnsObject$models[[i]], testing)$prediction),
           "pls_rf" = predicted[,i] <- as.character(predict(EnsObject$models[[i]], as.data.frame(plsTestX))$prediction),
           "pca_rf" = predicted[,i] <- as.character(predict(EnsObject$models[[i]], as.data.frame(pcaTestX))$prediction),
           "rpart" = predicted[,i] <- as.character(predict(EnsObject$models[[i]], newdata=data.frame(testing), type="class")),
           "pls_rpart" = predicted[,i] <- as.character(predict(EnsObject$models[[i]], newdata=data.frame(plsTestX), type="class")), 
           "adaboost" = predicted[,i] <- as.numeric(as.character(predict(EnsObject$models[[i]], newdata=data.frame(testing))$class)),
           "pls_adaboost" = predicted[,i] <- as.numeric(as.character(predict(EnsObject$models[[i]], newdata=data.frame(plsTestX))$class)),
           "xgb" = {xgpred <- predict(EnsObject$models[[i]],testtask)
           predicted[,i] <- as.character(factor(as.numeric(xgpred$data$response)-1, levels=ly))},
           "pls_xgb" = {pred <- predict(EnsObject$models[[i]], newdata=xgb.DMatrix(plsTestX))
           temp <- data.frame(t(matrix(pred, nrow = ncla, ncol=length(pred)/ncla)))
           names(temp) <- ly
           predicted[,i] <- as.character(factor(as.numeric(apply(temp, 1, which.max))-1,levels=ly))},
           "mlp" = predicted[,i] <- as.character(factor(keras.ens %>% predict_classes(testing, batch_size = 128), levels=ly))
    )
  }
  
  predicted <- apply(predicted, 2, as.numeric)
  
  # class probability by majority
  newclass <- factor(apply(predicted, 1, function(x) getmode(x)), levels = EnsObject$levels)
  
  res <- list()
  if(!is.null(y)){
    valM <- c("accuracy","sensitivity","kappa", "auc")
    nvalM <- length(valM)
    acc <- accuracy(y, newclass)
    sens <- sensitivity(y, newclass)
    kappa = KAPPA(y, newclass)
    auc <-  as.numeric(multiclass.roc(as.numeric(y), 
                                      as.numeric(newclass),quiet=T)$auc)
    ensemblePerformance <- matrix(c(acc,sens,kappa,auc),1,nvalM)
    colnames(ensemblePerformance) <- valM
    rownames(ensemblePerformance) <- "ensemble"
  }
  
  
  #############################################################################
  # predict using individual models
  #############################################################################
  if(length(EnsObject$indModels) > 0){
    indPred <- matrix(0, nrow(newdata), length(EnsObject$indModels))
    indProb <- list()
    testing <- newdata
    
    for(i in 1:length(EnsObject$indModels)){
      # constract components for PLS and PCA
      if(attr(EnsObject$indModels[[i]], "algorithm") %in% c("pls_lda","pls_qda","pls_rf","pls_rpart","pls_adaboost","pls_xgb")){
        R <- attr(EnsObject$indModels[[i]], "R")
        meanX <- attr(EnsObject$indModels[[i]], "meanX")
        plsTestX <- scale(testing, scale=FALSE, center=meanX)%*%R
      }
      if(attr(EnsObject$indModels[[i]], "algorithm") %in% c("pca_lda", "pca_qda", "pca_rf", "pca_lr")){
        pcaTestX <- prcomp(testing)$x[,1:as.numeric(attr(EnsObject$indModels[[i]], "ncomp"))]
      }
      
      switch(attr(EnsObject$indModels[[i]], "algorithm"),
             "svm" = {pred <-  predict(EnsObject$indModels[[i]], testing, prob=TRUE)
             indPred[,i] <- as.character(pred)},
             "rf" = {indPred[,i] <- as.character(predict(EnsObject$indModels[[i]], testing)$prediction)},
             "rf001" = {indPred[,i] <- as.character(predict(EnsObject$indModels[[i]], testing)$prediction)},
             "rf002" = {indPred[,i] <- as.character(predict(EnsObject$indModels[[i]], testing)$prediction)},
             "rf003" = {indPred[,i] <- as.character(predict(EnsObject$indModels[[i]], testing)$prediction)},
             "rang"     = {indPred[,i] <- as.character(predict(EnsObject$indModels[[i]], testing)$prediction)},
             "pls_rf" = {indPred[,i] <- as.character(predict(EnsObject$indModels[[i]], as.data.frame(plsTestX))$prediction)},
             "pca_rf" = {indPred[,i] <- as.character(predict(EnsObject$indModels[[i]], as.data.frame(pcaTestX))$prediction)},
             "rpart" = {indPred[,i] <- as.character(predict(EnsObject$indModels[[i]], newdata=data.frame(testing), type="class"))},
             "pls_rpart" = {indPred[,i] <- as.character(predict(EnsObject$indModels[[i]], newdata=data.frame(plsTestX), type="class"))}, 
             "adaboost" = {indPred[,i] <- as.numeric(as.character(predict(EnsObject$indModels[[i]], newdata=data.frame(testing))$class))},
             "pls_adaboost" = {indPred[,i] <- as.numeric(as.character(predict(EnsObject$indModels[[i]], newdata=data.frame(plsTestX))$class))},
             "xgb" = {xgpred <- predict(EnsObject$indModels[[i]],testtask)
             indPred[,i] <- as.character(factor(as.numeric(xgpred$data$response)-1, levels=ly))},
             "pls_xgb" = {pred <- predict(EnsObject$indModels[[i]], newdata=xgb.DMatrix(plsTestX))
             temp <- data.frame(t(matrix(pred, nrow = ncla, ncol=length(pred)/ncla)))
             names(temp) <- ly
             #indProb[[i]] <- temp
             indPred[,i] <- as.character(factor(as.numeric(apply(temp, 1, which.max))-1,levels=ly))},
             "mlp" = {indPred[,i] <- as.character(factor(keras.ind %>% predict_classes(testing, batch_size = 128), levels=ly))}
      )
    }
    
    indPred <- apply(indPred,2,as.numeric)
    
    valM <- c("accuracy","sensitivity","kappa","auc")
    nvalM <- length(valM)
    
    if(!is.null(y)){
      indPerformance <- matrix(0, length(EnsObject$indModels), length(valM))
      rownames(indPerformance) <- unlist(sapply(EnsObject$indModels, FUN = function(x) attr(x, "algorithm")))
      colnames(indPerformance) <- valM
      
      truth <- y
      for(i in 1:length(EnsObject$indModels))
        for(j in 1:nvalM)
          indPerformance [i,j] <- switch(valM[j],
                                         "accuracy" = accuracy(truth, factor(indPred[,i], levels=EnsObject$levels)),
                                         "sensitivity" = sensitivity(truth, factor(indPred[,i], levels=EnsObject$levels)),
                                         "kappa" = KAPPA(truth, factor(indPred[,i], levels=EnsObject$levels)),
                                         "auc" = as.numeric(multiclass.roc(as.numeric(truth), 
                                                                           as.numeric(factor(indPred[,i], levels = EnsObject$levels)),quiet=T)$auc)
          ) 
    }
  }
  
  if(is.null(y)){
    res <- list(yhat=newclass, pred=predicted)
  }else{
    if(length(EnsObject$indModels) > 0){
      res <- list(yhat=newclass, pred=predicted, ensemblePerf=ensemblePerformance,
                  indPerf=indPerformance)
    }else{
      res <- list(yhat=newclass, pred=predicted, ensemblePerf=ensemblePerformance)
    }
  }
  class(res) <- "predictEnsemble"
  res
}





