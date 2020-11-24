comb <- function(x, ...) {
  lapply(seq_along(x),
         function(i) c(x[[i]], lapply(list(...), function(y) y[[i]])))
}

predict.pls <- function(mod, newdata){
  # get the yhat from the pls regression in mod
  Yhat <- scale(newdata, scale = FALSE, center = mod$meanX) %*% mod$B
  probs <- (Yhat - min(Yhat))/(max(Yhat)-min(Yhat))	
  class <- apply(probs,1,function(x) which.max(x))-1
  list(class = class)
}

accuracy <- function(truth, predicted)
  if(length(truth) > 0)
    sum(truth==predicted)/length(truth) else    
      return(0)

sensitivity <- function(truth, predicted)
  # 1 means positive (present)
  if(sum(truth==1) > 0)
    sum(predicted[truth==1]==1)/sum(truth==1)   else
      return(0)

specificity <- function(truth, predicted)
  if(sum(truth==0) > 0)
    sum(predicted[truth==0]==0)/sum(truth==0)   else
      return(0)

# The multiclass adaptation of the G-mean as defined by Sun et al. 2006
MGmean <- function(truth, predicted){
  if(length(truth) > 0){
    conf_mat <- caret::confusionMatrix(truth,predicted)
    recall <- conf_mat$byClass[, "Recall"]
    names(recall) <- NULL
    MGmean <- (prod(recall,na.rm=TRUE))^(1/(length(recall) - sum(is.na(recall))) )
    return(MGmean)
  } else {
    return(0)
  }
}

convertScores <- function(scores){
  scores <- t(scores)
  ranks <- matrix(0, nrow(scores), ncol(scores))
  weights <- ranks
  for(i in 1:nrow(scores)){
    ms <- sort(scores[i,], decr=TRUE, ind=TRUE)
    ranks[i,] <- colnames(scores)[ms$ix]
    weights[i,] <- ms$x
  }
  list(ranks = ranks, weights = weights)
}

## Function to compute mode
getmode <- function(x){
  uniq <- unique(x)
  uniq[which.max(tabulate(match(x, uniq)))]
}

# Function to return character factors rather than numeric factors
num2charFac <- function(y,char.levs){
  levs1 <- char.levs
  ytrain <- as.character(y)
  ytrain2 <- ytrain
  levs2 <- factor(0:(length(levs1)-1))
  for(i in 1:length(levs1)){
    ind <- which(ytrain == levs2[i])
    ytrain2[ind] <- levs1[i]
  }
  ytrain <- factor(ytrain2,levels = char.levs)
  return(ytrain)
}

# Xgboost parameters
xgb_params <- list(objective = "multi:softprob",
                   eval_metric = "mlogloss",
                   max_depth = 10,
                   subsample = c(1),
                   colsample_bytree = 1,
                   num_class = num.class)