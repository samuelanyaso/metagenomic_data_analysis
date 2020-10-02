multiResultClass <- function(result1=NULL,result2=NULL,result3=NULL,result4=NULL,result5=NULL){
  me <- list(
    result1 = result1,
    result2 = result2,
    result3 = result3,
    result4 = result4,
    result5 = result5
  )
  ## set the name for the class
  class(me) <- append(class(me), "multiclass")
  return(me)
}


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
getmode <- function(xx){
  uniq <- unique(xx)
  uniq[which.max(tabulate(match(xx, uniq)))]
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



