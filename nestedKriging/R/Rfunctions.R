

.validCovTypes <- function() {
return(c("rational1", "rational2", "exp", "exp.approx", "gauss", "gauss.approx", "matern5_2", "matern3_2", "powexp", "white_noise"))
}

.checkModel <- function(X, Y, clusters, krigingType, nugget) {
  if (!is.matrix(X)) stop("'X' must be a matrix")
  if ((nrow(X)<1)||(ncol(X)<1)) stop("'X' must have at least one raw and one column")
  if (!is.numeric(X)) stop("'X' must contain numeric values")
  if (!all(is.finite(X))) stop("'X' must contain finite values")
  
  if(!is.numeric(Y)) stop("'Y' must be a vector of numeric values")
  if (length(Y)<1) stop("'Y' must contain at least one value")
  if (!all(is.finite(Y))) stop("'Y' must contain finite values")
  if(!(length(Y)==nrow(X))) stop("'Y' must have the same length as the number of rows in 'X'")
  
  if (!is.numeric(clusters)) stop("'clusters' must be a vector of integer values")
  if (length(clusters)<1) stop("'clusters' must contain at least one value")
  if (!all(is.finite(clusters))) stop("'clusters' must contain finite values")
  if (!isTRUE(all.equal(clusters, as.integer(clusters)))) stop("'clusters' must contain integer values")
  if (!all(clusters>=0)) stop("'clusters' must contain nonnegative integer values")
  if (!(length(clusters)==nrow(X))) stop("'clusters' must have the same length as the number of rows in X")
  clusters <- as.integer(round(clusters,0))

  validKrigingType = c("simple", "ordinary", "OKSK", "SKOK", "SKSK", "OKOK", "universal", "UKOK", "UKSK")
  if(class(krigingType)!="character") stop("'krigingType' must be one of the following:", paste(validKrigingType, collapse=", ") )
  if(!(krigingType) %in% validKrigingType) stop("'krigingType' must be one of the following:", paste(validKrigingType, collapse=", ") )

  if(!is.numeric(nugget)) stop("'nugget' must be a vector of numeric values")
  if (!all(is.finite(nugget))) stop("'nugget' must contain finite values")
  if (!all(nugget>=0)) stop("'nugget' must contain nonnegative real values")
}


.checkCovariances <- function(covType, param, sd2, expectedDimension) {
  validCovType = .validCovTypes()
  if(class(covType)!="character") stop("'covType' must be one of the following:", paste(validCovType, collapse=", ") )
  if(!(covType) %in% validCovType) stop("'covType' must be one of the following:", paste(validCovType, collapse=", ") )
  
  if(!is.numeric(param)) stop("'param' must be a vector of numeric values")
  if (!all(is.finite(param))) stop("'param' must contain finite values")
  if (length(param)<1) stop("'param' must contain at least one value")
  if(covType=="powexp") {
    if(!(length(param)==(2 * expectedDimension))) stop(paste("with covType='powexp', param must have the length ", 2 * expectedDimension, " = 2 * number of columns in X. param=(lengthscales, powers)"))
  } else { 
    if(!(length(param)==expectedDimension)) stop(paste("'param' must have the length ", expectedDimension, " = number of columns in X"))
  }
  if(!is.numeric(sd2)) stop("'sd2' must be a numeric value")
  if(sd2<0.0) stop("'sd2' must be a positive value")
}

.checkEnvironment <- function(tagAlgo, numThreads, numThreadsZones, verboseLevel, outputLevel, numThreadsBLAS, globalOptions) {
  if (!(is.character(tagAlgo))) stop("'tagAlgo' must be a character string")
  
  integerList=list(numThreadsZones, numThreads, numThreadsBLAS, verboseLevel, outputLevel)
  integerListStr=list("'numThreadsZones'", "'numThreads'", "'numThreadsBLAS'", "'verboseLevel'", "'outputLevel'")
  for(i in 1:length(integerList)) {
    if (!(is.numeric(integerList[[i]]))) stop(integerListStr[[i]], " must be an integer numeric value")
    if (!(length(integerList[[i]])==1)) stop(integerListStr[[i]], " must be ONE integer")
    if (abs(integerList[[i]]-as.integer(integerList[[i]]))>1e-5) stop(integerListStr[[i]], " must be an integer")
  }
  if (numThreadsZones<1) stop("'numThreadsZones' must be at least 1")
  if (numThreads<1) stop("'numThreads' must be at least 1")
  if (numThreadsBLAS<1) stop("'numThreadsBLAS' must be at least 1")
  
  if(!is.numeric(globalOptions)) stop("'globalOptions' must be a vector of numeric values")
  if (length(globalOptions)<1) stop("'globalOptions' must contain at least one value")
  if (!all(is.finite(globalOptions))) stop("'globalOptions' must contain finite values")
}

.checkPredPoints <- function(x, expectedColumns) {
  if (!is.matrix(x)) stop("'x' must be a matrix")
  if ((nrow(x)<1)||(ncol(x)<1)) stop("'x' must have at least one raw and one column")
  if (!is.numeric(x)) stop("'x' must contain numeric values")
  if (!all(is.finite(x))) stop("'x' must contain finite values")
  if (expectedColumns!=ncol(x)) stop("error: 'X' and 'x' must have the same number of columns")
}

setNumThreadsBLAS <-function(numThreadsBLAS=1, showMessage=TRUE) {
  RhpcBLASctl::blas_set_num_threads(numThreadsBLAS)
  if (showMessage) {  message("linear algebra libray BLAS threads set to ", numThreadsBLAS) }
}

#################

nestedKriging <- function(X, Y, clusters, x, covType, param, sd2, krigingType="simple", tagAlgo = "", numThreadsZones = 1L, numThreads = 16L, verboseLevel = 10L, outputLevel = 0L, numThreadsBLAS = 1L, globalOptions= as.integer( c(0)), nugget = c(0.0), trendX=NULL, trendx=NULL) {

  ################################################### basic check of input arguments validity
  .checkModel(X, Y, clusters, krigingType, nugget)
  dimension <- ncol(X)
  .checkPredPoints(x, dimension)
  .checkCovariances(covType, param, sd2, dimension)
  .checkEnvironment(tagAlgo, numThreads, numThreadsZones, verboseLevel, outputLevel, numThreadsBLAS, globalOptions)
  
  ################################################### (long 32 bits) integers

  clusters <- as.integer(round(clusters,0))
  numThreadsZones <- as.integer(round(numThreadsZones,0))
  numThreads <- as.integer(round(numThreads,0))
  numThreadsBLAS <- as.integer(round(numThreadsBLAS,0))
  verboseLevel <- as.integer(round(verboseLevel,0))
  outputLevel <- as.integer(round(outputLevel,0))

  ################################################### Set BLAS Threads and launch algo

  setNumThreadsBLAS(numThreadsBLAS, FALSE)
  .Call(`_nestedKriging_nestedKrigingDirect`, X, Y, clusters, x, covType, param, sd2, krigingType, tagAlgo, numThreadsZones, numThreads, verboseLevel, outputLevel, globalOptions, nugget, trendX, trendx)

}

############################################################# Utility OutputLevel

outputLevel <- function(nestedKrigingPredictions=FALSE, alternatives=FALSE, predictionBySubmodel=FALSE, covariancesBySubmodel=FALSE, covariances=FALSE) {
    value <- 0;
    if (alternatives) value <- value - 1;
    if (nestedKrigingPredictions) value <- value - 2;
    if (predictionBySubmodel) value <- value - 4;
    if (covariancesBySubmodel) value <- value - 8;
    if (covariances) value <- value - 16; 
    return (value);
  }

############################################################# HYPER PARAMETERS ESTIMATION


# This code is directly adapted from the nice and commented code of Clement Chevalier
# here adapted using nestedKriging multithreaded algo and possible alternatives methods

estimParamOld <- function(X,Y,cluster,covtype,q, sd2,krigingType,
                       niter,
                       linit,linf,lsup,
                       alpha=0.602,gamma=0.101,a=200,A=1,c=0.1,
                       periodmessage=1000,
                       nugget=0.0, method="NK", numThreads=8, seed=0)  {

  #possible defaultValues
  # alpha <- 0.602       # see, book bathnagar et al
  # gamma <- 0.101       # see, book bathnagar et al
  # a <- 200             # see, book bathnagar et al
  # A <- 1               # see, book bathnagar et al
  # c <- 0.1             # see, book bathnagar et al

  # X: n*d matrix of input points
  # Y: n-dimensional vector of scalar output
  # cluster: n-dimensional vector containing the group number of all n points, refers to N distinct groups
  #          the total number of groups N is fixed throughout the gradient descent
  # q: total number of points where LOO predictions are computed (fixed as a parameter of the gradient descent).
  # covtype: name the covariance name "gauss", "exp", "matern3_2", etc.
  # sd2: variance
  # OrdinaryKriging: 1 if ordinary kriging is used. Otherwise, Simple Kriging is used.

  # niter: number of iterations of the algorithm
  # linit: d-dimensional vector. Initial vector of correlation lengths (starting point of the gradient descent)
  # linf: d-dimensional vector. Minimum value for the correlation lengths (algorithms will refuse a move that goes below these)
  # lsup: d-dimensional vector. same with maximum values
  # alpha, gamma, a, A, c: scalar parameters of the gradient descent: cf book Bhatnagar et al. chapter 5
  # periodmessage: The gradient descent prints it status every periodmessage iterations
  # method: "NK" if LOO computed with nestedKriging, or "BCM", "RBCM", "POE", "GPOE", "SPV" faster alternatives for a first rough estimation


  # RETURN a list with the following fields:
  #
  # mvl: niter*d matrix of all the investigated correlation lengths
  # mLOOMSE: niter-dimensional vector of all the (stochastically) evaluated LOO-MSE
  # vhatl: d-dimensional vector of the correlation lengths at the end of the descent


  # Deduces needed quantities from the input arguments
  set.seed(seed)
  n <- nrow(X)
  d <- ncol(X)

  # compute uniquely required quantities, depending on the chosen method
  if (method=="NK") outputLevel <- 0  else outputLevel <- -1

  # Initializes the algorithm with the starting value
  lcurrent <- linit

  # Initializes the matrix which will contain the parameter estimation at each iteration
  mvl <- matrix(0,nrow=niter,ncol=d)
  mLOOMSE <- rep(0,times=niter)

  for (i in 1:niter) {

    if (i/periodmessage == floor(i/periodmessage) ) {
      # message displaying progression
      cat("##################################################################### \n")
      cat("iteration ",i,"\n")
      cat("current parameter vector: ",lcurrent,"\n")
      cat("System time", as.character(Sys.time()),"\n")
      cat("##################################################################### \n")
    }

    # See, book Bhatnagar et al. chapter 5
    ai <- a/((A+i+1)^alpha)
    deltai <- c/((i+1)^gamma)
    Deltai <- 2*rbinom(n=d,size=1,prob=1/2) - 1

    # The LOO error is NOT computed for all n points, but, instead, for q points chosen randomly among n
    indices <- rep(0,times=n)
    indices[ sample(x=1:n,size=q,replace=FALSE) ] <- 1

    # computation of the LOO errors and extracts the LOO-MSE
    lplus <- exp( log(lcurrent) + deltai*Deltai)

    resplus <- nestedKriging::looErrorsDirect(X=X, Y=Y, clusters=cluster, indices=indices , covType=covtype, param=lplus, sd2=sd2,
                                        krigingType=krigingType, tagAlgo='',
                                        numThreads=numThreads, verboseLevel=-1, outputLevel=outputLevel, globalOptions = c(0), nugget=nugget, method = method)
    LOOMSEplus <- resplus$looError
    #LOOMSEplus <- resplus$LOOErrors$looErrorDefaultMethod

    # computation of the LOO errors and extracts the LOO-MSE
    lminus <- exp(log(lcurrent) - deltai*Deltai)

    resminus <- nestedKriging::looErrorsDirect(X=X, Y=Y, clusters=cluster, indices=indices , covType=covtype, param=lminus, sd2=sd2,
                                         krigingType=krigingType, tagAlgo='nestedK_param',
                                         numThreads=numThreads, verboseLevel=-1, outputLevel=outputLevel, globalOptions = c(0), nugget=nugget, method = method)
    LOOMSEminus <- resminus$looError
    # LOOMSEminus <- resminus$LOOErrors$looErrorDefaultMethod

    # we obtain a proposal ...
    lproposal <- exp( log(lcurrent) - ai*( LOOMSEplus - LOOMSEminus )/( 2*deltai*Deltai )   )

    # ... which is accepted if within the bounds
    if ( sum(lproposal > linf) == d & sum(lproposal < lsup) == d ) lcurrent <- lproposal

    # we keep track of the current proposal and its LOO-MSE
    mvl[i,] <- lcurrent
    mLOOMSE[i] <- 0.5*(LOOMSEplus + LOOMSEminus)
  }

  # The final estimate is the last proposal
  vhatl <- mvl[niter,]

  # And we return it, together with the previous proposals.
  return( list(mvl=mvl,mLOOMSE=mLOOMSE,vhatl=vhatl) )
}


estimSigma2 <- function(X, Y, q, clusters, covType, krigingType, param, numThreads=8, nugget=0.0, method="NK", seed=0, trendX=NULL, trendx=NULL)  {
  # X: n*d matrix of input points
  # Y: n-dimensional vector of scalar output
  # clusters: n-dimensional vector containing the group number of all n points
  # covType: the covariance name "gauss", "exp", "matern3_2", etc.
  # krigingType: "ordinary" if ordinary kriging is used. Otherwise, "simple" and Simple Kriging is used.
  # param: d-dimensional vector of correlation lengths

  #RETURN estimate of the variance


  # Deduces needed quantities from the input arguments
  n <- nrow(X)
  d <- ncol(X)
  set.seed(seed)
  
  .checkModel(X, Y, clusters, krigingType, nugget)
  .checkCovariances(covType, param, 1.0, d)
  #.checkEnvironment(tagAlgo, numThreads, numThreadsZones, verboseLevel, outputLevel, numThreadsBLAS, globalOptions)

  # compute uniquely required quantities, depending on the chosen method
  if (method=="NK") outputLevel <- 0  else outputLevel <- -1

  chosenIndices <- sort(sample(x=1:n,size=q,replace=FALSE))
  indices <- rep(0,times=n)
  indices[ chosenIndices ] <- 1

  res <- nestedKriging::looErrors(X=X, Y=Y, clusters=clusters, indices=indices , covType=covType, param=param, sd2=1.0,
                                  krigingType=krigingType, tagAlgo='nestedK_sd2',
                                  numThreads=numThreads, verboseLevel=0, outputLevel=outputLevel, globalOptions = c(0), nugget=nugget, method = method, trendX=trendX, trendx=trendx)
  return( mean( ((Y[chosenIndices]-res$LOOErrors$meanDefaultMethod)^2) / res$LOOErrors$sd2DefaultMethod ) )
}

#looErrors <- function(X, Y, clusters, indices, covType, param, sd2, krigingType = "simple", tagAlgo = "", numThreadsZones = 1L, numThreads = 16L, verboseLevel = 10L, outputLevel = 1L, globalOptions = as.integer( c(0)), nugget = as.numeric( c(0)), method = "NK") {
#  .Call(`_nestedKriging_looErr#ors`, X, Y, clusters, indices, covType, param, sd2, krigingType, tagAlgo, numThreadsZones, numThreads, verboseLevel, outputLevel, globalOptions, nugget, method)
#}
#looErrorsDirect <- function(X, Y, clusters, indices, covType, param, sd2, krigingType = "simple", tagAlgo = "", numThreadsZones = 1L, numThreads = 16L, verboseLevel = 10L, outputLevel = 1L, globalOptions = as.integer( c(0)), nugget = as.numeric( c(0)), method = "NK") {
#  .Call(`_nestedKriging_looErrorsDirect`, X, Y, clusters, indices, covType, param, sd2, krigingType, tagAlgo, numThreadsZones, numThreads, verboseLevel, outputLevel, globalOptions, nugget, method)
#}

