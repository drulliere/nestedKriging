pkgname <- "nestedKriging"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
options(pager = "console")
base::assign(".ExTimings", "nestedKriging-Ex.timings", pos = 'CheckExEnv')
base::cat("name\tuser\tsystem\telapsed\n", file=base::get(".ExTimings", pos = 'CheckExEnv'))
base::assign(".format_ptime",
function(x) {
  if(!is.na(x[4L])) x[1L] <- x[1L] + x[4L]
  if(!is.na(x[5L])) x[2L] <- x[2L] + x[5L]
  options(OutDec = '.')
  format(x[1L:3L], digits = 7L)
},
pos = 'CheckExEnv')

### * </HEADER>
library('nestedKriging')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
cleanEx()
nameEx("estimParam")
### * estimParam

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: estimParam
### Title: Function to estimate the lenghtscale parameters
###   (hyperparameters).
### Aliases: estimParam
### Keywords: spatial models & Statistical Models & regression

### ** Examples

library(nestedKriging)

### chosen model

morris <- function(X) {
  d <- dim(X)[2]
  y <- X[,1]
  for (i in 1:d) {
    y <- y + 2*( (i/2 + X[,i] )/(1+i/2 + X[,i])  - 0.5)
  }
  y <- y + X[,1]*X[,2]
  y
}

f <- morris
d <- 2                   # considered test function is in dimension 2
n <- 300                 # Choice of the number of observations.
covType <- "exp"         # Choice of the parametric family of covariance function
N <- 10                  # Choice of the number of groups
krigingType <- "simple"  # Choice of the Kriging Type "simple" or "ordinary"
q <- 20                  # number of points used to estimate cross-validation errors
sd2 <- 150               # initial variance of the field

### chosen parameters for the stochastic gradient descent

seed <- 1                  # random seed, for reproducibility reasons
niter <- 1000              # number of iterations of the stochastic gradient descent
                           # On purpose we pick a very bad values for:
paramStart <- c(80,0.1)    # This is our initial 'guess' of the covariance parameters.
paramLower <- c(0.01,0.01) # search domain for the covariance parameters : lower bound
paramUpper <- c(100,100)   # search domain for the covariance parameters : upper bound

alpha <- 0.602       # see, book bathnagar et al
gamma <- 0.101       # see, book bathnagar et al
a <- 200             # see, book bathnagar et al
A <- 1               # see, book bathnagar et al
c <- 0.1             # see, book bathnagar et al

set.seed(seed)                                  # For repeatability
X <- matrix(runif(n*d) , ncol=d)                # Design of experiments
Y <-  f(X)                                      # observed responses
clusters <- kmeans(x = X,centers = N)$cluster   # Construction of the clusters

t1 <- Sys.time()
estimation <- nestedKriging::estimParam(X=X, Y=Y, clusters = clusters, q=q,
              covType = covType, niter=niter, paramStart = paramStart,
              paramLower = paramLower, paramUpper = paramUpper, sd2=sd2,
              krigingType=krigingType, seed=seed, alpha = alpha,gamma = gamma,
              a = a,A = A,c = c, tagAlgo="estimParam",numThreads=4,verboseLevel=10,
              nugget=0.0, method = "NK")

t2 <- Sys.time()
difftime(t2,t1)
optimalParam <- estimation$optimalParam
optimalParam

# now computing LOO errors for the optimal parameter
indices <- rep(1, n) # leave-one-out errors will be computed for all prediction points
lastLOOerror <- nestedKriging::looErrors(X=X,Y=Y,clusters=clusters, covType=covType,
                 indices=indices,param=estimation$optimalParam, krigingType=krigingType,
                 sd2=sd2, numThreads=4,verboseLevel=0)
meanSquareError <- lastLOOerror$LOOErrors$looErrorDefaultMethod

message('optimal param = (', paste0(round(optimalParam, digits=4),collapse=","), '),
          having Mean Square Error=', round(meanSquareError,digits=8))



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("estimParam", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("estimSigma2")
### * estimSigma2

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: estimSigma2
### Title: Estimate the prior variance of the underlying random field.
### Aliases: estimSigma2

### ** Examples

##---- see the example of looErrors and also demo 'demoK'



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("estimSigma2", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("looErrors")
### * looErrors

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: looErrors
### Title: Function to compute leave-one-out prediction errors.
### Aliases: looErrors
### Keywords: spatial models & Statistical Models & regression

### ** Examples

library(nestedKriging)
testFunction <- function(x) { x[2] + (x[1]-5*cos(x[2]))^2 + cos(3*x[1]) }

set.seed(8)
d <- 2
n <- 10000
X <- matrix(runif(n*d) , ncol=d)
Y <- apply(X=X,MARGIN = 1,FUN=testFunction)

covType <- "exp"
param <- c(1,1.6)
sd2 <- 150

N <- 100
clusters <- kmeans(x = X,centers = N)$cluster

q <- 100
index <- c(1:q)  # This is the choice of the points for which we compute a LOO error
indices <- rep(0,times=n)
indices[index] <- 1  # we built a vector with a 1 in positions 1,...,100 and 0 otherwise.

krigingType <- "ordinary"

t1 <- Sys.time()
estimation <- nestedKriging::looErrors(X=X, Y=Y, clusters=clusters, indices=indices ,
                  covType=covType, param=param, sd2=sd2, krigingType=krigingType,
                  tagAlgo='nested Kriging LOO', numThreads=8, verboseLevel=0,
                  outputLevel=0, globalOptions = c(0), nugget=0.0)
t2 <- Sys.time()
difftime(t2,t1)
predictedMeans <- estimation$mean
expectedValues <- estimation$LOOexpectedPrediction
meanSquareErrorMethod1 <- mean((predictedMeans-expectedValues)^2)
meanSquareErrorMethod2 <- estimation$LOOErrors$looErrorNestedKriging

message('the obtained mean square error is ',meanSquareErrorMethod1,' = ',meanSquareErrorMethod2)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("looErrors", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("looErrorsDirect")
### * looErrorsDirect

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: looErrorsDirect
### Title: Compute leave-one-out mean square error with minimal export.
### Aliases: looErrorsDirect

### ** Examples

## see example of looErrors and demo 'demoK'



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("looErrorsDirect", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("nestedKriging-package")
### * nestedKriging-package

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: nestedKriging-package
### Title: Nested Kriging Predictions for Large Datasets
### Aliases: nestedKriging-package

### ** Examples

# first launch some tests of the program
tests_run()

# example 1, a simple example of nestedKriging with small datasets
library(nestedKriging)
set.seed(1)

testFunction <- function(x) { x[2] + (x[1]-5*cos(x[2]))^2 + cos(3*x[1]) }

X <- matrix(runif(1000*2), ncol=2)              # 1000 initial design points, in dimension 2
Y <- apply(X=X, MARGIN = 1, FUN=testFunction)   # initial response for each design points
x <- matrix(runif(100*2), ncol=2)               # 100 prediction points, in dimension 2
clustering <- kmeans(X, centers=20)             # clustering of design points X into 20 groups

prediction <- nestedKriging(X=X, Y=Y, clusters=clustering$cluster, x=x ,
                            covType="matern5_2", param=c(1,1), sd2=10,
                            krigingType="simple", tagAlgo='example 1', numThreads=5)


realvalues <- apply(x, MARGIN = 1, FUN = testFunction) #real values to be predicted
pred_errors <- abs(realvalues - prediction$mean)
mean_error_Nested <- mean(pred_errors)
message("mean error Nested Kriging = ", mean_error_Nested)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("nestedKriging-package", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("nestedKriging")
### * nestedKriging

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: nestedKriging
### Title: Compute Nested Kriging Mean and Variance Predictions
### Aliases: nestedKriging
### Keywords: spatial models & Statistical Models & regression

### ** Examples

# example 1, a simple example of nestedKriging with small datasets
library(nestedKriging)
set.seed(1)

testFunction <- function(x) { x[2] + (x[1]-5*cos(x[2]))^2 + cos(3*x[1]) }

X <- matrix(runif(1000*2), ncol=2)              # 1000 initial design points, in dimension 2
Y <- apply(X=X, MARGIN = 1, FUN=testFunction)   # initial response for each design points
x <- matrix(runif(100*2), ncol=2)               # 100 prediction points, in dimension 2
clustering <- kmeans(X, centers=20)             # clustering of design points X into 20 groups

prediction <- nestedKriging(X=X, Y=Y, clusters=clustering$cluster, x=x ,
                            covType="matern5_2", param=c(1,1), sd2=10,
                            krigingType="simple", tagAlgo='example 1', numThreads=5)


realvalues <- apply(x, MARGIN = 1, FUN = testFunction) #real values to be predicted
pred_errors <- abs(realvalues - prediction$mean)
mean_error_Nested <- mean(pred_errors)
message("mean error Nested Kriging = ", mean_error_Nested)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("nestedKriging", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("setNumThreadsBLAS")
### * setNumThreadsBLAS

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: setNumThreadsBLAS
### Title: Set the Number of Threads Used by External Linear Algebra
###   Libraries (BLAS)
### Aliases: setNumThreadsBLAS

### ** Examples

library(nestedKriging)
setNumThreadsBLAS(1)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("setNumThreadsBLAS", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("tests_getCaseStudy")
### * tests_getCaseStudy

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: tests_getCaseStudy
### Title: Returns a Case Study Example for Manual Testing of the Package
### Aliases: tests_getCaseStudy

### ** Examples

library(nestedKriging)
caseStudyOne <- tests_getCaseStudy(1, "gauss")



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("tests_getCaseStudy", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("tests_getCodeValues")
### * tests_getCodeValues

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: tests_getCodeValues
### Title: Returns the Nested Kriging Predicion for a Selected Case Study,
###   for Manual Testing of the Package
### Aliases: tests_getCodeValues

### ** Examples

library(nestedKriging)
myResults <- tests_getCodeValues(1, "gauss", forceSimpleKriging = TRUE)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("tests_getCodeValues", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("tests_run")
### * tests_run

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: tests_run
### Title: Run all Implemented Tests for the 'nestedKriging' Package
### Aliases: tests_run

### ** Examples

library(nestedKriging)
tests_run(showSuccess = TRUE)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("tests_run", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("versionInfo")
### * versionInfo

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: versionInfo
### Title: Gives Information about the Version of this Package
### Aliases: versionInfo

### ** Examples

library(nestedKriging)
myinformations <- versionInfo()



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("versionInfo", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
### * <FOOTER>
###
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
