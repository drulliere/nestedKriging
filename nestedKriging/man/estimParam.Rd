\name{estimParam}
\alias{estimParam}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Function to estimate the lenghtscale parameters (hyperparameters).
}
\description{
The function \code{estimParam} estimates the lenghtscale parameters \code{param} to be used in the function \code{nestedKriging}. It uses a stochastic gradient descent which minimizes some Leave-One-Out cross-validation errors.
}
\usage{
estimParam(X, Y, clusters, q, covType, niter, paramStart, paramLower, paramUpper, sd2,
krigingType = "simple", seed = 0L, alpha = 0.602, gamma = 0.101, a = 200, A = 1, c = 0.1,
tagAlgo = "", numThreadsZones = 1L, numThreads = 16L, verboseLevel = 10L,
globalOptions = as.integer(c(0)), nugget = as.numeric(c(0)), method = "NK")
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{
Initial design points. \code{X} is a \eqn{n \times d}{n x d} matrix, where \eqn{n} is the number of input points and \eqn{d} is the dimension of each point (each line of \code{X} is a design input point).
}
  \item{Y}{
Initial responses. \code{Y} is a vector of responses of size \eqn{n}{n}. Each element of this vector is the value of an observed function at corresponding input point of \code{X}.
}
  \item{clusters}{
Cluster index of each input points. \code{clusters} is a vector of size  \eqn{n}{n} that gives the group number of each input point (i.e. the cluster to which each point is allocated). If input points are clustered into \eqn{N} groups (where \eqn{N} in \eqn{1..n}), then each value in \code{clusters} typically belongs to \eqn{1..N}. However, empty groups are allowed, and group numbers can also start from \eqn{0}. The \code{cluster} return value of the \code{kmeans} external procedure is a typical example of \code{clusters} input value.
}
  \item{q}{
number of Leave-One-Out points used to optimize the Leave-One-out error, among the \eqn{n}{n} input points.
}
  \item{covType}{
Covariance kernel family used by Kriging predictors (the multivariate kernel is obtained using a tensor product). Must be one of the following: \code{"exp"} (exponential kernel), \code{"gauss"} (gaussian square exponential kernel), \code{"matern3_2"} (Matern 3/2), \code{"matern5_2"} (Matern 5/2), \code{"powexp"} (power exponential kernel), \code{"white_noise"} (white noise kernel), \code{"rational"} (fast rational quadratic kernel, with power 1).
}
  \item{niter}{
Number of iterations of the stochastic gradient descent.
}
  \item{paramStart}{
Vector of size \eqn{d}{d} containing the initial value of the lengthscale parameter, used for the Stochastic gradient descent.}
  \item{paramLower}{
Vector of size \eqn{d}{d} containing the lower bounds for the value of the lengthscale parameter.
}
  \item{paramUpper}{
Vector of size \eqn{d}{d} containing the lower bounds for the value of the lengthscale parameter.
}
  \item{sd2}{
Variance parameter of the covariance kernel. \code{sd2} is a scalar that gives the variance of the underlying random field. Impacts prediction variance, but not prediction mean.
}
  \item{krigingType}{
Optional. String that specifies if one must use ordinary Kriging or simple Kriging. \code{"simple"}: Simple Kriging, \code{"ordinary"}: Ordinary Kriging (for the first Layer only). Default=\code{"simple"}.
}
  \item{seed}{
Optional. Seed used for the random generation of the chosen \eqn{q}{q} points where are evaluated the Leave-One-Out errors(results look random but a same seed gives the same results, so that a study can be reproduced).
}
  \item{alpha}{
Optional. scalar stochastic gradient descent parameter, see, book Bathnagar et al.
Default=\code{alpha = 0.602}.
}
  \item{gamma}{
Optional. scalar stochastic gradient descent parameter, see, book Bathnagar et al.
Default=\code{gamma = 0.101}.
}
  \item{a}{
Optional. scalar stochastic gradient descent parameter, see, book Bathnagar et al.
Default=\code{a = 200}.
}
  \item{A}{
Optional. scalar stochastic gradient descent parameter, see, book Bathnagar et al.
Default=\code{A = 1}.
}
  \item{c}{
Optional. scalar stochastic gradient descent parameter, see, book Bathnagar et al.
Default=\code{c = 0.1}.
}
  \item{tagAlgo}{
Optional. A string that will appear on output messages of the function. Useful to distinguish between outputs, or to check the progression of an algorithm that launches \code{nestedKriging} several times. Default=\code{""}.
}
\item{numThreadsZones}{
Optional (rare usage, experimental), recommended value=\code{1}. Number of threads used for prediction points. Divides the \eqn{q} prediction points into \code{numThreadsZones} separate zones, and run parallel independent predictions for each zone. Values larger than \code{1} may eventually be used in very specific cases: number of subgroups lower than the number of cores, large number of prediction points, specific architectures, false sharing problems... Default=\code{1}.
}
\item{numThreads}{
Optional. Number of threads used for parallel execution on subgroups (clusters), should be less than \eqn{N(N-1)/2} where \eqn{N} is the number of subgroups. Recommended = number of logical cores of your computer. Default=\eqn{16}.
}
\item{verboseLevel}{
Optional. Number of intermediate messages shown during the calculation. Default=\code{10}. 0= no messages, but eventual warnings. Negative= no messages, no warnings. Large number may be suited for very long calculations, small number for repeated calls of \code{nestedKriging}. Positive values may induce a slight computational overhead.
}
\item{globalOptions}{
Optional (rare usage), for developers only. A vector of integers containing global options that are used for development purposes. Useful for comparing different implementation choices. Default=\code{as.integer(c(0))}.
}
\item{nugget}{
Optional, a vector containing variances that will be added to the diagonal of the covariance matrix of \eqn{X}. If a real is used instead of a vector, or if the vector is of length lower than the number of rows \eqn{n} of the matrix \eqn{X}, the pattern is repeated along the diagonal. Default=\code{c(0.0)}.
}
\item{method}{
Optional, a string containing the default method to be used for evaluating Leave-One-Out errors: \code{"NK"} for nested Kriging, or \code{"POE"}, \code{"GPOE"}, \code{"GPOE_1N"}, \code{"BCM"}, \code{"RBCM"}, \code{"SPV"} for fast less precise alternatives (see output \code{alternatives} of the function \code{\link{nestedKriging}}, and also \url{http://proceedings.mlr.press/v37/deisenroth15.pdf}). Computed alternatives predictors: (Generalised) Product of Expert (POE/GPOE), (Robust) Bayesian Comittee Machine (BCM/RBCM) and Smallest Predictive Variance (SPV). Weights of GPOE are summing to one, they are proportional to the one of RBCM (differential entropy difference between the prior and the posterior), or set to \eqn{1/N}{1/N} for the result GPOE_1N. SPV gives the prediction of the submodel having the smallest conditional variance.
}
}
%%\details{
%%  ~~ If necessary, more details than the description above ~~
%%}
\value{
A list containg the following items:
\item{allParamIterations}{A matrix containing all iterations. Each line number \eqn{i}{i} contains the current parameter at iteration \eqn{i}{i}.}
\item{allErrorIterations}{A matrix containing all errors. Each line number \eqn{i}{i} contains the current leave-one-out error at iteration \eqn{i}{i}.}
\item{optimalParam}{The parameter obtained at the end of the stochastic gradient descent.}
\item{errorPointA}{The leave-one-out error obtained with the parameter \code{paramPointA}}
\item{errorPointB}{The leave-one-out error obtained with the parameter \code{paramPointB}}
\item{paramPointA}{One of the two points used for estimating the gradient, during the last iteration.}
\item{paramPointB}{One of the two points used for estimating the gradient, during the last iteration.}
\item{bestEncounteredParam}{Best parameter encountered during the stochastic gradient descent, in the sense that it minimizes the computed Leave-One-Out error. May be unstable (depends on the chosen \eqn{q}{q} points). }
\item{bestEncounteredError}{computed Leave-One-Out error at the point \code{bestEncounteredParam}.}
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
see, book bathnagar et al. for the stochastic Gradient Descent: Bhatnagar, S., Prasad, H. L., Prashanth, L. A. (2012). Stochastic recursive algorithms for optimization: simultaneous perturbation methods (Vol. 434). Springer.
}
%\author{
%%  ~~who you are~~
%}
%\note{
%%  ~~further notes~~
%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
many input parameters are similar to the ones of the function \code{\link{nestedKriging}}
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
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
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ spatial  }% use one of  RShowDoc("KEYWORDS")
\keyword{ models	&	Statistical Models & regression }% __ONLY ONE__ keyword per line
