\name{looErrors}
\alias{looErrors}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Function to compute leave-one-out prediction errors.
}
\description{
For a given point among design points, the function predicts the output at this point without using the observed output. This is done for all points indicated with a value \code{1} in the argument vector \code{indices}. The function gives the resulting predictions, along with the associated mean square error.
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
looErrors(X, Y, clusters, indices, covType, param, sd2, krigingType = "simple",
          tagAlgo = "", numThreadsZones = 1L, numThreads = 16L, verboseLevel = 10L,
          outputLevel = 1L, globalOptions = as.integer(c(0)),
          nugget = as.numeric(c(0)), method = "NK")
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
  \item{indices}{
vector of \eqn{n}{n} values containing either \code{1} if a point is selected to compute leave-one-out error, or \code{0} otherwise.
}
  \item{covType}{
Covariance kernel family used by Kriging predictors (unless specified, the multivariate kernel is obtained using a tensor product). Must be one of the following: \code{"exp"} (exponential kernel), \code{"gauss"} (gaussian square exponential kernel), \code{"matern3_2"} (Matern 3/2), \code{"matern5_2"} (Matern 5/2), \code{"powexp"} (power exponential kernel), \code{"white_noise"} (white noise kernel), \code{"rational2"} (fast radial rational quadratic kernel, with power 1), \code{"rational1"} (fast radial rational kernel, with norm 1), \code{"approx.gauss"} (fast approximation of gaussian kernel), \code{"approx.exp"} (fast approximation of exponential kernel).
}
  \item{param}{
Lengthscale parameters of the covariance kernel. \code{param} is a vector of size \eqn{d}{d}, where \eqn{d} is the dimension of input points. These parameter correspond to the lengthscale parameters of the chosen covariance kernel.
}
  \item{sd2}{
Variance parameter of the covariance kernel. \code{sd2} is a scalar that gives the variance of the underlying random field. Impacts prediction variance, but not prediction mean.
}
  \item{krigingType}{
Optional. String that specifies if one must use ordinary Kriging or simple Kriging. \code{"simple"}: Simple Kriging, \code{"ordinary"}: Ordinary Kriging (for the first Layer only). Default=\code{"simple"}.
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
  \item{outputLevel}{
Optional. Number giving the detail level of the function output. It is recommended to export only needed values in order to fasten computations. The function \link{outputLevel} may help you to find the right value. \code{0}: only gives prediction mean, variance and some durations, \code{1}: gives also intermediate predictions and weights affected to each of the \eqn{N} subgroup, for each of the \eqn{q} prediction points, \code{2}: gives potentially huge correlation and cross-covariance matrices among submodels, for each prediction point (total size of order \eqn{qN^2}).
Add the value \code{10} to get the conditional \eqn{q \times q}{q x q} covariance matrix between prediction points, which also involves more computation and \eqn{O(nq^2)} storage. Use a value \code{-1} to get alternative predictors (PoE, GPoE, BCM, RBCM, SPV...) or \code{-3} to get both alternatives and nested Kriging predictions. Default=\code{0}.
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
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{
A list containing the same return values as those of the function \code{\link{nestedKriging}}, depending on the choice of \code{outputLevel}. Furthermore, supplementary items in this list are:
\item{LOOexpectedPrediction}{the known values of \eqn{Y}{Y} that are predicted for chosen leave-one-out points. To be compared to the output \code{mean} to obtain leave-one-out errors.}
\item{LOOErrors}{A list containing the mean square error computed for the nested Kriging Method, and for the default method indicated in the argument \code{method}. Also contains two vectors returning the predicted mean and variance for each selected leave-one-out points in \code{indices}, and for the selected default method. Notice that the input argument \code{outputLevel} must be suited to required values.
}
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
%\references{
%% ~put references to the literature/web site here ~
%}
%\author{
%%  ~~who you are~~
%}
\note{See also demo \code{"demoJ"} for a detailed example of use.
%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
%}
\examples{
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
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ spatial  }% use one of  RShowDoc("KEYWORDS")
\keyword{ models	&	Statistical Models & regression }% __ONLY ONE__ keyword per line
