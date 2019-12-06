\name{nestedKriging}
\alias{nestedKriging}

\title{
Compute Nested Kriging Mean and Variance Predictions
}

%%\section{Other code examples}{
%%Some other demos \code{R} code are accessible:
%%
%%- \code{demo("nestedKrigingExample1")} or \code{example("nestedKriging")} runs above minimal code example.
%%
%%- \code{demo("nestedKrigingExample2")} runs a nested Kriging prediction in dimension 6 with 100000 points (requires package \code{DiceKriging}).
%%
%%}

%%\section{Disclaimer}{
%%This is a preliminary program, which is for illustrative purposes only. It is provided "as is" and "with all faults", without any warranties and support.
%%%%The authors assume no responsibility or liability for the use of the program. The authors makes no representations or warranties of any kind concerning the safety, suitability, lack of viruses, inaccuracies, typographical errors, or other harmful components of this program. There are inherent dangers in the use of any software, and you are solely responsible for determining whether this program is compatible with your equipment and other software installed on your equipment. You are also solely responsible for the protection of your equipment and backup of your data, and the authors will not be liable for any damages you may suffer in connection with using, modifying, or distributing this program. The authors reserves the right to make changes in the software without notification. The authors make no representation or warranty that such application will be suitable for the specified use without further testing or modification.
%%}

\description{
%%(\emph{Caution: this is a preliminary program, which is for illustrative purposes only, see disclaimer below}).
Predict the value of a multivariate function given observations of this function at some input points.

When a function \code{f} is observed only at some input points \code{X}, \code{nestedKriging} computes nested Kriging predictions at some given prediction points \code{x}: observations are \code{Y=f(X)}, where response vector \code{Y} and design matrix \code{X} are given, and the \code{nestedKriging} function gives a prediction of \code{f(x)}. The function typically returns mean and variance of the predictor.

The function requires dividing the \eqn{n} input points into \eqn{N} subgroups. It first calculates a Kriging predictor on each subgroup, and then aggregate predictors into a unique predictor. This way, it allows doing Kriging on larger datasets, as soon as \eqn{N} and \eqn{n/N} are not too large.

Hereafter, the predictor based on a subgroup of design points is called a submodel. In short, the function calculates mean and covariances of each submodel, then aggregate submodels into one predictor.
}
\usage{
nestedKriging(X, Y, clusters, x, covType, param, sd2, krigingType="simple", tagAlgo = "",
numThreadsZones = 1L, numThreads = 16L, verboseLevel = 10L, outputLevel = 0L,
numThreadsBLAS = 1L, globalOptions=as.integer(c(0)), nugget = c(0.0))
}
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
  \item{x}{
Prediction points. \code{x} is a \eqn{q \times d}{q x d} matrix, where \eqn{q} is the number of points where we want to predict the function \code{f}, and \eqn{d} is the dimension of each point (each line of \code{x} is a prediction point).
}
  \item{covType}{
Covariance kernel family used by Kriging predictors (unless specified, the multivariate kernel is obtained using a tensor product). Must be one of the following: \code{"exp"} (exponential kernel), \code{"gauss"} (gaussian square exponential kernel), \code{"matern3_2"} (Matern 3/2), \code{"matern5_2"} (Matern 5/2), \code{"powexp"} (power exponential kernel), \code{"white_noise"} (white noise kernel), \code{"rational"} (fast radial rational quadratic kernel, with power 1), \code{"approx.gauss"} (fast approximation of gaussian kernel).
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
  \item{numThreads}{
Optional. Number of threads used for parallel execution on subgroups (clusters), should be less than \eqn{N(N-1)/2} where \eqn{N} is the number of subgroups. Recommended = number of logical cores of your computer. Default=\eqn{16}.
}
  \item{verboseLevel}{
Optional. Number of intermediate messages shown during the calculation. Default=\code{10}. 0= no messages, but eventual warnings. Negative= no messages, no warnings. Large number may be suited for very long calculations, small number for repeated calls of \code{nestedKriging}. Positive values may induce a slight computational overhead.
}

  \item{outputLevel}{
Optional. Number giving the detail level of the function output. It is recommended to export only needed values in order to fasten computations. \code{0}: only gives prediction mean, variance and some durations, \code{1}: gives also intermediate predictions and weights affected to each of the \eqn{N} subgroup, for each of the \eqn{q} prediction points, \code{2}: gives potentially huge correlation and cross-covariance matrices among submodels, for each prediction point (total size of order \eqn{qN^2}).
Add the value \code{10} to get the conditional \eqn{q \times q}{q x q} covariance matrix between prediction points, which also involves more computation and \eqn{O(nq^2)} storage. Use a value \code{-1} to get alternative predictors (PoE, GPoE, BCM, RBCM, SPV...) or \code{-3} to get both alternatives and nested Kriging predictions. Default=\code{0}.
}

  \item{numThreadsZones}{
Optional (rare usage, experimental), recommended value=\code{1}. Number of threads used for prediction points. Divides the \eqn{q} prediction points into \code{numThreadsZones} separate zones, and run parallel independent predictions for each zone. Values larger than \code{1} may eventually be used in very specific cases: number of subgroups lower than the number of cores, large number of prediction points, specific architectures, false sharing problems... Default=\code{1}.
}
  \item{numThreadsBLAS}{
Optional (rare usage), recommended value=\code{1}. Number of threads used by external linear algebra libraries (BLAS). When BLAS uses more than one thread by default, it uses threads less efficiently than via \code{numThreads}, so that the recommended setting is \code{numThreadsBLAS=1}. Other settings may be useful in very specific cases: number of subgroups lower than the number of cores, other BLAS uses... This threads number is adjusted using external \code{R} package \code{RhpcBLASctl}. Default=\code{1}.
}
\item{globalOptions}{
Optional (rare usage), for developers only. A vector of integers containing global options that are used for development purposes. Useful for comparing different implementation choices. Default=\code{as.integer(c(0))}.
}
\item{nugget}{
Optional, a vector containing variances that will be added to the diagonal of the covariance matrix of \eqn{X}. If a real is used instead of a vector, or if the vector is of length lower than the number of rows \eqn{n} of the matrix \eqn{X}, the pattern is repeated along the diagonal. Default=\code{c(0.0)}.
}
}
%%\details{
%%}
\value{
The return value is a list containing following items
\item{mean}{Vector containing the mean prediction at each prediction point in \code{x}. The vector \code{mean} has size \eqn{q}{q}, where \eqn{q} is the number of prediction points.}
\item{sd2}{Vector containing the marginal variance prediction at each prediction point in \code{x}. The vector \code{sd2} has size \eqn{q}{q}, where \eqn{q} is the number of prediction points.}
\item{cov}{Conditional covariances between predictions at prediction points (under interpolation assumption). \code{cov} is a \eqn{q \times q}{q x q} matrix containing covariances given observations \eqn{Y(X)}{Y(X)}. \code{cov} is available if the argument \code{outputLevel} is greater than 10, which involves more computations and \eqn{O(nq^2)}{O(nq^2)} supplementary storage capacity. (see demo \code{"demoH"} for building conditional sample paths using \code{cov})}
\item{covPrior}{Unconditional covariances between predictions at prediction points (under interpolation assumption). \code{covPrior} is a \eqn{q \times q}{q x q} matrix containing prior covariances without considering observations \eqn{Y(X)}{Y(X)}. \code{cov} and \code{covPrior} are available if the argument \code{outputLevel} is greater than 10, which involves more computations and \eqn{O(nq^2)}{O(nq^2)} supplementary storage capacity.}
\item{duration}{Scalar containing the total duration, in seconds, of the internal \code{C++} algorithm.}
\item{durationDetails}{Dataframe containing the durations, in seconds, of different steps of the algorithm, and associated step names: \code{"partA"} computes kriging predictors on each subgroups, for all prediction points. \code{"partB"} computes cross-covariances between subgroups predictors. \code{"partC"} aggregates all subgroups predictors, using their cross-covariances. \code{"partD"}, when needed, finishes the computation of conditional covariances between prediction points. \code{"partE"}, when needed, finishes the computation of alternative predictors (POE, BCM, etc.)}
\item{sourceCode}{String containing the name of the algorithm and its version. It can be useful to ensure the replicability of some results, and to avoid confusions when comparing results with those obtained by other algorithms.}
\item{weights}{Matrix giving weights affected to each submodel, for each prediction point. \code{weights} is a \eqn{N \times q}{N x q} matrix, where \eqn{N} is the number of subgroups, and \eqn{q} is the number of prediction points. \code{weights} is empty if the argument \code{outputLevel} is strictly lower than 1.}
\item{mean_M}{List giving mean predictions for each submodel. \code{mean_M} is a \eqn{N \times q}{N x q} matrix. Each column corresponds to one prediction point; for this prediction point, the considered column gives the \eqn{N} predictions based on each subgroup, where \eqn{N} is the number of subgroups and \eqn{q} is the number of prediction points. Empty if the argument \code{outputLevel} is strictly lower than 1.}
\item{sd2_M}{Matrix giving conditional variance predictions for each submodel. \code{sd2_M} is a \eqn{N \times q}{N x q} matrix. Each column corresponds to one prediction point; for this prediction point, the considered column gives the \eqn{N} conditional prediction variances based on each subgroup, where \eqn{N} is the number of subgroups and \eqn{q} is the number of prediction points. Empty if the argument \code{outputLevel} is strictly lower than 1.}
\item{K_M}{(upper case K) Computed cross-covariances among submodels (i.e. among the \eqn{N} predictors based on each subgroup of design points). \code{K_M} is a list with \eqn{q} items, each item gives the \eqn{N^2}{N^2} components of the covariance matrix for one prediction point. \code{K_M} is empty if the argument \code{outputLevel} is strictly lower than 2.}
\item{k_M}{(lower case k) Computed covariances between submodels and response at prediction points. \code{k_M} is a list with \eqn{q} items, each item correspond to one prediction point, and gives the \eqn{N} components of the covariance vector between submodels and the considered prediction point response. \code{k_M} is empty if the argument \code{outputLevel} is strictly lower than 2.}
\item{Alternatives}{Computed alternatives predictors: (Generalised) Product of Expert (POE/GPOE), (Robust) Bayesian Comittee Machine (BCM/RBCM) and Smallest Predictive Variance (SPV). Dataframe containing for each predictor, the conditional mean and the conditional variance of the predictor given observations \eqn{Y(X)}{Y(X)}. Weights of GPOE are summing to one, they are proportional to the one of RBCM (differential entropy difference between the prior and the posterior), or set to \eqn{1/N}{1/N} for the result GPOE_1N. SPV gives the prediction of the submodel having the smallest conditional variance. For details, see \url{http://proceedings.mlr.press/v37/deisenroth15.pdf}. See demo \code{"demoI"} for a demo using alternatives.}
}
%%\references{
%%Rullière, D., Durrande, N., Bachoc, F., Chevalier, C. (2017), Nested Kriging predictions for datasets with a large number of observations, Statistics and Computing, in press. doi: 10.1007/s11222-017-9766-2. Preprint: \url{https://arxiv.org/abs/1607.05432}, paper: \url{https://dx.doi.org/10.1007/s11222-017-9766-2}.
%%}
%%\author{
%%  ~~who you are~~
%%}
\note{
The underlying Gaussian random field is assumed to be centered with a stationary covariance function, so that you must ensure that the response \eqn{Y}{Y} has been centered, that its eventual trend has been suppressed, and that the stationarity has been checked: the nested Kriging predictor provides in theory more accurate results that other mixture of experts, provided that the model assumptions are valid. If other alternatives (POE, BCM, SPV...) provide better results, then you probably need to distort both \eqn{X}{X} and \eqn{Y}{Y} in order to validate the model assumptions.

Consider using the function \code{\link{nestedKrigingDirect}} if you have repeated calls to \code{nestedKriging}. It will disable some checks and save computational time.

Notice that the total number of threads used for calculation is the product of \code{numThreads}, \code{numThreadsZones} and \code{numThreadsBLAS} (with default values \code{numThreadsZones}=\code{numThreadsBLAS}=1).

For clustering, in practice, using an approximate number of groups \eqn{N= sqrt(n)}{N = \sqrt n} obtained with kmeans procedure gives fast results. A low value of \eqn{N} gives more accurate results, but far slower to compute (the value \eqn{N=1} gives classical kriging Predictors).
}

\seealso{
see also details and references in the package help \code{\link{nestedKriging-package}} and paper \url{https://dx.doi.org/10.1007/s11222-017-9766-2}.

You can also launch some demos bu typing \code{demo(demoA)}, ..., \code{demo(demoI)} or \code{example(nestedKriging)}.
}

\examples{
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
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ spatial  }% use one of  RShowDoc("KEYWORDS")
\keyword{ models	&	Statistical Models & regression }% __ONLY ONE__ keyword per line


