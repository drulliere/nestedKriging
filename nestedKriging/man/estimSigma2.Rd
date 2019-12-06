\name{estimSigma2}
\alias{estimSigma2}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Estimate the prior variance of the underlying random field.}
\description{
Estimate the prior variance of the underlying random field, using leave-one-out errors. Usually used after the estimation of the lengthscales of the covariance kernel, via the function \code{\link{estimParam}}.
}
\usage{
estimSigma2(X, Y, q, clusters, covType, krigingType, param, numThreads = 8, nugget = 0,
            method = "NK", seed = 0)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{
Initial design points. \code{X} is a \eqn{n \times d}{n x d} matrix, where \eqn{n} is the number of input points and \eqn{d} is the dimension of each point (each line of \code{X} is a design input point).
}
  \item{Y}{
Initial responses. \code{Y} is a vector of responses of size \eqn{n}{n}. Each element of this vector is the value of an observed function at corresponding input point of \code{X}.
}
  \item{q}{
number of Leave-One-Out points used to optimize the Leave-One-out error, among the \eqn{n}{n} input points.
}
  \item{clusters}{
Cluster index of each input points. \code{clusters} is a vector of size  \eqn{n}{n} that gives the group number of each input point (i.e. the cluster to which each point is allocated). If input points are clustered into \eqn{N} groups (where \eqn{N} in \eqn{1..n}), then each value in \code{clusters} typically belongs to \eqn{1..N}. However, empty groups are allowed, and group numbers can also start from \eqn{0}. The \code{cluster} return value of the \code{kmeans} external procedure is a typical example of \code{clusters} input value.
}
  \item{covType}{
Covariance kernel family used by Kriging predictors (unless specified, the multivariate kernel is obtained using a tensor product). Must be one of the following: \code{"exp"} (exponential kernel), \code{"gauss"} (gaussian square exponential kernel), \code{"matern3_2"} (Matern 3/2), \code{"matern5_2"} (Matern 5/2), \code{"powexp"} (power exponential kernel), \code{"white_noise"} (white noise kernel), \code{"rational"} (fast radial rational quadratic kernel, with power 1), \code{"approx.gauss"} (fast approximation of gaussian kernel).
}
  \item{krigingType}{
Optional. String that specifies if one must use ordinary Kriging or simple Kriging. \code{"simple"}: Simple Kriging, \code{"ordinary"}: Ordinary Kriging (for the first Layer only). Default=\code{"simple"}.
}
  \item{param}{
Lengthscale parameters of the covariance kernel. \code{param} is a vector of size \eqn{d}{d}, where \eqn{d} is the dimension of input points. These parameter correspond to the lengthscale parameters of the chosen covariance kernel.
}
  \item{numThreads}{
Optional. Number of threads used for parallel execution on subgroups (clusters), should be less than \eqn{N(N-1)/2} where \eqn{N} is the number of subgroups. Recommended = number of logical cores of your computer. Default=\eqn{16}.
}
  \item{nugget}{
Optional, a vector containing variances that will be added to the diagonal of the covariance matrix of \eqn{X}. If a real is used instead of a vector, or if the vector is of length lower than the number of rows \eqn{n} of the matrix \eqn{X}, the pattern is repeated along the diagonal. Default=\code{c(0.0)}.
}
  \item{method}{
Optional, a string containing the default method to be used for evaluating Leave-One-Out errors: \code{"NK"} for nested Kriging, or \code{"POE"}, \code{"GPOE"}, \code{"GPOE_1N"}, \code{"BCM"}, \code{"RBCM"}, \code{"SPV"} for fast less precise alternatives (see output \code{alternatives} of the function \code{\link{nestedKriging}}, and also \url{http://proceedings.mlr.press/v37/deisenroth15.pdf}). Computed alternatives predictors: (Generalised) Product of Expert (POE/GPOE), (Robust) Bayesian Comittee Machine (BCM/RBCM) and Smallest Predictive Variance (SPV). Weights of GPOE are summing to one, they are proportional to the one of RBCM (differential entropy difference between the prior and the posterior), or set to \eqn{1/N}{1/N} for the result GPOE_1N. SPV gives the prediction of the submodel having the smallest conditional variance.
}
  \item{seed}{
Optional. Seed used for the random generation of the chosen \eqn{q}{q} points where are evaluated the Leave-One-Out errors(results look random but a same seed gives the same results, so that a study can be reproduced).
}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{
A scalar cntaining the estimated variance of the underlying random field \code{Y}.
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
%\note{
%%  ~~further notes~~
%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
%}
\examples{
##---- see the example of looErrors and also demo 'demoK'
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
