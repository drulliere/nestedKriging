\name{getKrigingPrediction}
\alias{getKrigingPrediction}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Compute classical kriging predictions
}
\description{
Compute classical kriging predictions, using either Simple or Ordinary Kriging.

Caution! This is NOT nestedKriging prediction. This is an intermediate calculation of the nestedKriging algorithm, given for side purposes, thus not optimized for a direct output (e.g. monothread here). No checks are performed on arguments. For nestedKriging predictions see the function \code{\link{nestedKriging}}.
}
\usage{
getKrigingPrediction(X, Y, x, param, covType, krigingType = "simple")
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{
Matrix containg points locations. Each row is a point. dimension \eqn{n x d}{n x d}.
}
  \item{Y}{
Initial responses. \code{Y} is a vector of responses of size \eqn{n}{n}. Each element of this vector is the value of an observed function at corresponding input point of \code{X}.
}
  \item{x}{
Prediction points. \code{x} is a \eqn{q \times d}{q x d} matrix, where \eqn{q} is the number of points where we want to predict the function \code{f}, and \eqn{d} is the dimension of each point (each line of \code{x} is a prediction point).
}
  \item{param}{
Vector containing lengthscales. dimension \eqn{d}{d} or \eqn{2d}{2d} for \code{"powexp"} kernel. 
}
  \item{covType}{
Covariance kernel family used by Kriging predictors (unless specified, the multivariate kernel is obtained using a tensor product). Must be one of the following: \code{"exp"} (exponential kernel), \code{"gauss"} (gaussian square exponential kernel), \code{"matern3_2"} (Matern 3/2), \code{"matern5_2"} (Matern 5/2), \code{"powexp"} (power exponential kernel), \code{"white_noise"} (white noise kernel), \code{"rational2"} (fast radial rational quadratic kernel, with power 1), \code{"rational1"} (fast radial rational kernel, with norm 1), \code{"gauss.approx"} (fast approximation of gaussian kernel), \code{"exp.approx"} (fast approximation of exponential kernel).
}
  \item{krigingType}{
Optional. String that specifies if one must use ordinary Kriging or simple Kriging. \code{"simple"}: Simple Kriging, \code{"ordinary"}: Ordinary Kriging (for the first Layer only). Default=\code{"simple"}.
}
}
\details{
See the second part of demo \code{"demoM"} or example below.
%%  ~~ If necessary, more details than the description above ~~
}
\value{
Returns a list containing
\item{mean}{The Kriging mean}
\item{unitVariance}{The Kriging variance, for a unit variance of the initial field. Just multiply by \code{sd2} for a field having a prior stationnary variance of \code{sd2}.}
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
# design matrix
X <- matrix(c(1.1, 2.2, 3.3, 4.5, 1.2, 1.3, 8.4, 2.1), 4, 2)
# observed responses
Y <- c(1.3, 2.5, 4.5, 6.7)
# new prediction points
xnew <- matrix(c(1.3, 2.6, 3.1, 4.2, 5.3, 6.7), 3, 2)

lengthScales <- c(1.3, 1.44)

covType <- "gauss"
krigingType <- "simple"

pred <- getKrigingPrediction(X, Y, xnew, lengthScales, covType, krigingType)
pred$mean
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
