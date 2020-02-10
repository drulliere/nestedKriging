\name{getCorrMatrix}
\alias{getCorrMatrix}
\title{
Correlation matrix of a set of points.
}
\description{
Calculate the correlations between points, based on the spatial location of points, a stationary kernel family and its lengthscales parameters. The function returns correlations, just multiply by the underlying variance to get covariances.

Caution! This is an intermediate calculation of the nestedKriging algorithm, given for side purposes, thus not optimized for a direct output (e.g. monothread here). No checks are performed on arguments.
}
\usage{
getCorrMatrix(X, param, covType)
}
\arguments{
  \item{X}{
Matrix containg points locations. Each row is a point. dimension \eqn{n x d}{n x d}.
}
  \item{param}{
Vector containing lengthscales. dimension \eqn{d}{d} or \eqn{2d}{2d} for \code{"powexp"} kernel. 
}
  \item{covType}{
Covariance kernel family used by Kriging predictors (unless specified, the multivariate kernel is obtained using a tensor product). Must be one of the following: \code{"exp"} (exponential kernel), \code{"gauss"} (gaussian square exponential kernel), \code{"matern3_2"} (Matern 3/2), \code{"matern5_2"} (Matern 5/2), \code{"powexp"} (power exponential kernel), \code{"white_noise"} (white noise kernel), \code{"rational2"} (fast radial rational quadratic kernel, with power 1), \code{"rational1"} (fast radial rational kernel, with norm 1), \code{"gauss.approx"} (fast approximation of gaussian kernel), \code{"exp.approx"} (fast approximation of exponential kernel).
}
}
\details{
See the first part of demo \code{"demoM"} or example below.
}
\value{
Returns a matrix \eqn{n x n}{n x n} containing the correlations.
}
%%\references{
%% ~put references to the literature/web site here ~
%%}
%%\author{
%%  ~~who you are~~
%%}
%%\note{
%%  ~~further notes~~
%%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%%\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
%%}
\examples{
X <- matrix(c(1.1, 2.2, 3.3, 4.5), 2, 2)
lengthScales <- c(1.3, 1.44)
getCorrMatrix(X, lengthScales, "gauss")
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
