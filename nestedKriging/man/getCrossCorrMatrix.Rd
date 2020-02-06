\name{getCrossCorrMatrix}
\alias{getCrossCorrMatrix}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Correlation matrix between two sets of points.
}
\description{
Calculate the cross-correlations between two sets of points, based on the spatial location of points, a stationary kernel family and its lengthscales parameters. The function returns correlations, just multiply by the underlying variance to get covariances.

This is an intermediate calculation of the nestedKriging algorithm, given for side purposes, thus not optimized for a direct output (e.g. monothread here).
}
\usage{
getCrossCorrMatrix(X1, X2, param, covType)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X1}{
Matrix containg points locations. Each row is a point. dimension \eqn{n x d}{n x d}.
}
  \item{X2}{
Matrix containg points locations. Each row is a point. dimension \eqn{m x d}{m x d}.
}
  \item{param}{
Vector containing lengthscales. dimension \eqn{d}{d} or \eqn{2d}{2d} for \code{"powexp"} kernel. 
}
  \item{covType}{
Covariance kernel family used by Kriging predictors (unless specified, the multivariate kernel is obtained using a tensor product). Must be one of the following: \code{"exp"} (exponential kernel), \code{"gauss"} (gaussian square exponential kernel), \code{"matern3_2"} (Matern 3/2), \code{"matern5_2"} (Matern 5/2), \code{"powexp"} (power exponential kernel), \code{"white_noise"} (white noise kernel), \code{"rational2"} (fast radial rational quadratic kernel, with power 1), \code{"rational1"} (fast radial rational kernel, with norm 1), \code{"gauss.approx"} (fast approximation of gaussian kernel), \code{"exp.approx"} (fast approximation of exponential kernel).
}
}
%%\details{
%%  ~~ If necessary, more details than the description above ~~
%%}
\value{
Returns a matrix \eqn{n x m}{n x m} containing the correlations.
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
X1 <- matrix(c(1.1, 2.2, 3.3, 4.5), 2, 2)
X2 <- matrix(c(1.3, 2.6, 3.1, 4.2, 5.3, 6.7), 3, 2)
lengthScales <- c(1.3, 1.44)
covType <- "gauss"
getCrossCorrMatrix(X1, X2, lengthScales, covType)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%%\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
%%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
