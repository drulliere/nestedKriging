\name{nestedKrigingDirect}
\alias{nestedKrigingDirect}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Same Function as \code{nestedKriging}, Without Checking Arguments and Without Adjusting the Number of Threads of Linear Algebra Library BLAS
}
\description{
Same function as the function \code{\link{nestedKriging}}, by direct call of a \code{C++} program, without preliminary checks of the arguments validity, and without adjusting the number of threads of linear algebra library BLAS. May be useful in case of a large number of call of the function, or when the default setting of BLAS is set externally.
}
%%\usage{
%%nestedKrigingDirect(X, Y, clusters, x, covtype, param, sd2, OrdinaryKriging = FALSE, tagAlgo = "", numThreadsZones = 0L, numThreads = 0L, verboseLevel = 10L, outputLevel = 1L)}
%- maybe also 'usage' for other objects documented here.
%%\arguments{see the function \code{nestedKriging}.}
\details{same usage and arguments as the function \code{\link{nestedKriging}}, without \code{numThreadsBLAS} parameter.
%%  ~~ If necessary, more details than the description above ~~
}
\value{same output value as the function \code{\link{nestedKriging}}.
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
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
