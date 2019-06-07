\name{looErrorsDirect}
\alias{looErrorsDirect}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Compute leave-one-out mean square error with minimal export.
}
\description{
Same as \code{\link{looErrors}}, but with a minimal export value.
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
%\usage{
%looErrorsDirect(X, Y, clusters, indices, covType, param, sd2, krigingType = "simple",
%            tagAlgo = "", numThreadsZones = 1L, numThreads = 16L, verboseLevel = 10L,
%            outputLevel = 1L, globalOptions = as.integer(c(0)),
%            nugget = as.numeric(c(0)), defaultLOOmethod = "NK")
%}
%- maybe also 'usage' for other objects documented here.
%\arguments{
%  \item{X}{
%%     ~~Describe \code{X} here~~
%}
%  \item{Y}{
%%     ~~Describe \code{Y} here~~
%}
%  \item{clusters}{
%%     ~~Describe \code{clusters} here~~
%}
%  \item{indices}{
%%     ~~Describe \code{indices} here~~
%}
%  \item{covType}{
%%     ~~Describe \code{covType} here~~
%}
%  \item{param}{
%%     ~~Describe \code{param} here~~
%}
%  \item{sd2}{
%%     ~~Describe \code{sd2} here~~
%}
%  \item{krigingType}{
%%     ~~Describe \code{krigingType} here~~
%}
%  \item{tagAlgo}{
%%     ~~Describe \code{tagAlgo} here~~
%}
%  \item{numThreadsZones}{
%%     ~~Describe \code{numThreadsZones} here~~
%}
%  \item{numThreads}{
%%     ~~Describe \code{numThreads} here~~
%}
%  \item{verboseLevel}{
%%     ~~Describe \code{verboseLevel} here~~
%}
%  \item{outputLevel}{
%%     ~~Describe \code{outputLevel} here~~
%}
%  \item{globalOptions}{
%%     ~~Describe \code{globalOptions} here~~
%}
%  \item{nugget}{
%%     ~~Describe \code{nugget} here~~
%}
%  \item{defaultLOOmethod}{
%%     ~~Describe \code{defaultLOOmethod} here~~
%}
%}
\details{
Same usage and arguments as \code{\link{looErrors}}. Only the output differs.
%%  ~~ If necessary, more details than the description above ~~
}
\value{
A list containing only one entry:
\item{looError}{The mean square leave-one-out error for the default selected method in argument \code{defaultLOOmethod}.}
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
## see example of looErrors and demo 'demoK'
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
