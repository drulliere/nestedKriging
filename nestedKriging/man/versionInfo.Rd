\name{versionInfo}
\alias{versionInfo}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Gives Information about the Version of this Package
}
\description{
Gives information about the version of the program (version, built) and the version of the interface.
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
versionInfo(outputLevel=0L)
}
\arguments{
  \item{outputLevel}{optional integer that set the level of detail of the output information. 0: basic information about the version of the program, 1: more detailed information about environment variables. Default=0.
}
}

%- maybe also 'usage' for other objects documented here.
%%\details{
%%  ~~ If necessary, more details than the description above ~~
%%}
\value{
return a list which contains:
    \item{versionCode}{The version of the code.}
    \item{built}{The built number (to distinguish variations among a given version).}
    \item{interfaceVersion}{The interface version number, which changes only when the internal interface of \code{\link{nestedKriging}} C++ function changes.}
    \item{interfacesDescription}{The description of \code{\link{nestedKriging}} C++ function interface.}
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
library(nestedKriging)
myinformations <- versionInfo()
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%%\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
%%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
