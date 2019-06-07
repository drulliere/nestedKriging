\name{tests_run}
\alias{tests_run}
\title{Run all Implemented Tests for the \code{nestedKriging} Package
}
\description{
This function runs various tests, including unit tests, comparison tests with other implementations, tests on final results...
}
\usage{
tests_run(showSuccess = FALSE, debugMode=FALSE)
}
\arguments{
  \item{showSuccess}{If \code{TRUE}, show the list of all tests succesfully passed. Otherwise, just show the list of failures and a summary.}
  \item{debugMode}{If \code{TRUE}, runs tests in debugging mode (announce test before launching it, etc.)}
}
%%\details{
%%  ~~ If necessary, more details than the description above ~~
%%}
\value{
returns a list containing following items:
\item{"ok"}{a boolean value indicating if all tests successfully passed.}
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
tests_run(showSuccess = TRUE)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
