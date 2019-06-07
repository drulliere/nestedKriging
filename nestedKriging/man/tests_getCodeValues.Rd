\name{tests_getCodeValues}
\alias{tests_getCodeValues}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Returns the Nested Kriging Predicion for a Selected Case Study, for Manual Testing of the Package}
\description{
For a given case study number, the function returns the nested Kriging prediction done by the \code{nestedKriging} function.
The detailed parameters of the case study can be obtained using the function \code{tests_getCaseStudy}.

%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
tests_getCodeValues(selectedCase, covType = "gauss", forceSimpleKriging = FALSE,
                    increaseParam = 0.0)
}
\arguments{
  \item{selectedCase}{chosen case study number.
}
  \item{covType}{the covariance kernel that should be used.
}
  \item{forceSimpleKriging}{if \code{TRUE} force the \code{nestedKriging} to use simple Kriging, otherwise use the value \code{ordinaryKriging} indicated in the case study.
}
  \item{increaseParam}{increase all lengthscale parameters by a given value, eventually negative if lengthscales are large enough (used to test numerical instability for small lenthscales).
}
}
%%\details{
%%  ~~ If necessary, more details than the description above ~~
%%}
\value{
Return the same output as \code{nestedKriging} function, see the help of this function.
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
\examples{
library(nestedKriging)
myResults <- tests_getCodeValues(1, "gauss", forceSimpleKriging = TRUE)
}
