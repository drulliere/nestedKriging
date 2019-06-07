\name{tests_getCaseStudy}
\alias{tests_getCaseStudy}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Returns a Case Study Example for Manual Testing of the Package
}
\description{
The function returns a list containing in particular most input arguments of the \code{\link{nestedKriging}} function, in particular design points \code{X}, observed responses \code{Y}, prediction points \code{x}, clusters \code{clusters}, etc.
It can be used in conjunction with \code{tests_getCodeValues} which returns the \code{nestedKriging} predicted values for a specified case study.
}
\usage{
tests_getCaseStudy(selectedCase, covType)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{selectedCase}{Case study number: a chosen number would always return the same case study (design, responses, etc.).
}
  \item{covType}{specifies the covariance family of the case study. (see \code{covType} argument of \code{nestedKriging} function)
}
}
%%\details{
%%  ~~ If necessary, more details than the description above ~~
%%}
\value{
Returns a list containing
\item{seed}{seed that is used, equal to the case study number.}
\item{q}{number of prediction points.}
\item{d}{dimension of each design point.}
\item{n}{number of initial design points.}
\item{N}{number of clusters.}
\item{...}{\code{X}, \code{x}, \code{Y}, etc. main input parameters of the \code{nestedKriging} function (see the help of this function).}
\item{gpsize}{vector containing the size of each cluster. This is a \eqn{N \times 1}{N x 1} vector.}
\item{pickx}{number of the prediction point that will be used for the prediction (belongs to \eqn{0...n-1})}

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
caseStudyOne <- tests_getCaseStudy(1, "gauss")
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
