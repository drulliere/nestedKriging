\name{setNumThreadsBLAS}
\alias{setNumThreadsBLAS}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Set the Number of Threads Used by External Linear Algebra Libraries (BLAS)
}
\description{Set the number of threads used by external linear algebra libraries (BLAS). When BLAS uses more than one thread by default, it uses threads less efficiently than parallel algorithm \code{\link{nestedKriging}}, so that the recommanded setting is \code{numThreadsBLAS=1}. However, it may be useful to change this setting in very specific cases: number of subgroups lower than the number of cores, other BLAS uses... This threads number is adjusted using external \code{R} package \code{RhpcBLASctl}.
}
\usage{
setNumThreadsBLAS(numThreadsBLAS = 1, showMessage = TRUE)
}
\arguments{
  \item{numThreadsBLAS}{The number of threads that should be affected to the external linear algebra library BLAS. Default=\code{1}.
}
  \item{showMessage}{If \code{TRUE}, shows a confirmation message for the action.
}
}
%%\details{
%%  ~~ If necessary, more details than the description above ~~
%%}
\value{ no value returned.
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
setNumThreadsBLAS(1)
}
