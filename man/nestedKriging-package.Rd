\name{nestedKriging-package}
\alias{nestedKriging-package}
\docType{package}
\title{
\packageTitle{nestedKriging}
}
\description{
\packageDescription{nestedKriging}
}
\details{
The DESCRIPTION file:
\packageDESCRIPTION{nestedKriging}
\packageIndices{nestedKriging}
Nested Kriging is an interpolation technique which uses clusters of data in order to deal with larger datasets than usual Kriging techniques.

When a function \code{f} is observed only at some input points, the function \code{\link{nestedKriging}} computes nested Kriging predictions at some given prediction points. Observations are \code{Y=f(X)}, where response vector \code{Y} and design matrix \code{X} are given, and the \code{nestedKriging} function gives a prediction of \code{f(x)}. The function \code{nestedKriging} typically returns mean and variance of the predictor. It requires dividing the \eqn{n} input points into \eqn{N} subgroups. It first calculates Kriging predictor on each subgroups, and then aggregate predictors into a unique predictor. This way, it allows doing Kriging on larger datasets, as soon as \eqn{N} and \eqn{n/N} are not too large.

The package mainly contains the function \code{\link{nestedKriging}}, which gives the prediction mean and variance of a Nested Kriging predictor. For repeated call of this function, it is recommended to use the function \code{\link{nestedKrigingDirect}} which do not check input arguments.

After installation of the package, it is recommended to launch the function \code{\link{tests_run}} which launches tests and checks if multithreading is available on your system.
}
\author{
\packageAuthor{nestedKriging}

Maintainer: \packageMaintainer{nestedKriging}
}

\references{
Rullière, D., Durrande, N., Bachoc, F., Chevalier, C. (2017), Nested Kriging predictions for datasets with a large number of observations, Statistics and Computing, in press. doi: 10.1007/s11222-017-9766-2. Preprint: \url{https://arxiv.org/abs/1607.05432}, paper: \url{https://dx.doi.org/10.1007/s11222-017-9766-2}.
}
\examples{
# first launch some tests of the program
tests_run()

# example 1, a simple example of nestedKriging with small datasets
library(nestedKriging)
set.seed(1)

testFunction <- function(x) { x[2] + (x[1]-5*cos(x[2]))^2 + cos(3*x[1]) }

X <- matrix(runif(1000*2), ncol=2)              # 1000 initial design points, in dimension 2
Y <- apply(X=X, MARGIN = 1, FUN=testFunction)   # initial response for each design points
x <- matrix(runif(100*2), ncol=2)               # 100 prediction points, in dimension 2
clustering <- kmeans(X, centers=20)             # clustering of design points X into 20 groups

prediction <- nestedKriging(X=X, Y=Y, clusters=clustering$cluster, x=x ,
                            covType="matern5_2", param=c(1,1), sd2=10,
                            krigingType="simple", tagAlgo='example 1', numThreads=5)


realvalues <- apply(x, MARGIN = 1, FUN = testFunction) #real values to be predicted
pred_errors <- abs(realvalues - prediction$mean)
mean_error_Nested <- mean(pred_errors)
message("mean error Nested Kriging = ", mean_error_Nested)
}
%%~~ Optionally other standard keywords, one per line, from file KEYWORDS in the R documentation directory ~~
%%\keyword{ package }
%%\seealso{
%%~~ Optional links to other man pages, e.g. ~~
%%~~ \code{\link[<pkg>:<pkg>-package]{<pkg>}} ~~
%%}
\section{Disclaimer}{
This is a preliminary program, which is for illustrative purposes only. It is provided "as is" and "with all faults", without any warranties and support.
%%The authors assume no responsibility or liability for the use of the program. The authors makes no representations or warranties of any kind concerning the safety, suitability, lack of viruses, inaccuracies, typographical errors, or other harmful components of this program. There are inherent dangers in the use of any software, and you are solely responsible for determining whether this program is compatible with your equipment and other software installed on your equipment. You are also solely responsible for the protection of your equipment and backup of your data, and the authors will not be liable for any damages you may suffer in connection with using, modifying, or distributing this program. The authors reserves the right to make changes in the software without notification. The authors make no representation or warranty that such application will be suitable for the specified use without further testing or modification.
}
