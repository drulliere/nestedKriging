\name{outputLevel}
\alias{outputLevel}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Get outputLevel parameter value as a function of the desired output.
%%  ~~function to do ... ~~
}
\description{
Get outputLevel parameter value as a function of the desired output. The user set boolean values of what must be computed, and the function returns an integer summarizing these needs, to be used in the \link{nestedKriging} function.
}
\usage{
outputLevel(nestedKrigingPredictions = FALSE, alternatives = FALSE, predictionBySubmodel = FALSE, covariancesBySubmodel = FALSE, covariances = FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{nestedKrigingPredictions}{
Boolean.  Shall we compute nested Kriging mean and variance predictions?
}
  \item{alternatives}{
Boolean.  Shall we compute predictions by alternatives techniques (PoE, gPoE, BCM, rBCM, SPV...)?
}
  \item{predictionBySubmodel}{
Boolean.  Shall we give in output all predictions by submodel (i.e. by cluster)?
}
  \item{covariancesBySubmodel}{
Boolean.  Shall we give in output covariances between submodels? caution! this may be large.
}
  \item{covariances}{
Boolean.  Shall we compute and give in output covariances between prediction points? caution! this may be large and more expensive to compute.
}
}
\details{
This function gives a simple way to fill the argument \code{outputLevel} in the function \link{nestedKriging}.

CAUTION! always ask for the minimal needed output, in order to avoid very large outputs (huge matrices) or supplementary computations. If you are uncertain, just use the default value of \code{outputLevel} argument in the function \link{nestedKriging}.
}
\value{
a relative integer summarizing all that the user wants to compute. Notice that all arguments set to FALSE leads to the value 0, which still computes nestedKriging basic predictions.
}
%\note{
%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\examples{

desiredOutput <- outputLevel(nestedKrigingPredictions=TRUE, alternatives=TRUE)
# this argument can be used in nestedKriging procedure, in outputLevel argument
# if one wants both nestedKriging predictions and alternatives ones.

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
