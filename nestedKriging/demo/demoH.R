
######################################################################
#                                                                    #
#  demoH.R : demo with tiny data in 1D using predicted covariances   #
#                                                                    #
######################################################################

# the purpose here is to check if predicted covariances lead to reasonable sample paths
# the data is kept very small so that one can visualize the different sample paths
# in practice, the method can be used for large n and q not too high

library(nestedKriging)
set.seed(13)

f <- function(t) { sin(5*t)+cos(10*t)/2 } # test function f

d <- 1       # dimension, to ease graphic representation t is a real and d=1
n <- 6       # number of observations, very small here
q <- 100     # number of prediction points, quite high for nice graphics paths

lengthscale <- 0.4 # model lengthscales, higher gives tighter paths
variance <- 1      # model variance, higher gives more widespread paths

X <- matrix(runif(n*d), ncol=d)      # initial design points, in dimension d
Y <- apply(X=X, MARGIN = 1, FUN=f)   # initial response for each design points
#x <- matrix(runif(q*d), ncol=d)      # prediction points, in dimension d
x <- matrix(seq(0,1,length.out=q), ncol=1)
N <- floor(sqrt(n))                  # number of clusters, sqrt(n) gives fast results
clustering <- kmeans(X, centers=N)   # clustering of design points X into 2 groups

# notice that outputLevel >= 10 is required to get predicted covariances

desiredOutput = outputLevel(nestedKrigingPredictions = TRUE, covariances = TRUE)
# in that case, computation time and storage needed are increased
prediction <- nestedKriging(X=X, Y=Y, clusters=clustering$cluster, x=x ,
                            covType="matern3_2", param=rep(lengthscale,d), sd2=variance, outputLevel = desiredOutput, verboseLevel=0,
                            krigingType="simple", tagAlgo='example 1', numThreads=4)

mu <- prediction$mean                         # mean of the predictor, vector of size q
sd2 <- prediction$sd2                         # variance of the predictor, vector of size q
sigma <- prediction$cov                       # covariance of the predictor, q x q matrix. (COVARIANCES)
realvalues <- apply(x, MARGIN = 1, FUN = f)   # real values to be predicted
message("mean error Nested Kriging = ", mean(abs(realvalues - mu))) #average error of the prediction

# Generate sample paths from the conditional process, using conditional cross-covariances sigma
MASS_loaded <- require(MASS) # for multivariate normal sampling
ggplot2_loaded <- require(ggplot2)
packages_loaded <- MASS_loaded && ggplot2_loaded

if (!packages_loaded) {

    message("the packages MASS and ggplot2 must be installed for this demo")

} else {

  n.samples <- 7
  #generate n.samples sample paths having the same size as mu
  trajec <- MASS::mvrnorm(n=n.samples, mu=mu, Sigma=sigma) #sample paths of multivariate normal, uses (COVARIANCES)

  # Collect data for graphic
  dftrajec <- data.frame(x=rep(x[,1],n.samples), y=matrix(t(trajec), byrow = TRUE, ncol=1), label=rep(1:n.samples, each=length(mu)))
  dfexpected <- data.frame(x=x[,1], y=realvalues)
  dfpoints <- data.frame(x=X[,1], y=Y, z=clustering$cluster)
  dfkrig <- data.frame(x=x[,1], y=mu, lwr= mu+1.96*sqrt(sd2), upr=mu-1.96*sqrt(sd2))

# Plot sample paths, observation points and realValues
linesize <- 1.2

xkcd_loaded <- FALSE
#xkcd_loaded <- require(xkcd) # uncomment to get xkcd style, if fonts are installed
myplot <- ggplot(dftrajec, aes(x)) +
  geom_line(data= dftrajec, aes(x=x, y=y, group=label, colour=as.factor(label)),  size=linesize, show.legend = FALSE) +
  geom_ribbon(data= dfkrig, aes(x=x, ymin=lwr,ymax=upr), alpha=0.1, fill="blue") +
  geom_line(data = dfexpected, aes(x=x, y=y), size=1, colour="black", linetype = "dashed") +
  scale_fill_brewer(palette = "Set1") +
  geom_point(data = dfpoints, mapping = aes(x = x, y = y, fill=factor(z)), size=5, shape = 21, colour = "black", stroke = 2) +
  labs(title = "Conditional sample paths", subtitle = "based on the nested Kriging predictor") +
  labs(caption = paste("using", N, "clusters and", n, "observations, dashed line = underlying function")) +
  labs(x = "x") +
  labs(y = "y") +
  labs(fill = "cluster")

  if (xkcd_loaded) {
      xrange <- range(x)
      yrange <- range(trajec)
      myplot <- myplot +   xkcdaxis(xrange, yrange, size=2) + theme_xkcd()
  } else {
    myplot <- myplot +   theme_gray()
  }
  print(myplot)
  message("done, see plots.")

}

