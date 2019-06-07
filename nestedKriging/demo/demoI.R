
##################################################################################
#                                                                                #
#  demoI.R : demo with tiny data in 1D with alternative aggregation techniques   #
#                                                                                #
##################################################################################

# the purpose here is to check if predicted covariances lead to reasonable sample paths
# the data is kept very small so that one can visualize the different sample paths
# in practice, the method can be used for large n and q not too high

library(nestedKriging)
set.seed(13)

f <- function(t) { sin(5*t)+cos(10*t)/2 } # test function f

d <- 1          # dimension, to ease graphic representation t is a real and d=1
n <- 6          # number of observations, very small here
q <- 200        # number of prediction points, quite high for nice graphics paths
nugget <- 0.0   # nugget added to the diagonal of X covariance matrix

lengthscale <- 0.4 # model lengthscales, higher gives tighter paths
variance <- 1      # model variance, higher gives more widespread paths

X <- matrix(runif(n*d), ncol=d)       # initial design points, in dimension d
Y <- apply(X=X, MARGIN = 1, FUN=f)    # initial response for each design points
x <- matrix(runif(q*d)*2-0.8, ncol=d) # prediction points, in dimension d
N <- floor(sqrt(n))                   # number of clusters, sqrt(n) gives fast results
clustering <- kmeans(X, centers=N)    # clustering of design points X into 2 groups

prediction <- nestedKriging(X=X, Y=Y, clusters=clustering$cluster, x=x ,
                            covType="matern3_2", param=rep(lengthscale,d), sd2=variance, outputLevel = 2, verboseLevel=0,
                            krigingType="simple", tagAlgo='demo I', numThreads=4, nugget=nugget)

mu <- prediction$mean                         # mean of the predictor, vector of size q
sd2 <- prediction$ sd2                        # variance of the predictor, vector of size q
realvalues <- apply(x, MARGIN = 1, FUN = f)   # real values to be predicted
message("mean error Nested Kriging = ", mean(abs(realvalues - mu))) #average error of the prediction

#use outputLevel = -1 to get Alternative predictors (PoE, GPoE, BCM, RBCM)
predictAlt <- nestedKriging(X=X, Y=Y, clusters=clustering$cluster, x=x ,
                            covType="matern3_2", param=rep(lengthscale,d), sd2=variance, outputLevel = -1, verboseLevel=0,
                            krigingType="simple", tagAlgo='demo I alternatives', numThreads=4, nugget=nugget)

dfpoints <- data.frame(x=X[,1], y=Y, z=clustering$cluster)
dfexpected <- data.frame(x=x[,1], y=realvalues)

dfNested <- data.frame(x=x[,1], y=prediction$mean,  var=prediction$sd2)

Alt <- predictAlt$Alternatives
dfPOE  <- data.frame(x=x[,1], y=Alt$meanPOE,  var=Alt$sd2POE)
dfGPOE <- data.frame(x=x[,1], y=Alt$meanGPOE, var=Alt$sd2GPOE)
dfGPOE_1N <- data.frame(x=x[,1], y=Alt$meanGPOE_1N, var=Alt$sd2GPOE_1N)
dfBCM  <- data.frame(x=x[,1], y=Alt$meanBCM,  var=Alt$sd2BCM)
dfRBCM <- data.frame(x=x[,1], y=Alt$meanRBCM, var=Alt$sd2RBCM)
dfSPV <- data.frame(x=x[,1], y=Alt$meanSPV, var=Alt$sd2SPV)

dfModel1 <- data.frame(x=x[,1], y=prediction$mean_M[1,], var=prediction$sd2_M[1,])
dfModel2 <- data.frame(x=x[,1], y=prediction$mean_M[2,], var=prediction$sd2_M[2,])

ggplot2_loaded <- require(ggplot2)
if (ggplot2_loaded) {

drawPlot <- function(dfChosen, name, title) {
linesize <- 1.2
myplot <- ggplot(dfChosen, aes(x)) +
  geom_ribbon(data= dfChosen, aes(x=x, ymin=y-1.96*sqrt(var),ymax=y+1.96*sqrt(var)), alpha=0.1, fill="blue") +
  geom_line(aes(y=y), colour="blue", size=linesize) +
  geom_point(data = dfpoints, mapping = aes(x = x, y = y, fill=factor(z)), size=5, shape = 21, colour = "black", stroke = 2) +
  labs(title = title, subtitle = paste("using the ", name, " predictor")) +
  geom_line(data = dfexpected, aes(x=x, y=y), size=1, colour="black", linetype = "dashed") +
  labs(caption = paste("using", N, "clusters and", n, "observations, dashed line = underlying function")) +
  labs(x = "x") +
  labs(y = "y") +
  labs(fill = "cluster")

print(myplot)
}

drawPlot(dfModel1, "Model 1", "Kriging predictions")
drawPlot(dfModel2, "Model 2", "Kriging predictions")

drawPlot(dfPOE, "POE", "Alternative aggregation technique")
drawPlot(dfGPOE, "GPOE", "Alternative aggregation technique")
drawPlot(dfGPOE_1N, "GPOE_1N", "Alternative aggregation technique")
drawPlot(dfBCM, "BCM",  "Alternative aggregation technique")
drawPlot(dfRBCM, "RBCM", "Alternative aggregation technique")
drawPlot(dfSPV, "SPV", "Alternative aggregation technique")

drawPlot(dfNested, "Nested Kriging", "Mixing Kriging predictions")

}
