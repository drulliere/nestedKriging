
library(DiceKriging)

# ---------------------------------
# a 1D example with only one group
# ---------------------------------

# observations
X <- c(0, 0.4, 0.6, 0.8, 1)   # design points
Y <- c(-0.3, 0, 0, 0.5, 0.9)  # corresponding observed responses

# new data = new points where we want to predict the response
xmin <- -0.5; xmax <- 2.5
x <- seq(from=xmin, to=xmax, by=0.005)

# model

trendX = cbind(X, X^2, 1)
trendx = cbind(x, x^2, 1)
formula <- Y~1+X+I(X^2)

param <- 1
sd2 <- 0.01

# results using DiceKriging

model <- km(formula=formula, design=data.frame(X=X), response=data.frame(Y=Y), 
            covtype="matern5_2", coef.cov = param, coef.var = sd2)

# Results with Universal Kriging formulae (mean and and 95% intervals)
p.UK <- predict(model, newdata=data.frame(X=x), type="UK")

############# Results using nestedKriging

n <- nrow(as.matrix(X))
q <- nrow(as.matrix(x))
cluster <- rep(1, times=n)

library(nestedKriging)


details <- outputLevel(nestedKrigingPredictions = TRUE, predictionBySubmodel = TRUE)
resu <- nestedKriging(X=as.matrix(X), Y=Y, x=as.matrix(x), covType="matern5_2", krigingType="UKOK", param=param, sd2=sd2, 
                      clusters = cluster, outputLevel=details, verboseLevel = 0, trendX=trendX, trendx=trendx)



maxDifferenceMean <- max(abs(resu$mean - p.UK$mean))
maxDifferenceVar <- max(abs(resu$sd2 - p.UK$sd^2))
maxDifference <- max(maxDifferenceMean, maxDifferenceVar)

message("DiceKriging and NestedKriging give the same prediction: ", maxDifference<1e-8)


# universal Kriging coefficients
beta_UK <- resu$beta_UK
trendUK <- trendx %*% beta_UK # trend on new data points
maxDiffTrend <- maxDifferenceTrend <- max(abs(trendUK - p.UK$trend))
message("DiceKriging and NestedKriging give the same trend: ", maxDiffTrend<1e-8)


################################### plots

color <- list(SK="grey", UK="orange", trendUK = "red")

upperSpace <- 1
plot(x, p.UK$mean, type="l", ylim=c(min(p.UK$lower95),max(p.UK$upper95))+c(0,upperSpace),
     xlab="x", ylab="y")
#lines(x, p.UK$trend, col="violet", lty=2)
lines(x, p.UK$lower95, col=color$UK, lty=2)
lines(x, p.UK$upper95, col=color$UK, lty=2)
#polygon(x = c(x, rev(x)), y = c(p.UK$lower95, rev(p.UK$upper95)),lty = 0, col = "#fff7ec")
lines(x, resu$mean, col=color$UK, lty=1, lwd=2)
lines(x, trendUK, col=color$trendUK, lty=4)
points(X, Y, col=color$trendUK, pch=19)
abline(h=0)

# Results with Simple Kriging (SK) formula. The difference between the width of
# SK and UK intervals are due to the estimation error of the trend parameters 
# (but not to the range parameters, not taken into account in the UK formulae).

model.SK <- km(formula=~1, coef.trend = 0, design=data.frame(X=X), response=data.frame(Y=Y), 
            covtype="matern5_2", coef.cov = param, coef.var = sd2)

p.SK <- predict(model.SK, newdata=data.frame(X=x), type="SK")
lines(x, p.SK$mean, type="l", ylim=c(-7,7), xlab="x", ylab="y", col=color$SK, lty=1, lwd=2)
lines(x, p.SK$lower95, col=color$SK, lty=2)
lines(x, p.SK$upper95, col=color$SK, lty=2)
points(X, Y, col="red", pch=19)
abline(h=0)

legend.text <- c("Universal Kriging (UK)", "Simple Kriging (SK)")
legend(x=xmin, y=max(p.UK$upper)+upperSpace, legend=legend.text, 
       text.col=c(color$UK, color$SK), col=c(color$UK, color$SK), 
       lty=2, bg="white")


############# Results using nestedKriging, using several clusters

library(nestedKriging)
# new data = new points where we want to predict the response

xmin <- -2.5; xmax <- 2.5
x <- seq(from=xmin, to=xmax, by=0.005)

X <- runif(n=20,xmin*0.8,xmax*0.7)   # design points
X <- sort(X) # ordered to ease clustering and plot reading

Y <- 0.5*sin(5*X)+3*sin(X)+(X)^2  # corresponding observed responses

n <- nrow(as.matrix(X))
q <- nrow(as.matrix(x))

## Clusters

N=2 # number of clusters
#set.seed(seed = 13)
#cluster <- as.integer(N*runif(n=n, 0, 1))+1 
# here ordered clusters to ease plot reading

cluster <- 1 + as.integer(seq(from=1,to=n)/((n+1)/N))

# Covariance model

covType = "UKOK" # first Layer universal, second Layer Ordinary

param <- 0.4
sd2 <- 0.01

trendX = cbind(X, X^2, 1)
trendx = cbind(x, x^2, 1)
formula <- Y~1+X+I(X^2)

# Prediction

details <- outputLevel(nestedKrigingPredictions = TRUE, predictionBySubmodel = TRUE)
resuTwoClusters <- nestedKriging(X=as.matrix(X), Y=Y, x=as.matrix(x), covType="matern5_2", krigingType="UKOK", param=param, sd2=sd2, 
                      clusters = cluster, outputLevel=details, verboseLevel = 0, trendX=trendX, trendx=trendx)


SumWeightsMinusOne <- max(abs(c(1,1) %*% resuTwoClusters$weights -1 ))
message("Second Layer ordinary => Weights are summing to one: ", SumWeightsMinusOne<1e-8)



trendByCluster <- trendx %*% resuTwoClusters$beta_UK # trend on new data points
weigthedTrendByCluster <- trendByCluster * t(resuTwoClusters$weights)
globalTrend <- weigthedTrendByCluster %*% rep(1,times=N)

colorCluster <- list("red", "blue")

plot(x, resuTwoClusters$mean, type="l", ylim=range(resuTwoClusters$mean), 
     xlab="x", ylab="y")
points(X[cluster==1], Y[cluster==1], col=colorCluster[[1]], pch=19)
points(X[cluster==2], Y[cluster==2], col=colorCluster[[2]], pch=19)
lines(x, trendByCluster[,1], col=colorCluster[[1]], lty=2)
lines(x, trendByCluster[,2], col=colorCluster[[2]], lty=2)
lines(x, globalTrend, col="orange", lty=1, lwd=2)
#lines(x, resuTwoClusters$weights[1,], col="grey", lty=1)
abline(h=0)
