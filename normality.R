library(data.table)
library(RVAideMemoire)
library(MVN)

setwd('Python Koopman/')

# BSM
t <- readRDS('coindataBalancedWide.RDS')
sub_t <- t[datetime > quantile(datetime, 0.9)]
filtered <- sub_t[,2:length(sub_t)]
cov_mat <- cov(filtered)
mat_mean <- colMeans(filtered)
sigma <- chol.default(cov_mat)
epsilon <- as.data.frame(solve(t(sigma) %*% sigma) %*% sigma %*% t(filtered - mat_mean))
setnames(epsilon, names(epsilon), t(sub_t[,1]))
# qq plot means left skewness
RVAideMemoire::mqqnorm(t(epsilon), main = "BSM Multi-normal Q-Q plot")
mshapiro.test(t(epsilon)) # p-value < 2.2e-16 which means that it is NOT normally distributed





# Koopman
epsilons <- fread(file = 'gedmd_epsilons.csv')
setnames(epsilons, names(epsilons), t(sub_t[,1]))
# qq plot means left skewness
RVAideMemoire::mqqnorm(t(epsilons), main = "Koopman Multi-normal Q-Q plot")
mshapiro.test(t(epsilons))  # p-value < 2.2e-16 which means that it is NOT normally distributed



result <- mvn(t(epsilons), mvnTest="hz")