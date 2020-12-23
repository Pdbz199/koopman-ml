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
epsilon_t <- t(epsilon)
setnames(epsilon, names(epsilon), t(sub_t[,1]))
# qq plot means left skewness
RVAideMemoire::mqqnorm(epsilon_t, main = "BSM Multi-normal Q-Q plot")
mshapiro.test(epsilon_t) # p-value < 2.2e-16 which means that it is NOT normally distributed

# Normality Tests




# Koopman
epsilons <- fread(file = 'gedmd_epsilons.csv')
epsilons_t <- t(epsilons)
setnames(epsilons, names(epsilons), t(sub_t[,1]))
# qq plot means left skewness
RVAideMemoire::mqqnorm(epsilons_t, main = "Koopman Multi-normal Q-Q plot")
mshapiro.test(epsilons_t)  # p-value < 2.2e-16 which means that it is NOT normally distributed

# Normality Tests
hz2 <- mvn(epsilons_t, mvnTest="hz")
hz2$multivariateNormality

dh2 <- mvn(epsilons_t, mvnTest="dh")
dh2$multivariateNormality


