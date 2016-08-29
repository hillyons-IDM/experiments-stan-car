## testing rw methods in STAN

# use Matrix and stan
library(Matrix)
library(rstan)

# generating some fake data
n = 100
mu = 5
sig_rw = 1
sig_noise = 5

# Q is the improper precision matrix
Q = matrix(0, nrow = n, ncol = n)
Q[cbind(2:n, 1:(n-1))] = -1
Q = pmin(Q, t(Q))
diag(Q) = -rowSums(Q)

# sparse Q
Q = Matrix(Q)

# generate rw increments
rw = cumsum(rnorm(n-1, 0, sig_rw))
rw = mu + c(0, rw) - sum(rw)/n

# generate data with noise
y = rnorm(rep(1,n), rw, sig_noise)

# plot random walk and rw + noise
matplot(cbind(rw, y), type = "l", lty = 1)

# extract_sparse_parts comes from the rstan package
stan.parts = extract_sparse_parts(Q)
str(stan.parts)

# data for stan with and without a constraint on the null space

stan.data.noconstraint = list(n = n, y = y, sparse_parts_size = as.numeric(sapply(stan.parts, length)),
                 w = stan.parts$w, v = stan.parts$v, u = stan.parts$u,
                 r = n-1, log_detQ_unscaled = sum(log(eigen(Q)$values[-n])))

stan.data.withconstraint = list(n = n, y = y, sparse_parts_size = as.numeric(sapply(stan.parts, length)),
                              w = stan.parts$w, v = stan.parts$v, u = stan.parts$u,
                              r = n-1, log_detQ_unscaled = sum(log(eigen(Q)$values[-n])), constraint_sd = 0.5)


# three experiments
#  1. Improper rw -- implicit flat prior on mean
#  2. Proper rw induced by penalty on mean
#  3. Proper rw with a hard constraint on mean = 0 + flat prior on intercept (should be same as 1)

out1 = stan(file = "stan/rw_improper_with_data_new.stan", data = stan.data.noconstraint)
out2 = stan(file = "stan/rw_soft_constraint_with_data_new.stan", data = stan.data.withconstraint)
out3 = stan(file = "stan/rw_hard_constraint_with_data_new.stan", data = stan.data.withconstraint)

out1.summary = summary(out1)[[1]]
out2.summary = summary(out2)[[1]]
out3.summary = summary(out3)[[1]]

# plots
windows(height = 12, width = 6)
par(mfrow=c(3,1))
matplot(cbind(rw, y), type = "l", lty = 1, main = "fitted with improper")
polygon(c(1:100, 100:1), c(out1.summary[3:102,4],rev(out1.summary[3:102,8])),
        col = rgb(0,0,1,0.1), border = rgb(0,0,1,0.1))
lines(out1.summary[3:102,1], col = "blue")

matplot(cbind(rw, y), type = "l", lty = 1, main = "fitted with soft constraint")
polygon(c(1:100, 100:1), c(out2.summary[3:102,4],rev(out2.summary[3:102,8])),
        col = rgb(0,0,1,0.1), border = rgb(0,0,1,0.1))
lines(out2.summary[3:102,1], col = "blue")

matplot(cbind(rw, y), type = "l", lty = 1, main = "fitted with hard constraint, flat intercept")
polygon(c(1:100, 100:1), c(out3.summary[grepl("fitted",rownames(out3.summary)),4],rev(out3.summary[grepl("fitted",rownames(out3.summary)),8])),
        col = rgb(0,0,1,0.1), border = rgb(0,0,1,0.1))
lines(out3.summary[grepl("fitted",rownames(out3.summary)),1], col = "blue")

windows()
pairs(cbind(out1.summary[3:102,1],out2.summary[3:102,1],out3.summary[grepl("fitted",rownames(out3.summary)),1]),
      panel = function(x,y) {points(x,y); abline(c(0,1),col="red", lwd=2)},
      main = "Comparison of posterior means for:\nno (1), soft (2), and hard constraint (3)")
