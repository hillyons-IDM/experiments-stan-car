/* This is stan code for a particular experiment of a random walk.
 * In this case a penality is added to vectors in the null space,
 * but this has limited effect.  The rw is fixed to mean zero by 
 * and a separate intercept with a flat prior is added.  This should be
 * indisinguishable from an improper rw that fits the data with an
 * implict flat prior on the mean.
 */

functions{
  real mult_normal_car_sparseQ_log (vector y, real tau, vector w, int[] u, int[] v, real r, real log_detQ_unscaled){
    int n;
    real out;
    n <- rows(y);
    out <- 0.5 * (r * log(2 * pi()) + (r * log(tau)) + log_detQ_unscaled) - 0.5 * tau * dot_product(y, csr_matrix_times_vector(n,n,w,v,u,y));
    return out;
  }
}
data{
  int n;
  real y[n];
  int sparse_parts_size[3];
  vector[sparse_parts_size[1]] w;
  int v[sparse_parts_size[2]];
  int u[sparse_parts_size[3]];
  real r;
  real log_detQ_unscaled;
  real<lower=0> constraint_sd;
}
parameters{
  real<lower=0> sig_rw; // std dev scale
  real<lower=0> sig_noise; // std dev scale
  vector[n] rw;
  real beta;
}
transformed parameters{
vector[n] rw_meanzero;
vector[n] fitted_mean;
real mu;
mu<-mean(rw);
rw_meanzero<-rw-mu;
fitted_mean<-beta+rw_meanzero;
}
model{
  //assume default priors for sigma terms but do not really need to
  rw~mult_normal_car_sparseQ(1 / (pow(sig_rw,2)), w, u, v, r, log_detQ_unscaled);
  y~normal(fitted_mean, sig_noise);
  increment_log_prob(normal_log(mu, 0, constraint_sd));
}
