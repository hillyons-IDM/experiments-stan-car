/* This is stan code for a particular experiment of a random walk.
 * The rw is improper as defined: no constraint is respected.
 * As such it will move "location" (think mean) to wherever the data resides,
 * i.e. a constrained RW with an improper flat prior on the constraint.
 * If there is no data this model doesn't make any sense.
 */

functions{
  real mult_normal_car_sparseQ_lpdf (vector y, real tau, vector w, int[] u, int[] v, real r, real log_detQ_unscaled){
    int n;
    real out;
    n = rows(y);
    out = 0.5 * (r * log(2 * pi()) + (r * log(tau)) + log_detQ_unscaled) - 0.5 * tau * dot_product(y, csr_matrix_times_vector(n,n,w,v,u,y));
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
}
parameters{
  real<lower=0> sig_rw; // std dev scale
  real<lower=0> sig_noise; // std dev scale
  vector[n] rw;
}
model{
  //assume default priors for sigma terms but do not really need to
  rw~mult_normal_car_sparseQ(1 / (pow(sig_rw,2)), w, u, v, r, log_detQ_unscaled);
  y~normal(rw, sig_noise);
}

