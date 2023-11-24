
data {
  int<lower=0> J; //number of groups
  int<lower=0> N1; //number of observations for group 1
  int<lower=0> N2; //number of observations for group 2
  int<lower=1> K; //number of features

  matrix[N1, K] x1;
  matrix[N2, K] x2;

  vector[N1] y1;
  vector[N2] y2;

  int prior;
}


parameters {
  real alpha1;
  vector[K] beta1;

  real alpha2;
  vector[K] beta2;

  real<lower=0> sigma;
}

model {
  // default priors
  if (prior == 1) {
    alpha1 ~ normal(110000, 50000);
    beta1 ~ normal(0, 1);

    alpha2 ~ normal(110000, 50000);
    beta2 ~ normal(0, 1);


    sigma ~ normal(0, 10);
  }
  // wide priors
  else if (prior == 2) {
    alpha1 ~ normal(110000, 100000);
    beta1 ~ normal(0, 10);

    alpha2 ~ normal(110000, 100000);
    beta2 ~ normal(0, 10);

    sigma ~ normal(0, 100);

  }
  // narrow priors
  else if (prior == 3) {
    alpha1 ~ normal(110000, 10000);
    beta1 ~ normal(0, 0.5);

    alpha2 ~ normal(110000, 10000);
    beta2 ~ normal(0, 0.5);

    sigma ~ normal(0, 0.5);
  }


  //likelihood
  y1 ~ normal(alpha1 + x1*beta1, sigma);
  y2 ~ normal(alpha2 + x2*beta2, sigma);
}

generated quantities {
  //posterior predictive distribution for posterior predictive check
  real y1_rep[N1] = normal_rng(alpha1 + x1*beta1, sigma);
  real y2_rep[N2] = normal_rng(alpha2 + x2*beta2, sigma);

  //log-likelihood
  vector[N1+N2] log_lik;

  for (i in 1:N1) {
    log_lik[i] = normal_lpdf(y1[i] | x1[i] * beta1 + alpha1, sigma);
  }

  for (i in 1:N2) {
    log_lik[i+N1] = normal_lpdf(y2[i] | x2[i] * beta2 + alpha2, sigma);
  }

}
