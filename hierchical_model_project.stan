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
  
  real mu0_alpha;
  real<lower=0> sigma0_alpha;
  real mu0_beta;
  real<lower=0> sigma0_beta;
  
  real alpha1;
  vector[K] beta1;
  
  real alpha2;
  vector[K] beta2;
  
  real alpha3;
  vector[K] beta3;
  
  real<lower=0> sigma;
}

model {
  // default priors 
  if (prior == 1) {
    mu0_alpha ~ normal(150000,50000);
    sigma0_alpha ~ normal(60000,20000);
    mu0_beta ~ normal(0, 5000);
    sigma0_beta ~ normal(10000, 3000);
    
    alpha1 ~ normal(mu0_alpha,sigma0_alpha);
    beta1 ~ normal(mu0_beta, sigma0_beta);
    
    alpha2 ~ normal(mu0_alpha,sigma0_alpha);
    beta2 ~ normal(mu0_beta,sigma0_beta);
  
    sigma ~ normal(0, 20000);
  }
     // change sigma0
  else if (prior == 2) { 
    mu0_alpha ~ normal(150000,30000);
    sigma0_alpha ~ normal(60000,2000);
    mu0_beta ~ normal(0, 2);
    sigma0_beta ~ inv_chi_square(0.05);
    
    alpha1 ~ normal(mu0_alpha,sigma0_alpha);
    beta1 ~ normal(mu0_beta, sigma0_beta);
    
    alpha2 ~ normal(mu0_alpha,sigma0_alpha);
    beta2 ~ normal(mu0_beta,sigma0_beta);
  
    sigma ~ normal(0, 20000);
    
  }
  // change sigma0
  else if (prior == 3) { 
    mu0_alpha ~ normal(150000,30000);
    sigma0_alpha ~ normal(60000,100000);
    mu0_beta ~ normal(0, 2);
    sigma0_beta ~ inv_chi_square(10);
    
    alpha1 ~ normal(mu0_alpha,sigma0_alpha);
    beta1 ~ normal(mu0_beta, sigma0_beta);
    
    alpha2 ~ normal(mu0_alpha,sigma0_alpha);
    beta2 ~ normal(mu0_beta,sigma0_beta);
  
    sigma ~ normal(0, 20000);
  }
  // change mu0
  else if (prior == 4) { 
    mu0_alpha ~ normal(0,100);
    sigma0_alpha ~ normal(0,100);
    mu0_beta ~ normal(0, 20);
    sigma0_beta ~ inv_chi_square(1);
    
    alpha1 ~ normal(mu0_alpha,sigma0_alpha);
    beta1 ~ normal(mu0_beta, sigma0_beta);
    
    alpha2 ~ normal(mu0_alpha,sigma0_alpha);
    beta2 ~ normal(mu0_beta,sigma0_beta);
    
    sigma ~ normal(0, 20000);
  }
  // change mu0
  else if (prior == 5) { 
    mu0_alpha ~ normal(200,500);
    sigma0_alpha ~ normal(0,100);
    mu0_beta ~ normal(20, 50);
    sigma0_beta ~ inv_chi_square(1);
    
    alpha1 ~ normal(mu0_alpha,sigma0_alpha);
    beta1 ~ normal(mu0_beta,sigma0_beta);
    
    alpha2 ~ normal(mu0_alpha,sigma0_alpha);
    beta2 ~ normal(mu0_beta,sigma0_beta);
  
    sigma ~ normal(0, 20000);
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

