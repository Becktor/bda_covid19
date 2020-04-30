#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import countryinfo as ci
import pystan
import matplotlib.pyplot as plt
import mplcyberpunk
import pickle
plt.style.use("cyberpunk")

countries = ["Denmark", "Finland", "Norway", "Sweden"] # contries analyzed 
print(ci.CountryInfo("Denmark").population())
data = pd.read_csv("data/time_series_covid_19_confirmed.csv")
data = data.loc[data['Country/Region'].isin(countries)]
data = data[2:].reset_index()
data = data.drop(columns=['index', 'Province/State', 'Lat', 'Long'])

# Normalize data by population
data_normalized = data.apply(lambda row: pd.to_numeric(row[1:]) / ci.CountryInfo(row[0]).population(), axis=1)
data_normalized.insert(0, "Country", data.head()["Country/Region"]) 
data_normalized.set_index('Country', inplace=True)
print(data_normalized.head())

plt.figure(figsize=(20,10))
for c in countries:
    plt.plot(np.arange(data_normalized.shape[1]),data_normalized.loc[c], label = c)
    print(c)
    
plt.legend(fontsize=20)
mplcyberpunk.add_glow_effects()
plt.show()


# In[ ]:


if False:
    with open("model_fit.pkl", "rb") as f:
        data_dict = pickle.load(f)
    fit = data_dict['fit']


# In[13]:


pooled_model = """

functions { // define logistic growth model
  real[] logisticgrowth(real t, real[] y,real[] theta,real[] x_r,int[] x_i) {
    real dydt[x_i[1]];
    
    for (i in 1:x_i[1]){
        dydt[i] = theta[1] * y[i] * (1-y[i]/theta[2]);
    }
    
    return dydt;
  }
}
data {
  int<lower=0> T; // number of time steps (days)
  int<lower=0> n_countries; // number of countries
  real y0[n_countries]; // inital condition for ode solver
  real z[T,n_countries]; // percentage of population infected 
  real t0; // inital time step in days (0)
  real ts[T]; // time progression in days
}
transformed data {
  real x_r[0]; //  data values used to evaluate the ODE system
  int x_i[1]; //  integer data values used to evaluate the ODE system.
  x_i[1] = n_countries;
}
parameters {
  real theta[2]; // parameter for growth model
  real<lower=0> sigma; 
}
model {
    // define model using uniform uninformative priors 
  real y_hat[T,n_countries];

// solve ode - find mean percentage infected for given timestep
  y_hat = integrate_ode_rk45(logisticgrowth, y0, t0, ts, theta, x_r, x_i); 
  
  for (t in 1:T) {
    for (i in 1:n_countries) {
      z[t,i] ~ normal(y_hat[t,i], sigma);
    }
  }
}
generated quantities{
// generate predictions for comparison between data and prediction
  real y_pred[T,n_countries];
  real z_pred[T,n_countries];
  y_pred = integrate_ode_rk45(logisticgrowth, y0, t0, ts, theta, x_r, x_i );
  for (t in 1:T) {
    for(i in 1:n_countries){
      z_pred[t,i] = y_pred[t,i] + normal_rng(0,sigma);
    }
  }
}

"""


# In[ ]:


T = data_normalized.shape[1]
n_countries = data_normalized.shape[0]
z = data_normalized.values.T
y0 = np.zeros(data_normalized.shape[0])

stan_data = {'T': T, 'n_countries': n_countries, 'z': z,
             'y0': y0, 't0': 0, 'ts': np.arange(1,T+1)}

sm = pystan.StanModel(model_code=pooled_model)
fit_pooled = sm.sampling(data=stan_data, iter=4000, chains=4, control=dict(max_treedepth=15))


# In[17]:


with open("model_fit.pkl", "wb") as f:
    pickle.dump({'model' : model, 'fit' : fit}, f, protocol=-1)

