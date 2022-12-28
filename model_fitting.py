#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:21:06 2021

@author: celiaberon
"""


# import jax.numpy as jnp
# from jax import jit, lax, value_and_grad
from tqdm.auto import trange
import numpy as np
import scipy as sci
import scipy.optimize as scio
# import torch

    
def log_probability_rflr(parameters, sessions):
    
    # compute probability of next choice
    ll = 0.0
    n = 0
    for choices, rewards in sessions:        
        # initialize "belief state" for this session
        
        ll += _log_prob_single_rflr(parameters, choices, rewards)
        n += len(choices) - 1
            
    return ll / n

def likfun_torch(x,data):
   
    #  parameters
    alpha_pos = x[0]        # learning rate
    alpha_neg = x[1]        # learning rate
    C = np.int0(max(np.unique(data['c']))+1) # number of options
    v = np.zeros((C,))  # initial values
    V = torch.zeros((data['N'],))
    rpe = np.zeros((data['N'],))
    for n in range(data['N']):
        c = np.int0(data['c'][n])
        r = data['r'][n]          
        rpe[n] = r - v[c]   # reward prediction error
        v[c] = v[c] + alpha_pos*rpe[n]*(rpe[n]>0) + alpha_neg*rpe[n]*(rpe[n]<0)    # update values
        V[n]= v[c]
    
    [lik,yhat] = regress_torch(V,data['v'])
    print(lik)
    return lik

def regress_torch(X, Y):

    #remove missing data points for the regression(for DA data mostly)
    new_col = torch.ones((X.shape[0],1))
    X2 = torch.from_numpy(np.append(new_col,X[...,np.newaxis], 1)).float() # add intercept
    Y2 = torch.unsqueeze(Y,1).float()
    #  solves the equation a @ x = b
    aa = torch.mm(X2.T,X2)
    bb = torch.mm(X2.T,Y2)
    b = torch.linalg.solve(aa,bb)

    Yhat = torch.mm(X2,b) 
    sdy = torch.sqrt(torch.nanmean((Y-Yhat)**2))
    pdf_ = torch.from_numpy(sci.stats.norm.pdf(Y,Yhat,sdy))
    lik = torch.sum(torch.log(pdf_))
    
    return lik, Yhat

def regress_py(X, Y):
    '''
    Linear regression.
    
    USAGE: [lik,Yhat] = myregress(X,Y)
    
    INPUTS:
      X - design matrix
      Y - outcome vector
    
    OUTPUTS:
      lik - log-likelihood
      Yhat - predicted outcomes

    Modified by Sandra Romero Pinto 2022, to python
    '''
    
    #remove missing data points for the regression(for DA data mostly)

    new_col = np.ones((X.shape[0],1))
    X2 = np.append(new_col,X[...,np.newaxis], 1) # add intercept

    b = np.linalg.lstsq((np.dot(X2.T,X2)),(np.dot(X2.T,Y)),rcond=None)[0]
    # b = (X2.T*X2)\(X2.T*Y2);

    Yhat = np.dot(X2,b) 
    sdy = np.sqrt(np.nanmean((Y-Yhat)**2))
    lik = np.sum(np.log(sci.stats.norm.pdf(Y,Yhat,sdy)))
    
    return lik, Yhat

def likfun_full(x,data,model):
    '''
    Compute log-likelihood for a single subject. 
    Licking is modeled as a linear function of value at the time of cue.
    
    USAGE: [lik,latents] = likfun(x,data,model)
    
    INPUTS:
      x - parameter vector (see below for interpretation of parameters)
      data - data structure (see fit_models.m)
      model - which model to fit
    
    OUTPUTS:
      lik - log-likelihood of the data
      latents - structure with the following fields:
          .V - [N x 1] reward expectation at time of cue
          .rpe - [N x 1] reward prediction error at time of outcome
          .lick - [N x 1] predicted lick signal
    
    Sam Gershman & Benedicte Babayan 2017
    Modified by Sandra Romero Pinto  September 2022
    '''

    #  parameters
    if model == 'symmetric':
        alpha_ = x[0]
        beta_ = x[1]
    elif model == 'asymmetric':
        alpha_pos = x[0]        # learning rate
        alpha_neg = x[1]        # learning rate
        beta_ = x[2]
    elif model == 'reward_sensitivity':
        alpha_ = x[0]
        rew_sens = x[1]
        beta_ = x[2]

    C = np.int0(max(np.unique(data['c']))+1) # number of options
    
    if 'v0' in data.keys():
        v =  data['v0']
    else:
        v = np.zeros((C,))  # initial values

    V = np.zeros((data['N'],))
    rpe = np.zeros((data['N'],))
    for n in range(data['N']):
        c = np.int0(data['c'][n])
        r = data['r'][n]          
        V[n]= beta_*v[c]
        if model == 'symmetric':
            rpe[n] = r - v[c]   # reward prediction error
            v[c] = v[c] + alpha_*rpe[n]    # update values
        elif model == 'asymmetric':
            rpe[n] = r - v[c]   # reward prediction error
            v[c] = v[c] + alpha_pos*rpe[n]*(rpe[n]>0) + alpha_neg*rpe[n]*(rpe[n]<0)    # update values
        elif model == 'reward_sensitivity':
            rpe[n] = rew_sens*r - v[c]   # reward prediction error
            v[c] = v[c] + alpha_*rpe[n]    # update values
            
        

    [lik,yhat] = regress_py(V,data['v'])

    return lik, yhat

def likfun(x,data,model):
    '''
    Compute log-likelihood for a single subject. 
    Licking is modeled as a linear function of value at the time of cue.
    
    USAGE: [lik,latents] = likfun(x,data,model)
    
    INPUTS:
      x - parameter vector (see below for interpretation of parameters)
      data - data structure (see fit_models.m)
      model - which model to fit
    
    OUTPUTS:
      lik - log-likelihood of the data
      latents - structure with the following fields:
          .V - [N x 1] reward expectation at time of cue
          .rpe - [N x 1] reward prediction error at time of outcome
          .lick - [N x 1] predicted lick signal
    
    Sam Gershman & Benedicte Babayan 2017
    Modified by Sandra Romero Pinto  September 2022
    '''

    #  parameters
    if model == 'symmetric':
        alpha_ = x[0]
        beta_ = x[1]
        # v0 = x[1]
    elif model == 'asymmetric':
        alpha_pos = x[0]        # learning rate
        alpha_neg = x[1]        # learning rate
        beta_ = x[2]
        # v0 = x[2]
    elif model == 'reward_sensitivity':
        alpha_ = x[0]
        rew_sens = x[1]
        beta_ = x[2]
        # v0 = x[2]

    C = np.int0(max(np.unique(data['c']))+1) # number of options
    # v = v0*np.ones((C,))
    if 'v0' in data.keys():
        v =  data['v0']
    else:
        v = np.zeros((C,))  # initial values

    V = np.zeros((data['N'],))
    rpe = np.zeros((data['N'],))
    # beta_=1
    for n in range(data['N']):
        c = np.int0(data['c'][n])
        r = data['r'][n]          
        V[n]= beta_*v[c]
        if model == 'symmetric':
            rpe[n] = r - v[c]   # reward prediction error
            v[c] = (v[c] + alpha_*rpe[n])    # update values
        elif model == 'asymmetric':
            rpe[n] = r - v[c]   # reward prediction error
            v[c] = (v[c] + alpha_pos*rpe[n]*(rpe[n]>0) + alpha_neg*rpe[n]*(rpe[n]<0))    # update values
        elif model == 'reward_sensitivity':
            rpe[n] = rew_sens*r - v[c]   # reward prediction error
            v[c] = (v[c] + alpha_*rpe[n])    # update values
            
        

    [lik,yhat] = regress_py(V,data['v'])

    return lik

def rlfit_predict(data,results,model):
    '''
      Compute log predictive probability of new data.
     
      USAGE: logp = mfit_predict(data,results)
     
      INPUTS:
        data - [S x 1] data structure
        results - results structure
     
      OUTPUTS:
        logp - [S x 1] log predictive probabilities for each subject
     
      Sam Gershman, June 2015
    '''
    
    # logp = results.likfun(results.x(s,:),data(s));
    logp = results['likfun'](results['x'],data,model)

    return logp
#% %

def rlfit_logposterior(x,param,data,likfun):
    '''
    Evaluate log probability of parameters under the (unnormalized) posterior.
    
    USAGE: logp = rlfit_logposterior(x,param,data,likfun)
    
    INPUTS:
      x - parameter values
      param - parameter structure
      data - data structure
      likfun - function handle for likelihood function
    
    OUTPUTS:
      logp - log unnormalized posterior probability
    
    Sam Gershman, July 2015
    Modified by Sandra Romero Pinto, September 2022
    '''
    model = param['model']
    logp = likfun(x,data,model)
    
    for k in range(len(param['logpdf'])):
        logp = logp + param['logpdf'][k](x[k])

    return logp

def rlfit_optimize(_llfun, param, data_, nstarts=4):
    
    '''
    Find maximum a posteriori parameter estimates.
    
    USAGE: results = mfit_optimize(likfun,param,data,[nstarts])
    
    INPUTS:
      likfun - likelihood function handle
      param - [K x 1] parameter structure
      data - [S x 1] data structure
      nstarts (optional) - number of random starts (default: 5)
    
    OUTPUTS:
      results - structure with the following fields:
                  .x - [S x K] parameter estimates
                  .logpost - [S x 1] log posterior
                  .loglik - [S x 1] log likelihood
                  .bic - [S x 1] Bayesian information criterion
                  .aic - [S x 1] Akaike information criterion
                  .H - [S x 1] cell array of Hessian matrices
                  .latents - latent variables (only if likfun returns a second argument)
    
    Sam Gershman, July 2017
    Modified by Sandra Romero Pinto, September 2022
    '''
    results = dict()
    K = len(param['ub'])
    results['K'] = K
    results['param'] = param
    results['likfun'] = _llfun
    lb = param['lb']
    ub = param['ub']
    model = param['model']


    f_ = lambda x: - rlfit_logposterior(x,param,data_,_llfun)
    for i in range(nstarts):
        x0 = np.zeros((K,))
        for k in range(K):
            x0[k] = (ub[k]-lb[k]) * np.random.random_sample() + lb[k]
        bounds = scio.Bounds(lb, ub)
        res = scio.minimize(f_, x0, bounds=bounds,method='L-BFGS-B')
        logp = - np.asarray(res.fun)
        x_fitted = np.asarray(res.x)
        if i == 0 or results['logpost'] < logp:
            results['logpost'] = logp
            results['loglik'] = _llfun(x_fitted,data_,model)
            results['x'] = x_fitted
            f_ = lambda x: - rlfit_logposterior(x,param,data_,_llfun)
            
    results['bic']= K*np.log(data_['N']) - 2*results['loglik']
    results['aic'] = K*2-3*results['loglik']

    return results

def rlsim_pav(x,R,P,N,model):
    
    '''
    USAGE: data = rlsim(x,R,N)
    
    INPUTS:
      x - parameter vector:
          x(1) - learning rate for pos
          x(2) - learning rate for neg
    
      R - [1 x n_cues] reward probabilities for each cue 
      P - [1 x n_cues] probability of each cue 
      N - number of trials
    
    OUTPUTS:
      data - structure with the following fields
              .r - [N x 1] rewards
              .v - [N x 1] value predictions 
              .c - [N x 1] cues  
    '''

    n_cues = len(P)
    v = np.zeros((n_cues,1))   # initial values
    data = dict()
    data['N'] = N 

    if model == 'symmetric':
        lr = x[0]
        # map_lick = x[1]
    elif model == 'asymmetric':
        lr_p = x[0]
        lr_n = x[1]
        # map_lick = x[2]
    elif model == 'reward_sensitivity':
        lr = x[0]
        r_sens = x[1]
        # map_lick = x[3]
    else :
        print('Model not within the options, choose: symmetric , asymmetric, reward_sensitivity ')
        raise NotImplementedError

    data['c'] = np.zeros((N,))
    data['r'] = np.zeros((N,))
    data['v'] = np.zeros((N,))
    data['trial'] = np.zeros((N,))
    for n in np.arange(N):
        c = np.argwhere(np.random.multinomial(1, P, size=1))[0][1] # random cue based on probs  
        r = np.double(np.random.rand()<R[c])            # reward feedback
        if model== 'symmetric':
            v[c] = v[c] + lr*(r-v[c]) 
        elif model== 'asymmetric':
            v[c] = v[c] + lr_p*(r-v[c])*((r-v[c])>0) + lr_n*(r-v[c])*((r-v[c])<0)          # update values
        elif model== 'reward_sensitivity':
            v[c] = v[c] + lr*(r*r_sens-v[c]) 

        data['c'][n] = c 
        data['r'][n] = r 
        data['v'][n] = v[c]
        
        data['trial'][n] = n 


    return data 

def create_param_opt(model):

    param = dict()

    if model == 'symmetric':
        param['name'] = ['alpha_','beta_']#,'v0']
        param['logpdf'] = [lambda x: 0 ,lambda x: 0 ]  # uniform prior
        param['lb'] = [0.001,0.1] # lower bound
        param['ub'] = [1,10] # upper bound
        param['model'] = model
        
    elif model == 'asymmetric':
        param['name'] = ['alpha_pos','alpha_neg','beta_']#,'v0']
        param['logpdf'] = [lambda x: 0 ,lambda x: 0,lambda x: 0  ]  # uniform prior
        param['lb'] = [0.001, 0.001,0.1] # lower bound
        param['ub'] = [1, 1, 10] # upper bound
        param['model'] = model
    
    elif model == 'reward_sensitivity':
        param['name'] = ['alpha_','rew_sens','beta_']#,'v0']
        param['logpdf'] = [lambda x: 0 ,lambda x: 0  ,lambda x: 0 ]  # uniform prior
        param['lb'] = [0.001, 0.001 ,0.1] # lower bound
        param['ub'] = [1, 10 ,10] # upper bound
        param['model'] = model
    
    return param