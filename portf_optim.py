#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import cplex


# In[2]:


def strat_buy_and_hold(x_init, cash_init, mu, Q, cur_prices):
    x_optimal = x_init
    cash_optimal = cash_init
    return x_optimal, cash_optimal


# In[3]:


def strat_equally_weighted(x_init, cash_init, mu, Q, cur_prices):
    total_value = np.dot(cur_prices,x_init)+cash_init
    weight_optimal = np.array([1/len(x_init)]*len(x_init)) # Equal weight = 1/n
    allocated_value = weight_optimal*total_value
    
    x_optimal = np.floor(allocated_value/cur_prices) # Rounding procedure
    transaction_fee = 0.005 * np.dot(cur_prices,abs(x_optimal-x_init))
    cash_optimal = total_value - np.dot(cur_prices,x_optimal) - transaction_fee
    
    return x_optimal, cash_optimal


# In[4]:


def strat_min_variance(x_init, cash_init, mu, Q, cur_prices):
    total_value = np.dot(cur_prices,x_init)+cash_init
    
    n = len(x_init)
    
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize) # Minimize objective
    c  = [0.0] * n # No linear objective
    lb = [0.0] * n
    ub = [1.0] * n
    
    Atilde = []
    for k in range(n):
        Atilde.append([[0,1],[1,0]]) # One column of ones times corresponding weights
    
    var_names = ["w_%s" % i for i in range(1,n+1)] 
    cpx.linear_constraints.add(rhs=[1.0,0], senses="EE")                  
    cpx.variables.add(obj=c, lb=lb, ub=ub, columns=Atilde, names=var_names)
    
    qmat = [[list(range(n)), list(2*Q[k,:])] for k in range(n)] # Sparse matrix of Q
    cpx.objective.set_quadratic(qmat) # Quadratic objective
    cpx.parameters.threads.set(6)
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.solve()
    weight_optimal = np.array(cpx.solution.get_values())
    
    allocated_value = weight_optimal*total_value
    
    x_optimal = np.floor(allocated_value/cur_prices) #Rounding procedure
    transaction_fee = 0.005 * np.dot(cur_prices,abs(x_optimal-x_init))
    cash_optimal = total_value - np.dot(cur_prices,x_optimal) - transaction_fee
    
    return x_optimal, cash_optimal


# In[5]:


def strat_max_Sharpe(x_init, cash_init, mu, Q, cur_prices):
    total_value = np.dot(cur_prices,x_init)+cash_init
    
    n = len(x_init)+1
    r_rf = 0.025
    daily_rf = r_rf/252
    diff = mu - daily_rf
    
    coe_k = np.zeros((20,1))
    Q = np.hstack((Q,coe_k))
    coe_k = np.zeros((1,21))
    Q = np.vstack((Q,coe_k)) # New row and column for risk free asset
    
    Atilde = []
    for k in range(20):
        Atilde.append([[0,1],[diff[k],1]])
    Atilde.append([[0,1],[0,-1]]) 
    
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    c  = [0.0] * n
    lb = [0.0] * n
    ub = [np.inf] * n 
    
    var_names = ['y_%s'% i for i in range(1,n+1)]
    cpx.linear_constraints.add(rhs=[1.0,0],senses='EE')
    cpx.variables.add(obj=c,lb=lb,ub=ub,columns=Atilde,names=var_names)

    qmat = [[list(range(n)), list(2*Q[k,:])] for k in range(n)]
    cpx.objective.set_quadratic(qmat) # Quadratic objective
    cpx.parameters.threads.set(6)
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.solve()
    
    w_maxSharpe = np.array(cpx.solution.get_values()) # Optimal weight                        
    weight = w_maxSharpe[0:20]/w_maxSharpe[20]
    
    allocated_value = weight*total_value
    
    x_optimal = np.floor(allocated_value/cur_prices) # Rounding procedure
    transaction_fee = 0.005 * np.dot(cur_prices,abs(x_optimal-x_init))
    cash_optimal = total_value - np.dot(cur_prices,x_optimal) - transaction_fee
    
    return x_optimal, cash_optimal


# In[6]:


# Input file
input_file_prices = 'Daily_closing_prices.csv'
# Read data into a dataframe
df = pd.read_csv(input_file_prices)
# Convert dates into array [year month day]
def convert_date_to_array(datestr):
    temp = [int(x) for x in datestr.split('/')]
    return [temp[-1], temp[0], temp[1]]

dates_array = np.array(list(df['Date'].apply(convert_date_to_array)))
data_prices = df.iloc[:, 1:].to_numpy()
dates = np.array(df['Date'])
# Find the number of trading days in Nov-Dec 2019 and
# compute expected return and covariance matrix for period 1
day_ind_start0 = 0
day_ind_end0 = len(np.where(dates_array[:,0]==2019)[0])
cur_returns0 = data_prices[day_ind_start0+1:day_ind_end0,:] / data_prices[day_ind_start0:day_ind_end0-1,:] - 1
mu = np.mean(cur_returns0, axis = 0)
Q = np.cov(cur_returns0.T)
# Remove datapoints for year 2019
data_prices = data_prices[day_ind_end0:,:]
dates_array = dates_array[day_ind_end0:,:]
dates = dates[day_ind_end0:]
# Initial positions in the portfolio
init_positions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 902, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17500])

# Initial value of the portfolio
init_value = np.dot(data_prices[0,:], init_positions)
print('\nInitial portfolio value = $ {}\n'.format(round(init_value, 2)))


# In[7]:


# Initial portfolio weights
w_init = (data_prices[0,:] * init_positions) / init_value

# Number of periods, assets, trading days
N_periods = 6*len(np.unique(dates_array[:,0])) # 6 periods per year
N = len(df.columns)-1
N_days = len(dates)

# Annual risk-free rate for years 2020-2021 is 2.5%
r_rf = 0.025
# Number of strategies
strategy_functions = ['strat_buy_and_hold', 'strat_equally_weighted', 'strat_min_variance', 'strat_max_Sharpe']
strategy_names     = ['Buy and Hold', 'Equally Weighted Portfolio', 'Minimum Variance Portfolio', 'Maximum Sharpe Ratio Portfolio']
#N_strat = 1  # comment this in your code
N_strat = len(strategy_functions)  # uncomment this in your code
fh_array = [strat_buy_and_hold, strat_equally_weighted, strat_min_variance, strat_max_Sharpe]
portf_value = [0] * N_strat
x = np.zeros((N_strat, N_periods),  dtype=np.ndarray)
cash = np.zeros((N_strat, N_periods),  dtype=np.ndarray)


# In[8]:


for period in range(1, N_periods+1):
    # Compute current year and month, first and last day of the period
    if dates_array[0, 0] == 20:
        cur_year  = 20 + math.floor(period/7)
    else:
        cur_year  = 2020 + math.floor(period/7)

    cur_month = 2*((period-1)%6) + 1
    day_ind_start = min([i for i, val in enumerate((dates_array[:,0] == cur_year) & (dates_array[:,1] == cur_month)) if val])
    day_ind_end = max([i for i, val in enumerate((dates_array[:,0] == cur_year) & (dates_array[:,1] == cur_month+1)) if val])
    print('\nPeriod {0}: start date {1}, end date {2}'.format(period, dates[day_ind_start], dates[day_ind_end]))
   
    # Prices for the current day
    cur_prices = data_prices[day_ind_start,:]

    # Execute portfolio selection strategies
    for strategy in range(N_strat):
        # Get current portfolio positions
        if period == 1:
            curr_positions = init_positions
            curr_cash = 0
            portf_value[strategy] = np.zeros((N_days, 1))
        else:
            curr_positions = x[strategy, period-2]
            curr_cash = cash[strategy, period-2]

      # Compute strategy
        x[strategy, period-1], cash[strategy, period-1]= fh_array[strategy](curr_positions, curr_cash, mu, Q, cur_prices)

      # Verify that strategy is feasible (you have enough budget to re-balance portfolio)
      # Check that cash account is >= 0
      # Check that we can buy new portfolio subject to transaction costs

      ###################### Insert your code here ############################
        # Validation Procedure
        if cash[strategy, period-1] < 0:
            # Adjust the portfolio with the ratio of each stock
            cur_portfolio_value = np.dot(cur_prices,curr_positions) + curr_cash
            ratio = x[strategy, period-1]/sum(x[strategy, period-1])
            excess_cash = abs(cash[strategy, period-1])*ratio
            excess_position = np.ceil(excess_cash/cur_prices) # The units of stocks need to sell
            x[strategy, period-1] = x[strategy, period-1] - excess_position # New optimal protfolio
            new_tran_fee = np.dot(cur_prices , abs(x[strategy, period-1]-curr_positions)) * 0.005
            # New cash account value
            cash[strategy, period-1] = cur_portfolio_value - np.dot(cur_prices,x[strategy, period-1]) - new_tran_fee

      # Compute portfolio value
        p_values = np.dot(data_prices[day_ind_start:day_ind_end+1,:], x[strategy, period-1]) + cash[strategy, period-1]
        portf_value[strategy][day_ind_start:day_ind_end+1] = np.reshape(p_values, (p_values.size,1))
        print('  Strategy "{0}", value begin = $ {1:.2f}, value end = $ {2:.2f}'.format( strategy_names[strategy], 
             portf_value[strategy][day_ind_start][0], portf_value[strategy][day_ind_end][0]))

      
    # Compute expected returns and covariances for the next period
    cur_returns = data_prices[day_ind_start+1:day_ind_end+1,:] / data_prices[day_ind_start:day_ind_end,:] - 1
    mu = np.mean(cur_returns, axis = 0)
    Q = np.cov(cur_returns.T)


# In[9]:


# Plot results
###################### Insert your code here ############################
# Daily value of 4 portfolio strategy
plt.figure(figsize=(16,12))
plt.plot(portf_value[0],label='Buy and Hold')
plt.plot(portf_value[1],label='Equally Weighted Portfolio')
plt.plot(portf_value[2],label='Minimum Variance Portfolio')
plt.plot(portf_value[3],label='Maximum Sharpe Ratio Portfolio')
plt.legend()
plt.title('Figure 1: Daily Portfolio Values for Each Strategy')
plt.xlabel('Time')
plt.ylabel('Portfolio Values')
plt.savefig('Daily Portfolio Values for Each Strategy.png')
plt.show()


# In[10]:


# Dynamic change of minimum variance
assets = list(df.columns.values)
assets = assets[1:]

trading_period = np.arange(0,13,1)

w_init = (data_prices[0,:]*init_positions)/init_value

x[2, period-1], cash[2, period-1]= fh_array[2](curr_positions, curr_cash, mu, Q, cur_prices)

asset_weight =[]
asset_weight.append(w_init)

# Calculate weight of each protfolio
# Because of the rounding procedure, the weight might not be exactly 1, but very close to 1
for period in range(1, N_periods+1):
    if dates_array[0, 0] == 20:
        cur_year  = 20 + math.floor(period/7)
    else:
        cur_year  = 2020 + math.floor(period/7)

    cur_month = 2*((period-1)%6) + 1
    day_ind_start = min([i for i, val in enumerate((dates_array[:,0] == cur_year) & (dates_array[:,1] == cur_month)) if val])
    day_ind_end = max([i for i, val in enumerate((dates_array[:,0] == cur_year) & (dates_array[:,1] == cur_month+1)) if val])

    # Prices for the current day
    cur_prices = data_prices[day_ind_start,:]
    
    total_value = np.dot(cur_prices,x[2, period-1])+cash[2, period-1]
    asset_weight.append((cur_prices*x[2, period-1])/total_value)
    
df_minvar = pd.DataFrame(asset_weight,columns = assets)

df_minvar.plot(figsize=(16,12),cmap="tab20")
plt.title('Figure 2: Dynamic Change with Minimum Variance Portfolio Strategy')
plt.xlabel('Periods')
plt.ylabel('Weights of Portfolio')
plt.legend(loc='upper left')
plt.savefig('Dynamic Change Minimum Variance.png')
plt.show()


# In[11]:


# Dynamic change of maximum Sharpe ratio
x[3, period-1], cash[3, period-1] = fh_array[3](curr_positions, curr_cash, mu, Q, cur_prices)

asset_weight_3 =[]
asset_weight_3.append(w_init)
# Calculate weight of each protfolio
# Because of the rounding procedure, the weight might not be exactly 1, but very close to 1
for period in range(1, N_periods+1):
    if dates_array[0, 0] == 20:
        cur_year  = 20 + math.floor(period/7)
    else:
        cur_year  = 2020 + math.floor(period/7)

    cur_month = 2*((period-1)%6) + 1
    day_ind_start = min([i for i, val in enumerate((dates_array[:,0] == cur_year) & (dates_array[:,1] == cur_month)) if val])
    day_ind_end = max([i for i, val in enumerate((dates_array[:,0] == cur_year) & (dates_array[:,1] == cur_month+1)) if val])

    cur_prices = data_prices[day_ind_start,:]
    
    total_value = np.dot(cur_prices,x[3, period-1])+cash[3, period-1]
    asset_weight_3.append((cur_prices*x[3, period-1])/total_value)

df_maxsharpe = pd.DataFrame(asset_weight_3,columns = assets,index=trading_period)

df_maxsharpe.plot(figsize=(16,12),cmap="tab20")
plt.title('Figure 3: Dynamic Change with Maximun Sharpe Ratio Portfolio Strategy')
plt.xlabel('Periods')
plt.ylabel('Weights of Portfolio')
plt.legend(loc='upper left')
plt.savefig('Dynamic Change Maximun Sharpe.png')
plt.show()


# In[12]:


x[2, :]

