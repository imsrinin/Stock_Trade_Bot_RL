#assumptions aorund whih the implementation would be done to reduce the action space comlexity
#we ignore the transaction costs
#we sell all shares and not partially!
#when buying, we try to buy as many as possible
#when buying multiple stocks, we buy 1 stock each in round robin fashion
#sell before buy to maximize the funds to buy
#states: shares owned, prices of shares that we own and, cash
#rewards: change in th value of portfolio (just like robinhood visalization of total value of the account)

#imlementation  choices
#using SGD with mommentum
#target_q = Q(s,a) for all action except for the action taken for loss function {target_q = r + gamma*max(Q(s,a))}

#dataset: apple, motorola and, starbucks stock prices from feb 2013 to feb 2018

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import argparse
import pickle
import itertools
from datetime import datetime

def get_data():
    df = pd.read_csv('/Users/srini/Desktop/indep_study/RL_Udemy/machine_learning_examples-master/tf2.0/aapl_msi_sbux.csv') #rows = stcoks, cols = values of stock over time
    return df.values 

def Buffer(env):
    states =[]
    # choose action randomly from action space and normalize the states
    for _ in range(env.n_step):
        a = np.random.choice(env.action_space) 
        s,r,done,info=env.step(a)
        states.append(s)
        if done:
            break
    scaled = StandardScaler()
    scaled.fit(states)
    return scaled 


class Model(object):
    def __init__(self,ip_dims,n_actions):
        #random weights initilization and normalization for better convergence while learning 
        self.W = (1/np.sqrt(ip_dims))*np.random.randn(ip_dims,n_actions) 
        self.b = np.random.randn(n_actions)
        self.mW = 0
        self.mb = 0
        self.loss = []
        
    def predict(self,X):
        return np.dot(X,self.W) + self.b
    
    def grad(self,X,Y,lr=0.01,m=0.9):
        # grad of W = (2/N*K)* sum of (Y-Ypred)*X
        Yhat = self.predict(X)
        gW = (2/np.prod(Y.shape))*np.dot(X.T,(Y-Yhat))
        # grad of b = (2/N*K) * sum of (Y-Ypred)
        gb = (2/np.prod(Y.shape))*(Y-Yhat).sum(axis=0)
        #updating momentum terms
        self.mW = m*self.mb - lr*gW
        self.mb = m*self.mb - lr*gb
        #updating weights
        self.W += self.mW
        self.b += self.mb
        
        mse = np.mean((Y-Yhat)**2)
        self.loss.append(mse)
        
    def save_model(self,path):
        np.savez(path,W=self.W,b=self.b)
    
    def load_model(self,path):
        loaded = np.load(path)
        self.W = loaded['W']
        self.b = loaded['b']

        
class StockEnv(object):
    #states --> [num_stock1,num_stock2,num_stock3,
    #            price_stock1,price_stock2,price_stock3
    #            cash_owned]
    #actions --> 3^3 combinations of individual 3 actions for all 3 stocks 
    #action map {sell:0,buy:1,hold:2}
    
    def __init__(self,data,investment=20000):
        self.investment = investment
        self.stock_hist = data
        self.n_step, self.n_stock = self.stock_hist.shape
        
        self.curr_step = None
        self.stock_owned = None
        self.stock_price = None
        self.fund_left = None
        
        self.action_space = np.arange(3**self.n_stock)
        #creating action space
        self.action_list = list(map(list, itertools.product([0,1,2], repeat =self.n_stock)))
        #creates actions as an array with each element indicating the action for that stock
        #[action_stock1,action_stock2,action_stock3] --> {sell:0,buy:1,hold:2}
        self.state_dims = 2*n_stock + 1
        self.reset()
        
    def reset(self):
        self.curr_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_hist[self.curr_step]
        self.funds_left = self.investment
        return self.get_obs()
    
    def step(self,action):
        #store the old value for {total amount left in the account}
        old_tot = self.get_tot()
        
        #move to next day
        self.curr_step +=1
        self.stock_price = self.stock_hist[self.curr_step]
        #check if we went through all the previous data of stock history
        done = self.curr_step ==  self.n_step -1
        
        #perform trade from what we learnt till now
        self.trade(action)
        #update {total amount left in the account}
        curr_tot = self.get_tot()
        #calculate profits 
        r = curr_tot - old_tot
        #track portfolio
        info = {'curr_tot': curr_tot}
        #trying to mimic the openAI gym functions and hence formulated retun this way
        return self.get_obs(), r, done, info
    
    def get_obs(self):
        #creating the state object
        obs = np.empty(self.state_dims)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:2*self.n_stock] = self.stock_price
        obs[-1] = self.funds_left
        return obs
    
    def get_tot(self):
        return np.dot(self.stock_owned,self.stock_price) + self.funds_left
    
    def trade(self,action):
        action_vect = self.action_list[action]
        sell_idx = []
        buy_idx = []
        
        #loop through the action vector to update but and sell indicies
        for i,a in enumerate(action_vect):
            if a == 0:
                sell_idx.append(i)
            if a ==1:
                buy_idx.append(i)
        #selling all the stocks if the sell decision was made        
        if sell_idx:
            for i in sell_idx:
                self.funds_left +=self.stock_price[i]*self.stock_owned[i]
                self.stock_owned[i] = 0
        #buying the stocks in roundrobin if the decision was made to but multiple stocks
        if buy_idx:
            can_buy= True
            while can_buy:
                for i in buy_idx:
                    if self.funds_left >= self.stock_price[i]:
                        self.funds_left -= self.stock_price[i]
                        self.stock_owned[i] += 1
                    else:
                        can_buy = False
        
        
class Agent(object):
    def __init__(self,state_dims,action_dims):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.gamma = 0.9
        self.alpha = 0.1
        self.ep = 1
        self.ep_min = 0.01
        self.ep_decay = 0.99
        self.model = Model(state_dims,action_dims)
        
    def act(self,state):
        if np.random.rand() < self.ep:
            return np.random.choice(self.action_dims)
        else:
            return np.argmax(self.model.predict(state)[0])
    
    def train(self,state,action,reward,state_,done):
        target = (reward + self.gamma*np.amax(self.model.predict(state),axis=1)) if done else reward
        
        target_all = self.model.predict(state)
        target_all[0,action] = target
        
        #SGD to train
        self.model.grad(state,target_all)
        
        #minimizing exploration over time
        if self.ep > self.ep_min:
            self.ep *= self.ep_decay
        
    def load(self,name):
            self.model.load_model(name)
        
    def save(self,name):
            self.model.save_model(name)
            

def play_ep(agent,env,is_train):
    s = env.reset()
    scaled = StandardScaler()
    s = scaled.fit_transform([s])
    done = False
    while not done:
        a = agent.act(s)
        s_,r,done,info = env.step(a)
        s_ = scaled.fit_transform([s_])
        if is_train:
            agent.train(s,a,r,s_,done)
        s = s_
    return info['curr_tot']


if __name__ == '__main__'          :
    
    #config variables
    models_folder = 'linear_rl_trader_models'
    rewards_folder = 'linear_rl_trader_rewards'
    n_ep = 10000
    investment = 20000
    
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '-mode', type=bool, required = True,
    #                    help = 'True if training')
    # args = parser.parse_args()
    
    train = True
    
    data = get_data()
    n_step,n_stock = data.shape
    
    n_train = n_step//2
    
    train_data = data[:n_train]
    test_data = data[n_train:]
    
    env = StockEnv(train_data,investment)
    state_dims = env.state_dims
    action_dims = len(env.action_space)
    agent = Agent(state_dims,action_dims)
    buff = Buffer(env)
    
    #keep track of portfolio
    portfolio = []
    if train == False: # testing
        with open(f'{models_folder}/buff.pkl','rb') as f:
            buff = pickle.load(f)
            
        env = StockEnv(test_data,investment)
        
        agent.ep = 0.01
        
        agent.load(f'{models_folder}/trade_bot.npz')
        
    for e in range(n_ep):
        t = datetime.now()
        tot = play_ep(agent,env,train)
        dt = datetime.now() - t
        print(f'episode: {e+1}/{n_ep}, episode end value: {tot:.2f}, duration: {dt}')
        portfolio.append(tot)
        
    #saving model 
    if train:
        agent.save(f'{models_folder}/trade_bot.npz')
        
        with open(f'{models_folder}/buff.pkl', 'wb') as f:
            pickle.dump(buff,f)
        
        plt.plot(agent.model.loss)
        plt.show()
