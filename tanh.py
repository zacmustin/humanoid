# Evolution Strategies BipedalWalker-v2
# Code first taken from 404akhan's BipedalWalker-v2 solution posted here: https://gym.openai.com/evaluations/eval_QjN2JTCQRFu7rNgm0iGy4A/
# gives good solution at around iter 100 in 5 minutes
# for testing model set reload=True

import gym
import numpy as np
import _pickle as pickle
import pprint
import sys


import matplotlib.pyplot as plt #matplotlib is for graphing

env = gym.make('BipedalWalker-v2')
np.random.seed(10)

hl_size = 100 #hidden layer size
version = 1
npop = 100 # number of episodes 
sigma = 0.1
alpha = 0.03
iter_num = 300
aver_reward = None
allow_writing = True
reload = False # reload = True basically loads a pre-made (and pretty good) model - it's supposed to be kind of a demo
iterations = 1000

#graphing set up
areward = []



print(hl_size, version, npop, sigma, alpha, iter_num)

if reload:
    model = pickle.load(open('model-pedal%d.p' % version, 'rb')) # loads pre-made model
else: # creates new, random model
    model = {}
    #np.random.randn fills with random samples from standardized normal distribution
    model['W1'] = np.random.randn(24, hl_size) / np.sqrt(24) # input-hiddenlayer ... 24 x hl_size
    model['W2'] = np.random.randn(hl_size, 4) / np.sqrt(hl_size) # hiddenlayer-output

def get_action(state, model):
    #print(state)
    hl = np.matmul(state, model['W1'])
    hl = np.tanh(hl) # hyperbolic tan -- super high corrections when far from 1, at about 0.9 corrections are miniscule
    action = np.matmul(hl, model['W2'])
    action = np.tanh(action)
    #env.render()
    return action


def f(model, render=False):
    state = env.reset() #resets environment
    total_reward = 0 #resets reward
    for t in range(iter_num):
        if render: env.render()

        action = get_action(state, model) #choose action based on current state and model
        state, reward, done, info = env.step(action)
        total_reward += reward # tracks rewards

        if done: #done retrieved from ^^ 
            break
    return total_reward

if reload:
    iter_num = 10000
    for i_episode in range(10): #runs only 10 episodes
        print(f(model, True))
    sys.exit('demo finished') # quits running program when demo is over

for i in range(iterations):
    N = {}
    for k, v in model.items():
        N[k] = np.random.randn(npop, v.shape[0], v.shape[1])
                                # 24 x hl_size  # hl_size x 4
    R = np.zeros(npop) # makes a list with npop elements and fills it with zeros

    for j in range(npop):
        model_try = {}
        for k, v in model.items():
            model_try[k] = v + sigma*N[k][j] #parameters based on random variables assigned in previous for loop
        R[j] = f(model_try) # runs an episode with these 50 similar models

    A = (R - np.mean(R)) / np.std(R) #transforming data to normal standard distribution
    for k in model:
        model[k] = model[k] + alpha/(npop*sigma) * np.dot(N[k].transpose(1, 2, 0), A) #adjusts parameters based on normal standard distribution

    cur_reward = f(model)
    aver_reward = aver_reward * 0.9 + cur_reward * 0.1 if aver_reward != None else cur_reward
    print('iter %d, cur_reward %.2f, aver_reward %.2f' % (i, cur_reward, aver_reward))
    
    #Code for graphing
    areward.append(aver_reward)
    
    if i == iterations-1:
        iters = [j for j in range(len(areward))]
        print(iters)
        plt.scatter(iters,areward, np.pi * 8, 'blue', alpha=0.5)
        plt.savefig('tanh')

    #End of graphing code
    if i % 10 == 0 and allow_writing:
        pickle.dump(model, open('model-pedal%d.p' % version, 'wb'))
