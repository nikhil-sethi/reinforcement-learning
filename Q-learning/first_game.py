import gym
import numpy as np
import time
from IPython.display import clear_output

env = gym.make("FrozenLake-v0")
actionSpace_size = env.action_space.n
stateSpace_size = env.observation_space.n

q_table = np.zeros((stateSpace_size, actionSpace_size))

alpha = 0.1 # learning rate
epsilon = 1 # e-greedy parameter
epsilon_min = 0.0
epsilon_max = 1
epsilon_decay =0.001

gamma = 0.99 # discount rate

num_eps = 10000
max_steps = 100

reward_list = []

for episode in range(num_eps):
    current_state = env.reset()
    done = False
    episode_reward = 0
    for j in range(max_steps):
        if np.random.rand() > epsilon:
            action = np.argmax(q_table[current_state, :])
        else:
            action = env.action_space.sample() #random sample

        new_state, reward, done, info = env.step(action)    #single time step
        #print(reward)
        new_action = np.argmax(q_table[new_state, :])
        q_star = reward + gamma*np.max(q_table[new_state, :]) #bellman equation

        q_table[current_state, action] = (1-alpha)*q_table[current_state,action] +(alpha)*q_star

        current_state = new_state
        episode_reward += reward

        if done:
            break


    epsilon = epsilon_min + (epsilon_max-epsilon_min)*np.exp(-epsilon_decay*episode)
    reward_list.append(episode_reward)

reward_perThousand = np.split(np.array(reward_list), num_eps/1000)
print("Average reward per thousand episodes \n ", np.mean(reward_perThousand, 1))

for episode in range(3):
    current_state = env.reset()
    done = False
    print("Episode number: ", episode+1, "\n\n")
    time.sleep(1)
    for step in range(max_steps):
        clear_output(wait= True)
        env.render()

        action = np.argmax(q_table[current_state, :])
        new_state, reward, done, info = env.step(action)
        time.sleep(0.3)
        if done:
            clear_output(wait=True)
            env.render()
            if reward:
                print("***** You found the frisbee *****")
                time.sleep(3)
            elif not reward:
                print("***** You fell into a hole *****")
                time.sleep(3)
            clear_output(wait= True)
            break
        current_state = new_state

'''
#create environment
actions =
states =

r= np.random.rand()
epsilon= 1
alpha= 01
gamma=0.9
q_table= zeros(statesize, actionsize)
r_table=

# start game
for num iters:
    reset environment
    for each episode:
        s= current state
        if r>epsilon:
            #exploitation- choose action with maximum qval
        else:
            #explore - take random action
            a= env.actionspace.sample
        q_star= 
        
        loss= q_star-q[s,a]
        
        qtable[s,a]= (1-alpha)*qtable[s,a]+(alpha)*q_star
    
    epsilon *=0.9  #decay

'''