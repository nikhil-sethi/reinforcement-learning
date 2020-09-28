import numpy as np
import gym
import matplotlib.pyplot as plt
from PIL import Image as PILimg
from itertools import count
import shelve
# import torch
# from IPython import display
# import torchvision.transforms as T
# from collections import namedtuple
# from itertools import count


# constants
plt.figure(2)

lr = 0.001
gamma = 0.999
epsilon = 1
epsilon_max = 1
epsilon_min = 0.01
epsilon_decay = 0.001
beta1 = 0.9
beta2 = 0.999
eps = 1e-8


num_eps = 400
N = 100000 # replay memory size
n = 256 # batch size
replay_mem = []
step_size=[lr]*3

#environment
env = gym.make("CartPole-v0").unwrapped
env.reset()
#env.state=np.array([ 0.00211603,  0.02750717, -0.0075166,   0.04541408])
current_screen = env.render('rgb_array')[160:480, :, :]  # crop
current_screen = np.array(PILimg.fromarray(np.ascontiguousarray(current_screen)).resize((90, 40))).astype(np.float32)/255  # contigous>tensor>resize
current_screen = np.expand_dims(current_screen.transpose((2,0,1)), axis=0) # make CHW>add batch size dimension(helps in concatenation)
# black_screen = np.zeros_like(screen) # initial state is all black

black_screen = np.zeros([1,3,40,90]).astype(np.float32) # initial state is all black

h = black_screen.shape[2]
w = black_screen.shape[3]
ch = black_screen.shape[1] # number of channels

# Neural Network
# create  and initialise the network
inputLayer_size = h*w*ch # for three channels of each image
hidden1_size = 24
hidden2_size = 32
outputLayer_size = env.action_space.n

# kaiming initialisation
theta1 = np.random.uniform(-1, 1, size=(hidden1_size, inputLayer_size+1))    * np.sqrt(1/(inputLayer_size+1))  # 24 x 10801
theta2 = np.random.uniform(-1, 1, size=(hidden2_size, hidden1_size+1))       * np.sqrt(1/(hidden1_size+1))     # 32 x 25
theta3 = np.random.uniform(-1, 1, size=(outputLayer_size, hidden2_size+1))   * np.sqrt(1/(hidden2_size+1))     # |actions| x 33
theta1_t = np.random.uniform(-1, 1, size=(hidden1_size, inputLayer_size+1))  * np.sqrt(1/(inputLayer_size+1))  # 24 x 10801
theta2_t = np.random.uniform(-1, 1, size=(hidden2_size, hidden1_size+1))     * np.sqrt(1/(hidden1_size+1))     # 32 x 25
theta3_t = np.random.uniform(-1, 1, size=(outputLayer_size, hidden2_size+1)) * np.sqrt(1/(hidden2_size+1))     # |actions| x 33
exp_avg = [np.zeros_like(theta1), np.zeros_like(theta2), np.zeros_like(theta3)]
exp_avg_sq = [np.zeros_like(theta1), np.zeros_like(theta2), np.zeros_like(theta3)]

episode_durations = []
cum_step = 0
opt_step = 0
def sigmoid(x):  # i didn't want to do this okay..
    return 1/(1+np.exp(-x))

def RELU(x):
    a = np.copy(x)
    a[a <= 0] = 0
    return a

def dRELU(x):
    a=np.copy(x)
    a[a <= 0] = 0
    a[a > 0] = 1
    return a

# RL
for episode in range(num_eps):
    env.reset()
    #get current screen
    current_screen = env.render('rgb_array')[160:480, :, :]  # crop
    current_screen = np.array(PILimg.fromarray(np.ascontiguousarray(current_screen)).resize((90, 40))).astype(np.float32) / 255  # contigous>tensor>resize
    current_screen = np.expand_dims(current_screen.transpose((2, 0, 1)),axis=0)  # make CHW>add batch size dimension(helps in concatenation)
    # starting state is a black screen
    current_state = black_screen
    cum_cost = 0

    # print(env.state)
    for step in count():
        # initial state forward pass to get q(s) vals
        a1 = np.insert(current_state.reshape(1, inputLayer_size), 0, [1], axis=1) # 1
        a2 = np.insert(RELU(a1 @ theta1.T), 0, [1], axis=1) # 1 x (24+1)
        a3 = np.insert(RELU(a2 @ theta2.T), 0, [1], axis=1) # 1 x (32+1)
        q_current = a3 @ theta3.T # 1 x 2

        epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-cum_step * epsilon_decay)
        if np.random.rand() > epsilon:
            action = np.argmax(q_current)
        else:
            action = env.action_space.sample()
        #action=0
        _, reward, done, _ = env.step(action)

        # A final state is a black screen
        if done:
            new_state = black_screen
        else:
            new_screen = env.render('rgb_array')[160:480, :, :]  # crop
            new_screen = np.array(PILimg.fromarray(np.ascontiguousarray(new_screen)).resize((90, 40))).astype(np.float32)/255  # contigous>tensor>resize
            new_screen = np.expand_dims(new_screen.transpose((2, 0, 1)), axis=0)  # make CHW>batch size
            new_state = new_screen - current_screen
            current_screen = new_screen

        # push experiences
        if len(replay_mem) < N:
            replay_mem.append([current_state, action, reward, done, new_state])
        else:
            #push newest experience to tail if full
            replay_mem = replay_mem[1:]
            replay_mem.append([current_state, action, reward, done, new_state])

        #iteration update

        current_state = new_state

        # pick random sample from replay memory of input size
        if len(replay_mem) >= n:
            memory_sample = np.array(replay_mem)[np.random.choice(len(replay_mem), n, replace= False), :]
            X = np.concatenate(memory_sample[:, 0]).reshape(n, inputLayer_size)
            actions = memory_sample[:, 1]
            rewards = memory_sample[:, 2]
            done_all = memory_sample[:, 3]
            X_t = np.concatenate(memory_sample[:, 4]).reshape(n, inputLayer_size)
    # Training policy network
        # create design matrix
            X = np.hstack((np.ones((n, 1)), X)) # 256 x 10800+1
        # feed forward the batch to get q(s,a)vals
            a2 = np.hstack((np.ones((n, 1)), RELU(X @ theta1.T))) # 256 x (24+1)
            a3 = np.hstack((np.ones((n, 1)), RELU(a2 @ theta2.T))) # 256 x (32+1)
            q_current = a3 @ theta3.T # last layer is linear (no activation function as this matches with the target q distribution.
            #q_current = h[np.arange(n), actions.astype(int)] # 256 x 2  --> 256 x 1

        # calculate loss through another pass
            X_t = np.hstack((np.ones((n, 1)), X_t))  # 256 x 10800+1
            # take the next state as input for target network
            a2_t = np.hstack((np.ones((n, 1)), RELU(X_t @ theta1_t.T)))  # 256 x (24+1) #
            a3_t = np.hstack((np.ones((n, 1)), RELU(a2_t @ theta2_t.T)))  # 256 x (32+1)
            h_t = a3_t @ theta3_t.T # 256 x 2
            # temp[done_all] = 0  # set all final states to 0 qval
            temp = np.zeros(n)
            temp[~done_all.astype(bool)] = np.max(h_t[~done_all.astype(bool)], axis=1) # keep maximum of qval for non final states
            q_target = np.copy(q_current)
            q_target[np.arange(n), actions.astype(int)] = rewards.astype(np.float32) + gamma*temp

            # RELU(mean squared error
            cost = np.mean((q_target[np.arange(n), actions.astype(int)] - q_current[np.arange(n), actions.astype(int)])**2)# + (lam/2/n)*np.sum(theta1**2 + theta2**2+theta3**2)
            cum_cost +=cost
        # backpropagate loss
            del4 = 2*(q_current - q_target)   # 256x2
            del3 = del4 @ theta3*dRELU(a3)   # # (256x2) * (2x 33) = 256x33 layer3 error
            del2 = del3[:, 1:] @ theta2*dRELU(a2)  # (256x32) * (32x 25) = 256x25 layer 4 error

            delta3 = del4.T @ a3  # (1x256) * (256x33) = 2x33
            delta2 = del3[:, 1:].T @ a2  # (32x256) * (256x25) = 32x25
            delta1 = del2[:, 1:].T @ X  # (24x256) * (256x10801) = 24x10801

            #stacked to add regularisation in future
            theta1_grad = np.hstack((delta1[:,0].reshape(hidden1_size,1), delta1[:,1:]))/n
            theta2_grad = np.hstack((delta2[:,0].reshape(hidden2_size,1), delta2[:,1:]))/n
            theta3_grad = np.hstack((delta3[:,0].reshape(outputLayer_size,1), delta3[:,1:]))/n
            # get new theta vals using GD

            #TODO optimization
            opt_step += 1
            bias_corr1 = 1 - beta1 ** opt_step
            bias_corr2 = 1 - beta2 ** opt_step
            for i,grad in enumerate((theta1_grad, theta2_grad, theta3_grad)):
                exp_avg[i] = beta1*exp_avg[i] + (1-beta1)*grad
                exp_avg_sq[i] = beta2*exp_avg_sq[i] + (1-beta2)*(grad**2)
                denom = eps+np.sqrt(exp_avg_sq[i]/bias_corr2)
                step_size[i] = (exp_avg[i]*lr)/bias_corr1/denom

            theta1 -= step_size[0]
            theta2 -= step_size[1]
            theta3 -= step_size[2]

        cum_step += 1
        if done:
            episode_durations.append(step)
            plt.plot(episode, episode_durations[-1], 'ro')
            #moving avergage
            if episode > 101:
                moving_avg = np.mean(episode_durations[-100:])
            else:
                moving_avg=0
            #plt.pause(0.005)
            plt.plot(episode, moving_avg, "b*")

            plt.pause(0.005)

            print("Training Cost: ", cum_cost/step)
            print("100 Episode average", moving_avg)
            break


    if int(episode % 10)==0:  # change target thetas every 10 episodes
        theta1_t = np.copy(theta1)
        theta2_t = np.copy(theta2)
        theta3_t = np.copy(theta3)


'''
psedocode

-constants

gym.make
reset environment
get current screen
process screen: crop > contigous > resize > add batch dim
get black screen of same size as processed screenshot
create network
initialise random theta weights

for each episode
    reset environment
    current_state = black screen

    for each time step
        forward pass network with current_state(1x10800) to get current q(s) values (1x2)
        if random num > epsilon:
            select action which has larger q value
        else;
            select random action
        reward, done = take above action in environment 
        
        new_screen = take screenshot and process it like above
        new_state = current_state-new_screen
        store (current_state, action, reward, new_state) in replay memory
        
        if replay memory is bigger than the batch size:
            sample batchsize from replay memory
            X = batch sized input for policy
            X_t = batch input for target
            
            q_current = forward pass X into policy net
            q_target = forward pass X_t into target net
            check if any target vals are not finishing states
            q_star = reward + gamma*np.argmax(q_target) # bellman equation
            
            cost = mean squared error of q_star and q_current
            grad = backpropagate the cost over policy net
            check gradient using finite differences
            
            theta1 = theta1 - alpha*grad
            theta2 = theta2 - alpha*grad
            theta3 = theta3 - alpha*grad
        
        current_state = new_state
        if step reaches multiple of 10
            change target network weights to policy weights    
        if done:
            break    
    decay epsilon
    
    
    
# get params
import shelve
s=  shelve.open("shelve") 
theta1=s.get("theta1")
theta2=s.get("theta2")
theta3=s.get("theta3")
theta1_t=s.get("theta1_t")
theta2_t=s.get("theta2_t")
theta3_t=s.get("theta3_t")
actions = s.get("actions")
done_all = s.get("done_all")
rewards = s.get("rewards")
X= s.get("X")
X_t =s.get("X_t")
s.close()

    
'''