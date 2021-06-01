import numpy as np
import airsim

import gym
from gym import spaces

class AirSimEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, image_shape):
        self.observation_space = spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
        self.viewer = None

    def __del__(self):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _compute_reward(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def render(self):
        return self._get_obs()

# import setup_path

import math
import time
from argparse import ArgumentParser

import gym
from gym import spaces
from PIL import Image
def norm(vec):
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)

def unit_vector(vec):
    if np.any(vec):
        n = norm(vec)
        return vec / n, n
    return np.zeros(3)

class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape
        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }
        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(7)
        self.goal_loc = np.array([70,10,-10])
        self.start_dist = norm(self.goal_loc)
        self._setup_flight()
        self.image_request = airsim.ImageRequest(
            0, airsim.ImageType.DepthPerspective, True, False
        )
        self.reward_coeff = 4
        self.old_reward = 0
        self.time_step = 1
        self.yaw_rate = 10
        self.pitch_rate = 10*3.14/180  

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
        # self.drone.takeoffAsync().join()
        # self.drone.moveToPositionAsync(0.0, 0.0, -10, 10).join()
        start_vel = 2*unit_vector(self.goal_loc)[0]
        self.drone.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, -10), airsim.to_quaternion(0, 0, 0)), True, vehicle_name='')
        self.drone.moveByVelocityAsync(start_vel[0], start_vel[1], start_vel[2], 2).join()

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))


        return im_final.reshape([84, 84, 1])

    def _get_obs(self):
        responses = self.drone.simGetImages([self.image_request])
        image = self.transform_obs(responses)

        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        collision = self.drone.simGetCollisionInfo().has_collided
        # collision_dist = self.drone.simGetCollisionInfo().
        self.state["collision"] = collision

        return image

    def _do_action(self, action):
        quad_offset = self.interpret_action(action)

        quad_vel = unit_vector(self.goal_loc-self.drone.getMultirotorState().kinematics_estimated.position.to_numpy_array())[0]
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity.to_numpy_array()
        self.drone.moveByVelocityAsync(
            quad_vel[0] + quad_offset[0],
            quad_vel[1] + quad_offset[1],
            quad_vel[2] + quad_offset[2],
            1,
        ).join()

    def _compute_reward(self, obs):
        done = 0
        reward =0
        
        pos = np.array(list(
                    (self.state["position"].x_val,
                    self.state["position"].y_val,
                    self.state["position"].z_val)))

        if self.state["collision"]:
            reward = -100


        elif 80<pos[0]<-1 or 10<pos[1]<-10 or pos[2]<-25:
            done =1

        else:
            to_goal = pos-self.goal_loc
            dist = norm(to_goal)
            

            reward = self.reward_func(dist) #//10

        reward_diff = reward-self.old_reward 
        self.old_reward = reward
        print(reward_diff)
        if reward_diff <= 0:
            reward -= 10
            done = 1

        return reward, done

    def reward_func(self, dist):
        c = self.reward_coeff
        d = self.start_dist

        reward = -(c+d)*(c/(d-dist+c)-1)
        if reward <0:
            reward = -50
        
        return reward    

    def step(self, action):
        self._do_action2(action)

        obs = self._get_obs()
        reward, done = self._compute_reward(obs)

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):

        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 2:
            quad_offset = (0, 0, -self.step_length)
        elif action == 3:
            quad_offset = (0, 0, 0)
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
        elif action == 5:
            quad_offset = (0, 0, self.step_length)
        else:
            quad_offset = (0, self.step_length, 0)

        return quad_offset

    def _do_action2(self,action):
        if action ==0:
            self.drone.rotateByYawRateAsync(self.yaw_rate,self.time_step)
        elif action==1:
            self.drone.moveByAngleRatesZAsync(0,self.pitch_rate,0,-10,self.time_step)
        elif action ==2:
            self.drone.rotateByYawRateAsync(-self.yaw_rate,self.time_step)

env = AirSimDroneEnv('127.0.0.1',0.25,(50,50))
env.drone.enableApiControl(False)
import matplotlib.pyplot as plt

# # plt_img = plt.imshow(env._get_obs())
while True:
    # env._do_action(2)
    img=env._get_obs()
    # plt_img.set_data(img)
    reward,done = env._compute_reward()
    # obs, reward, done, state = env.step()
    print(reward)
    if done:
        env.reset()
        env.drone.enableApiControl(False)
        # plt_img.set_data(img)
    plt.pause(0.001)