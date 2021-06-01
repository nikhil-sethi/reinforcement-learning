from os import sep
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import math

def norm_2d(vec):
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2)


def outside(agent):
    if any(np.abs(agent.state.p_pos) > 1):
        return True
    return False

class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 10
        world.num_agents = num_agents
        num_adversaries = 0
        num_landmarks = 0

        world.geofence = np.array([
            [1, 1],   # bottom left
            [-1, -1],
        ])
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            # if i!=1:
                # agent.movable = False
            agent.id = i
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.01
            agent.r_cohere=1.2
            agent.r_align = 0.7
            agent.r_coll = 0.3
            agent.pre_vel= np.zeros(2)
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08
        # make initial conditions
        self.reset_world(world)
        # world.agents[0].color = np.array([0.85, 0.35, 0.35])
        # for i in range(1, world.num_agents):
        #     world.agents[i].color = np.random.random(size=(3,))#np.array([0.35, 0.35, 0.85])
        # # random properties for landmarks
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.color = np.array([0.15, 0.15, 0.15])

        return world

    def reset_world(self, world):
        # random properties for agents
        world.agents[0].color = np.array([0.85, 0.35, 0.35])
        for i in range(1, world.num_agents):
            world.agents[i].color = np.random.random(size=(3,))#np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])
            
        # set goal landmark
        # goal = np.random.choice(world.landmarks)
        # goal.color = np.array([0.15, 0.65, 0.15])
        # for agent in world.agents:
        #     agent.goal_a = goal

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.random.uniform(-1, +1, world.dim_p)#np.zeros(world.dim_p)
            # print(agent.state.p_pos)
            # if agent.id ==1:
            #     agent.state.p_pos = np.array([-0.1,-0.1])#np.random.uniform(-1, +1, world.dim_p)
            #     agent.state.p_vel = np.array([1,1])
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []

            for l in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # alignment reward: minimiza velocity differences
        # separation/collision reward: maximise short range repulsion
        sep_rew = 0
        sep_count = 0
        align_rew = 0
        align_count = 0
        cohere_rew = 0
        cohere_count = 0
        for other in world.agents:
            dist = norm_2d(other.state.p_pos-agent.state.p_pos)
            if  agent.r_align < dist:   # 1.2
                cohere_rew -= dist
                cohere_count +=1
            elif  agent.r_coll < dist < agent.r_align:   # 0.7
                align_rew -= norm_2d(other.state.p_vel-agent.state.p_vel)
                align_count+=1
            elif 0 < dist < agent.r_coll:   # 0.2
                sep_rew += (dist**0.3-agent.r_coll**0.3)
                sep_count +=1

        if cohere_count:
            cohere_rew /= cohere_count
        if align_count:
            align_rew /= align_count
        if sep_count:
            sep_rew /= sep_count
        # negative reward for staying at rest. Bewakoof agents
        vel_mag = norm_2d(agent.state.p_vel)
        # least acceleration too avoid jumps and vibrations
        vel_mag_del = -(norm_2d(agent.pre_vel) - norm_2d(agent.state.p_vel))
        theta = -(abs(math.atan(agent.pre_vel[1]/agent.pre_vel[0]) - math.atan(agent.state.p_vel[1]/agent.state.p_vel[0])))
        # acc_mag = -norm_2d(agent.pre_vel - agent.state.p_vel)+norm_2d(agent.pre_vel + agent.state.p_vel)
    
        vel_rew = 0
        if vel_mag < 0.1:
            vel_rew = vel_mag-0.1
        motion_rew =  2*vel_mag_del + 1*vel_mag
        # wall collisions: velocity alignment with virtual shill agents
        wall_rew = 0
        distances = np.abs((agent.state.p_pos - world.geofence).flatten())
        
        min_dist = np.min(distances)
        if min_dist < 0.10:
            wall_id = np.argmin(distances)
            if wall_id == 0:
                shill_vel = [-1,0]
            elif wall_id == 1:
                shill_vel = [0,-1]
            elif wall_id == 2:
                shill_vel = [1,0]
            elif wall_id == 3:
                shill_vel = [0,1]

            wall_rew -= norm_2d(np.array(shill_vel)-agent.state.p_vel)
        if outside(agent):
            wall_rew -= 4
        # print("{:8.4f},{:8.4f},{:8.4f},{:8.4f},{:8.4f},{:8.4f},{:8.4f}".format(2*align_rew, cohere_rew, 2*sep_rew,2*vel_mag_del, 1.5*vel_mag, 2*motion_rew, 3*wall_rew))
        return 2*align_rew + +1*cohere_rew + 5*sep_rew + 2*motion_rew + 3*wall_rew

        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = True
        shaped_adv_reward = True

        # Calculate negative reward for adversary
        adversary_agents = self.adversaries(world)
        if shaped_adv_reward:  # distance-based adversary reward
            adv_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
        else:  # proximity-based adversary reward (binary)
            adv_rew = 0
            for a in adversary_agents:
                if np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) < 2 * a.goal_a.size:
                    adv_rew -= 5

        # Calculate positive reward for agents
        good_agents = self.good_agents(world)
        if shaped_reward:  # distance-based agent reward
            pos_rew = -min(
                [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
        else:  # proximity-based agent reward (binary)
            pos_rew = 0
            if min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents]) \
                    < 2 * agent.goal_a.size:
                pos_rew += 5
            pos_rew -= min(
                [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
        return pos_rew + adv_rew

    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark
        shaped_reward = True
        if shaped_reward:  # distance-based reward
            return -np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:  # proximity-based reward (binary)
            adv_rew = 0
            if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < 2 * agent.goal_a.size:
                adv_rew += 5
            return adv_rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame


        # entity_pos = []
        # for entity in world.landmarks:
        #     entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # # entity colors
        # entity_color = []
        # for entity in world.landmarks:
        #     entity_color.append(entity.color)
        # communication of all other agents
        other_pos = []
        other_vel = []
        for other in world.agents:
            # dist = norm_2d(other.state.p_pos-agent.state.p_pos)
            if other is agent: continue
            p_rel = other.state.p_pos-agent.state.p_pos
            v_rel = other.state.p_vel-agent.state.p_vel
            other_pos.append(p_rel)
            other_vel.append(v_rel)

        # vel_mag = norm_2d(agent.state.p_vel)
        # wall observations
        distances = np.abs((agent.state.p_pos - world.geofence).flatten())
        min_dist = min(distances)
        wall_id = np.argmin(distances)
        if wall_id == 0:
            shill_vel = [-1,0]
        elif wall_id == 1:
            shill_vel = [0,-1]
        elif wall_id == 2:
            shill_vel = [1,0]
        elif wall_id == 3:
            shill_vel = [0,1]

        
        shill_vel_rel = np.array(shill_vel)-agent.state.p_vel
        vel_del = agent.pre_vel-agent.state.p_vel
        self_state = [[min_dist], agent.state.p_vel, shill_vel_rel]
        # if not agent.adversary:
        #     return np.concatenate([agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos)
        # else:
        return np.concatenate(self_state +other_pos + other_vel)
