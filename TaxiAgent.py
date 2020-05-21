import gym
import random
import pickle
import numpy as np

class TaxiAgent:
    def __init__(self, citizen_no, generation_no, alpha, gamma, epsilon, numEpisode, env):
        self.citizen_no = citizen_no
        self.generation_no = generation_no
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.numEpisode = numEpisode
        self.env = env
        self.env.reset()
        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])


    def train(self):
        for i in range(0,self.numEpisode) :
            state = env.reset()
            reward = 0
            done =False
            while not done :
                if random.uniform(0,1) < self.epsilon :
                    action = env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])
                next_state, reward, done, info = env.step(action)

                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])

                new_value = (1 - self.alpha) * old_value + self.alpha * ( reward + self.gamma * next_max )

                self.q_table[state, action] = new_value

                state = next_state
            i = i + 1

    def test(self):
        total_epochs, total_penalties = 0, 0
        episodes = 100
        for i in range(0,episodes):
            state = self.env.reset()
            epochs, penalties, reward = 0, 0, 0

            done = False
    
            while not done:
                action = np.argmax(self.q_table[state])
                state, reward, done, info = env.step(action)

            if reward == -10:
                penalties += 1

            epochs += 1

            total_penalties += penalties
            total_epochs += epochs

        avg_steps_per_epsiodes = total_penalties/episodes
        avg_penalties = total_penalties/episodes

        return avg_steps_per_epsiodes,avg_penalties

if __name__ == "__main__" :
    print ("Testing!!")
    env = gym.make('Taxi-v3').env

    agent = TaxiAgent(0,
                      0,
                      0.1,
                      0.6,
                      0.1,
                      100000,
                      env);

    agent.train()
    agent.test()