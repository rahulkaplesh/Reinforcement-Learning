import gym
import random
import pickle
import numpy as np

def load_obj():
    with open('obj/q_table.pkl', 'rb') as f:
        return pickle.load(f)

q_table = load_obj()
print(q_table[328])

total_epochs, total_penalties = 0, 0
episodes = 100

env = gym.make('Taxi-v3').env

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")