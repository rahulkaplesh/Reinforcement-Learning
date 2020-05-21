import gym
import random
import pickle
import numpy as np

env = gym.make('Taxi-v3').env

state = env.encode(3, 1, 2, 0) # Giving our initial state to the environment ( taxi row, taxi column, passenger index, destination index )
print(state)
env.s = state

q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

all_epochs = []
all_penalities = []

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:

        if random.uniform(0,1) < epsilon :
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * ( reward + gamma * next_max )

        q_table[state, action] = new_value

        if reward == -10 :
            penalties += 1

        state = next_state
        epochs += 1

    if i % 100 == 0 :
        print('Episode : ', i)

with open('obj/q_table.pkl', 'wb') as f:
    pickle.dump(q_table, f, pickle.HIGHEST_PROTOCOL)

print ('Training Finished !!')

print('QTable Obtained:', q_table)

print(q_table[328])