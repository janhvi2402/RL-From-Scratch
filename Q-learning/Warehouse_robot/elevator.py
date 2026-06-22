import numpy as np
import random

class Elevator():
    def __init__(self, num_floors=5):
        self.start = 2
        self.num_floors = num_floors

    def reset(self, num_passengers=2):
        self.state = self.start
        # randomly place passengers on distinct floors
        floors = random.sample(range(self.num_floors), num_passengers)
        self.waiting=[]
        for floor in range(self.num_floors):
            if floor in floors:
                self.waiting.append(1)
            else:
                self.waiting.append(0)
        return self.state, tuple(self.waiting)


 # random.sample(range(5), 2) picks 2 UNIQUE floors from [0,1,2,3,4]
    # e.g. → [2, 4]  (never repeats same floor)
    # unlike random.randint which could pick same floor twice


 # builds a list of 5 elements, 1 if that floor has passenger, else 0
    # if floors=[2,4]:
    # i=0 → 0 not in [2,4] → 0
    # i=1 → 1 not in [2,4] → 0
    # i=2 → 2 in [2,4]     → 1
    # i=3 → 3 not in [2,4] → 0
    # i=4 → 4 in [2,4]     → 1
    # result: [0, 0, 1, 0, 1]

    
    def waiting_to_index(self):
        # convert [1,0,1,0,0] → integer bitmask (0-31)
        return int("".join(map(str, self.waiting)), 2)
    # converts waiting list to a single integer for Q-table indexing
    # step 1: map(str, self.waiting)
    # [1,0,1,0,0] → ['1','0','1','0','0']
    
    # step 2: "".join(...)
    # ['1','0','1','0','0'] → "10100"
    
    # step 3: int("10100", 2)
    # reads "10100" as binary → 20 in decimal

    def step(self, action):
        s = self.state
        ns = s - 1 if action == 0 else s + 1

        if not (0 <= ns < self.num_floors):
            # out of bounds — penalize by passengers still waiting
            reward = -sum(self.waiting)
            next_state = self.state
        else:
            next_state = ns
            if self.waiting[ns] == 1:
                self.waiting[ns] = 0   # pick up passenger
                reward = 20 - sum(self.waiting)  # +20 for pickup, minus remaining
            else:
                reward = -sum(self.waiting)      # penalize by passengers waiting

        self.state = next_state
        done = sum(self.waiting) == 0            # all passengers picked up
        return next_state, tuple(self.waiting), reward, done


env = Elevator()
Q = np.zeros((5, 32, 2))   # (floor, passenger_bitmask, action)
epsilon = 0.1
alpha = 0.1
gamma = 0.9
episodes = 5000

# Training
for episode in range(episodes):
    state, waiting = env.reset(num_passengers=2)
    done = False

    while not done:
        s = state
        w = env.waiting_to_index()

        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmax(Q[s, w])

        ns, next_waiting, reward, done = env.step(action)
        nw = env.waiting_to_index()

        Q[s, w, action] += alpha * (
            reward + gamma * np.max(Q[ns, nw]) - Q[s, w, action]
        )
        state = ns

# Testing
state, waiting = env.reset(num_passengers=2)
done = False
path = [state]
print(f"Start: floor {state}, Passengers waiting at: {[i for i,w in enumerate(waiting) if w]}")

for _ in range(30):
    w = env.waiting_to_index()
    action = np.argmax(Q[state, w])
    ns, waiting, reward, done = env.step(action)
    path.append(ns)
    state = ns
    if done:
        break

print("Path:", path)
print("Passengers remaining:", sum(env.waiting))