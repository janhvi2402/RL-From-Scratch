import numpy as np
import random



class Grid():
    def __init__(self):
        self.start = (0,0)
        self.goal = (5,5)
        self.obstacles = [(1,1),(1,2),(2,2),(4,1),(4,3)]
        self.package = (3,3)
        self.collected = 0

    def reset(self):
        self.collected = 0
        self.state = self.start
        return self.state, self.collected

    def step(self, action):
        r, c = self.state
        if action == 0:
            nr, nc = r-1, c
        elif action == 1:
            nr, nc = r, c+1
        elif action == 2:
            nr, nc = r+1, c
        else:
            nr, nc = r, c-1

        if not (0 <= nr <= 5 and 0 <= nc <= 5) or (nr, nc) in self.obstacles:
            reward = -5
            next_state = self.state
        else:
            next_state = (nr, nc)
            reward = -1

            if (nr, nc) == self.package:
                self.collected = 1

            if (nr, nc) == self.goal:
                reward = 100 if self.collected else -10

        self.state = next_state
        done = (next_state == self.goal and self.collected == 1)
        return next_state, self.collected, reward, done


Q = np.zeros((6, 6, 2, 4))
env = Grid()

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
alpha = 0.1
gamma = 0.9
episodes = 30

for episode in range(episodes):
    state, collected = env.reset()
    done = False

    while not done:
        r, c = state
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[r, c, collected])

        next_state, next_collected, reward, done = env.step(action)
        nr, nc = next_state

        Q[r,c,collected,action] += alpha * (
            reward + gamma * np.max(Q[nr, nc, next_collected]) - Q[r, c, collected, action]
        )

        state, collected = next_state, next_collected

    epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Bug 1 fixed: per episode

# Testing
state, collected = env.reset()
path = [state]
done = False
max_steps = 50

for _ in range(max_steps):
    r, c = state
    action = np.argmax(Q[r, c, collected])
    state, collected, reward, done = env.step(action)
    path.append(state)
    if done:
        break

if done:
    print(f"Path (len {len(path)}):", path)
else:
    print(f"Incomplete path after {max_steps} steps:", path)  # Bug 2 fixed: outside loop