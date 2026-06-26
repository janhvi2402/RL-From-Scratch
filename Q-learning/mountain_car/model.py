import numpy as np

class Environment():
    def __init__(self):
        self.start=()


#delta = target - prediction
#  w = w + alpha * delta * features(state)
import numpy as np

alpha = 0.1
gamma = 0.9

# weights
w = np.array([0.0, 0.0])

def features(state):
    return np.array([1.0, state])

def value(state):
    return np.dot(w, features(state))

# Transition: A(0) -> B(1)
s = 0
s_next = 1
reward = -1

target = reward + gamma * value(s_next)
prediction = value(s)

delta = target - prediction

w += alpha * delta * features(s)

print(w)