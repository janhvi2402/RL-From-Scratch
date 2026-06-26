import numpy as np
import random
class Environment():
    def __init__(self):
        self.start=2
        self.steps = 0
    def reset(self):
        self.state=self.start
        self.steps = 0
        return self.state

    def step(self,action):
        s=self.state
        if action==1:
            ns=self.state+1
        else:
            ns=self.state-1

        if not(0<=ns<=4) :
            reward=-5
            next_state=self.state
        else:
            if(ns==4):
                reward=10
                next_state=ns
            else:
                self.steps += 1
                reward = -self.steps
                next_state=ns
        self.state=next_state
        done=(next_state==4)
        return reward,next_state,done
#delta = target - prediction
#  w = w + alpha * delta * features(state)

x=np.array([(1,0),
   (1,1),
   (1,2),
   (1,3),
   (1,4)
   ])

actions=[-1,1]

w= np.zeros(2)
V = np.dot(x, w)

episodes=10
gamma=0.9
alpha=0.1
epsilon=0.2
env=Environment()
for episode in range(episodes):
    state=env.reset()
    done=False

    while not done:
        action=random.choice(actions)
            
        reward,next_state,done=env.step(action)
        prediction = np.dot(w, x[state])

        if done:
            target = reward
        else:
            target = reward + gamma * np.dot(w, x[next_state])

        delta = target - prediction
        w = w + alpha * delta * x[state]

        state=next_state
    V = np.dot(x, w)

    print(f"Episode {episode}")
    print("Weights:", w)
    print("Values :", V)
    print("-"*30)

print("State:", state)
print("Reward:", reward)
print("Prediction:", prediction)
print("Target:", target)
print("Delta:", delta)
print("Weights:", w)
print("----------------")

