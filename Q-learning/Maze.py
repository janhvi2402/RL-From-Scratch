import numpy as np

#create an environment

class GridWorld:
    def __init__(self):
        self.size=4
        self.start=(0,0)
        self.goal=(3,3)
        self.obstacle=[(1,1),(2,1)]

    def reset(self):
        self.state=self.start
        return self.state
    
    def step(self,action):
        r,c=self.state
        #up
        if(action==0):
            nr,nc=r-1,c

        #down
        elif(action==1):
            nr,nc=r+1,c

        #left
        elif(action==2):
            nr,nc=r,c-1
        
        #right
        else:
            nr,nc=r,c+1

        if not (0<=nr<4 and 0<=nc<4):
            reward=-5
            next_state=self.state
        elif(nr,nc) in self.obstacle:
            reward=-5
            next_state=self.state
        else:
            next_state=(nr,nc)
            if next_state==self.goal:
                reward=100
            else:
                reward=-1

            
        self.state=next_state
        done=(next_state==self.goal)
        return next_state, reward, done
    

#create Q table
env=GridWorld() #object
Q=np.zeros((4,4,4)) #row,col,action
print(Q.shape)

#training agent
alpha=0.1
gamma=0.9
epsilon=0.2

episodes=1000

for episode in range(episodes):

    state=env.reset()
    done=False

    while not done:
        r,c=state
        if np.random.rand()<epsilon:
            action=np.random.randint(4)
        else:
            action=np.argmax(Q[r,c])#argmax gives index (argmax(Q[current_state]-action to take now.)) 

        next_state,reward,done=env.step(action)
        nr,nc=next_state

        #Qlearning update
        Q[r,c,action]+=alpha*(reward+gamma*np.max(Q[nc,nr])-Q[r,c,action])  #max next state   '''If I were in that next state, what is the value of the best action available there'''

        state=next_state

#test agent
state=env.reset()
path=[state]
done=False

while not done:
    r,c=state
    action=np.argmax(Q[r,c])
    state,reward,done=env.step(action)
    path.append(state)

print(path)