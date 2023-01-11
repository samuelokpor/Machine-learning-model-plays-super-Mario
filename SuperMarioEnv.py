import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

state = env.reset()
print(env.step(1)[3])
#print(env.reset())

#create a flag -restart or not
'''done = True
for step in range(100000):#loop through each frame in the game
    if done:
        #start the game
        env.reset() 
    #Do random actions
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
env.close() '''