from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt

from SuperMarioEnv import  gym_super_mario_bros, JoypadSpace, SIMPLE_MOVEMENT

#create the base environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# simplify controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
#grayscale the env
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap insode the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env =VecFrameStack(env, 4, channels_order='last')


state  = env.reset()
#plt.imshow(state)
print(state.shape)
state, reward, done, info = env.step([env.action_space.sample()])
'''plt.figure(figsize=(10,8))
for idx in range(state.shape[3]):
    plt.subplot(1, 4, idx+1)
    plt.imshow(state[0][:,:,idx])

plt.show()'''