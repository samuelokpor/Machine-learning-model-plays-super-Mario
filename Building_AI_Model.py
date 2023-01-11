import os
from stable_baselines3 import PPO 
from stable_baselines3.common.callbacks import BaseCallback
from Preprocess_Env import env

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'



class TrainAndLoggingCallBack(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallBack, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path


    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
    

#setup Model saving callback
callback = TrainAndLoggingCallBack(check_freq=10000, save_path=CHECKPOINT_DIR)

#AI model started
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)

#model starts learning here!
#model.learn(total_timesteps=1000000, callback=callback)

model =PPO.load('./train/best_model_1000000')

#starting the game
state = env.reset()
#loop through the game
while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
