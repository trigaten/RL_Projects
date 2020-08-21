from gym.envs.registration import register
 
register(id='Filler-v0', 
    entry_point='filler.envs:FillerEnv', 
)