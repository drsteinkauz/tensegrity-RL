from gym.envs.registration import register

register(
    id='tr_obs27_env-v0',
    entry_point='tr_obs27_env.envs:tr_obs27_env',
    max_episode_steps=5000,
)