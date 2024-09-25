from gym.envs.registration import register

register(
    id='tr_obs45_env-v0',
    entry_point='tr_obs45_env.envs:tr_obs45_env',
    max_episode_steps=5000,
)