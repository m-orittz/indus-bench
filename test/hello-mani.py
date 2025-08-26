# https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html


import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers import RecordEpisode

env = gym.make(
    "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode="rgb", # there is also "state_dict", "rgbd", "state"...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    render_mode="rgb_array",
    sim_backend="gpu",
    render_backend="gpu"
    )
#test recording?
env = RecordEpisode(
        env,
        output_dir=f"/home/moritzwagner//Videos/sim-output",
        save_trajectory=False,
        trajectory_name=f"hello-test-1",
        save_video=True,
        video_fps=30
    )

print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0) # reset with a seed for determinism
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    #env.render()  # a display is required to render
    print(f"Obs shape: {obs.shape}, Reward shape {reward.shape}, Done shape {done.shape}")
env.close()

# env.reset()
# for n in range(200):
#     obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
#     print(f"Step: {n}")
#     if terminated or truncated:
#         print(f"DONE!")
#         env.reset()
# env.close()