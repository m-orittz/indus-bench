# https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html


import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers import RecordEpisode

env = gym.make(
    "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode="rgb", # there is also "state_dict", "rgbd", "state"...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    #render_mode="human"
    sim_backend="gpu",
    render_backend="gpu",
    viewer_camera_configs=dict(shader_pack="rt-fast")
)
#test recording?
env = RecordEpisode(
        env,
        output_dir=f"~/Videos/sim-output",
        trajectory_name=f"hello-test-1",
        info_on_video=True,
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