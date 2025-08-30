# https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html

import os, time, glob
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode

# Something is broken, why is there a '~' folder now?
OUTPUT_DIR = os.path.expanduser("~/Videos/sim-output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

env = gym.make(
    "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    max_episode_steps=200,
    obs_mode="rgb", # there is also "state_dict", "rgbd", "state"...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    render_mode="rgb_array",
    sim_backend="gpu",
    render_backend="gpu"
    )
#test recording?
env = RecordEpisode(
        env,
        output_dir=OUTPUT_DIR,
        save_trajectory=False,
        info_on_video=False,
        #trajectory_name=f"hello-test-1",
        save_video=True,
        #video_fps=30,
        max_steps_per_video=gym_utils.find_max_episode_steps_value(env),
        render_substeps=True
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
    print(f"Done? {done}, Terminated? {terminated}")

# Trigger saving (same as CLI)
env.reset()

# None of the options below work, maybe check discord?
# Turns out vlc is broken ... WHY?
# Now, explicitly tell the RecordEpisode wrapper to finish ffmpeg
# print("Recording complete, finalizing video to disk…")
# if hasattr(env, "flush"):
#     env.flush()   # <- this waits for video writer to close

# Let I/O flush, mimic CLI pause - probably not required - vlc's fault
print("Recording complete, finalizing video to disk…")
time.sleep(2)

env.close()

# env.reset()
# for n in range(200):
#     obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
#     print(f"Step: {n}")
#     if terminated or truncated:
#         print(f"DONE!")
#         env.reset()
# env.close()