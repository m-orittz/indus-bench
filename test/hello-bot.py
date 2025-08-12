import gymnasium as gym
import gym_pusht            # PushT: https://github.com/huggingface/gym-pusht

STEP_RANGE = 2 * 300 # default auto-reset happens after 300steps

def main():
    # obs_type 'state' for a state-vector, 'image' for image 
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="human")
    obs, info = env.reset()
    obsInitial = obs
    print("Initial observation:", obsInitial)

    for step in range(STEP_RANGE):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        image = env.render()
        #print(f"Step {step}: reward={reward:.3f}")#, terminated={terminated}, truncated={truncated}")
        if terminated or truncated:
            print("Terminated/Trunc'd")
            obs, info = env.reset()

    print("Initial observation:", obsInitial)
    print("  Final observation:", obs)
    

    env.close()

if __name__ == "__main__":
    main()
