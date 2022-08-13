import jaxlib
import jax
import gym_minigrid
import gym

def main():
    env = gym.make('MiniGrid-Empty-5x5-v0')
    env.reset()
    print(env.render("rgb_array"))


if __name__ == '__main__':
    main()
