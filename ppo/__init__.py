import jaxlib
import jax
import tensorflow as tf
# import gym_minigrid
# import gym
# import tree
from pprint import pprint

def main():
    pprint(tf.config.list_physical_devices())
    pprint(jax.devices())
    # structure = [[1], [[[2, 3]]], [4]]
    # print(tree.flatten(structure))
    # env = gym.make('MiniGrid-Empty-5x5-v0')
    # env.reset()
    # print(env.render("rgb_array"))


if __name__ == '__main__':
    main()
