#!/usr/bin/env python
# demonstration of markers (visual-only geoms)

import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.generated.const import GEOM_SPHERE
import glob
from matplotlib import cm
import gym
import matplotlib.pyplot as plt
import threading
import sys
sys.path.append("/media/erdi/erdihome_hdd/Codes/outpace/outpace_official/")
import mujoco_maze

path = "/media/erdi/erdihome_hdd/Codes/outpace/paper_writing/paper_plots/training_results/PointUMaze/strategy_sT/PointUMaze-v0/2024.03.12/151449_test/"
index_number = 100
robot_trajectory_total = []
intermediate_goal_total = np.zeros((1,2))
for i in range(index_number,index_number +120000, 3000):
    # observation = np.load(path + "trajectory/" + str(i) + ".npy", allow_pickle=True)
    # achieved_trajectory = observation[:, 6:8]
    # intermediate_goal = observation[:, 8:]
    # intermediate_goal_total.append(intermediate_goal)
    # robot_trajectory_total.append(achieved_trajectory)
    intermediate_goal = np.load(path + "trajectory/generated_curriculum_points" + str(i) + ".npy", allow_pickle=True).round(decimals=1)
    intermediate_goal_total = np.append(intermediate_goal_total, intermediate_goal,axis=0)


# robot_trajectory = np.array(robot_trajectory_total).reshape(-1,2)

intermediate_goals = intermediate_goal_total
# intermediate_goals  = intermediate_goals.round(decimals=1)
# intermediate_goals = np.unique(intermediate_goals,axis=0)



env = gym.make("PointUMaze-v1")
print(gym.__file__)

obs = env.reset()

# jump_step = 1
cmap = plt.get_cmap('gist_rainbow', int(intermediate_goals.shape[0]/1.1))
add_marker = True
tmp = env.render('rgb_array', width=1014*4,height=1014*4)
for i in range(intermediate_goals.shape[0]):
    env.wrapped_env.viewer.add_marker(type=GEOM_SPHERE, pos=np.asarray(list(intermediate_goals[i, 0:2]) + [0.7]),
                                      rgba=cmap(i), size=np.asarray(([0.1] * 3)), label="")
    # env.wrapped_env.viewer.add_marker(type=GEOM_SPHERE, pos=np.asarray(list(robot_trajectory[i, 0:2]) + [0.7]),
    #                                   rgba=cmap(i), size=np.asarray(([0.1] * 3)), label="")
tmp = env.render('rgb_array', width=1014*4,height=1014*4)
plt.imshow(tmp)
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)# Turn off axis
plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0)
