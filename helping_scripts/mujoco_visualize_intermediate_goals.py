#!/usr/bin/env python
# demonstration of markers (visual-only geoms)

import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.generated.const import GEOM_SPHERE
import glob
from matplotlib import cm


environment = "pointn"


if environment == "ant":
    environment_path = "./environments/ant_model.xml"
elif environment == "pointn":
    environment_path = "./environments/pointn.xml"
elif environment == "pointu":
    environment_path = "./environments/pointn.xml"
elif environment == "pointspiral":
    environment_path = "./environments/pointspiral.xml"        
elif environment == "sawyer_peg_push":
    environment_path = "./environments/metaworld_assets/sawyer_xyz/sawyer_peg_push.xml"
else:
    raise NotImplementedError
model = load_model_from_path(environment_path)
sim = MjSim(model)
viewer = MjViewer(sim)

index = 5500
observation = np.load("/home/erdi/Desktop/Storage/Publications/diffusion_curriculum/diffusion_curriculum/saved_log/PointNMaze-v0/2023.06.16/192700_test/trajectory/"+str(index)+".npy")
achieved_trajectory = observation[:,6:8]
intermediate_goal = observation[:,8:]


colors = cm.gist_rainbow(range(observation.shape[0]))



# achieved_trajectory = np.load("/home/erdi/Desktop/Storage/Publications/outpace_official/svaed/final_goal.npy")
while True:
    if environment == "sawyer_peg_push":
        for i in range(0,achieved_trajectory.shape[0],1):
            viewer.add_marker(  type=GEOM_SPHERE,
            pos=np.asarray(list(achieved_trajectory[i,0:3])),
            rgba=colors[i],
            size=np.asarray(([0.01]*3)),
            label="",
            )

            viewer.add_marker(  type=GEOM_SPHERE,
            pos=np.asarray(list(intermediate_goal[i,0:3])),
            rgba=[1,0,0,1],
            size=np.asarray(([0.03]*3)),
            label="",
            )
    else:
        for i in range(0,achieved_trajectory.shape[0],1):
            viewer.add_marker(  type=GEOM_SPHERE,
                                pos=np.asarray(list(achieved_trajectory[i,0:2]) + [0.7]),
                                rgba=colors[i],
                                size=np.asarray(([0.05]*3))
                            )
            
            viewer.add_marker(  type=GEOM_SPHERE,
                                pos=np.asarray(list(intermediate_goal[i,0:2]) + [0.7]),
                                rgba=[1,0,0,1],
                                size=np.asarray(([0.15]*3))
                            )
        

        
    tmp = viewer.render()