import numpy as np
import plotly.graph_objects as go
import plotly
import torch




PointUMaze = go.Mesh3d(
        # 8 vertices of a cube
        x=np.array([ 0, 20,  0, 20,  0,  0, 20, 20,  4,  4,  4,  4, 16, 16, 16, 16,  4,  4, 4,  4, 12, 12, 12, 12]) - 6,
        y=np.array( [20, 20, 20, 20,  0,  0, -0, -0, 16, 12, 16, 12, 16, 16,  4,  4,  4,  4, 8,  8,  8,  8, 12, 12]) - 6,
        z=np.array([2, 2, 0, 0, 2, 0, 2, 0, 2, 2, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0,]) ,

        i=[0, 2, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 9, 11, 1, 3, 18, 16, 16, 14, 14, 12, 12, 8, 8, 9, 9, 22, 11, 21, 11, 11, 10, 10, 13, 13, 15, 15, 17, 17],
        j=[1, 1, 0, 0, 4, 4, 9, 9, 8, 8, 12, 12, 14, 14, 16, 16, 18, 18, 20, 20, 22, 22, 6, 6, 0, 0, 4, 4, 6, 6, 1, 1, 0, 0, 18, 18, 23, 19, 19, 5, 5, 2, 2, 3, 3, 7, 7, 5],
        k=[2, 3, 5, 2, 7, 5, 10, 11, 13, 10, 15, 13, 17, 15, 19, 17, 21, 19, 23, 21, 11, 23, 3, 7, 16, 4, 14, 6, 12, 1, 8, 0, 9, 18, 22, 20, 21, 11, 5, 10, 2, 13, 3, 15, 7, 17, 5, 19],
        opacity=0.2,
        color='#DC143C',
        name='input',
    )

PointNMaze = go.Mesh3d(
        # 8 vertices of a cube
      x = np.array([20., -0., 20.,  0., 20., 20., -0.,  0.,  8., 16.,  8., 16.,  8.,
        8., 16., 16., 16., 16.,  4.,  4.,  4.,  4., 12., 12., 12., 12.,
        4.,  4.,  4.,  4., 16., 16.]) - 6,

y = np.array([ 0.,  0.,  0.,  0., 28., 28., 28., 28., 20., 20., 20., 20., 16.,
       16., 16., 16.,  4.,  4.,  4.,  4.,  8.,  8.,  8.,  8., 12., 12.,
       12., 12., 24., 24., 24., 24.]) - 6,

z = np.array([2., 2., 0., 0., 2., 0., 2., 0., 2., 2., 0., 0., 2., 0., 2., 0., 2.,
       0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0.]),

        i=[0, 2, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 9, 11, 1, 3, 14, 20, 0, 1, 1, 22, 20, 20, 6, 6, 4, 4, 8, 9, 9, 4, 27, 29, 29, 31, 31, 25, 27, 27, 3, 3, 2, 2, 13, 15, 15, 2],
        j=[1, 1, 0, 0, 4, 4, 9, 9, 8, 8, 12, 12, 14, 14, 16, 16, 18, 18, 20, 20, 22, 22, 24, 24, 26, 26, 28, 28, 30, 30, 6, 6, 16, 6, 16, 16, 18, 24, 24, 26, 26, 28, 28, 30, 12, 12, 14, 14, 3, 3, 7, 7, 5, 23, 23, 21, 21, 19, 19, 17, 10, 10, 11, 11],
        k=[2, 3, 5, 2, 7, 5, 10, 11, 13, 10, 15, 13, 17, 15, 19, 17, 21, 19, 23, 21, 25, 23, 27, 25, 29, 27, 31, 29, 11, 31, 3, 7, 0, 1, 1, 18, 20, 20, 26, 6, 28, 4, 30, 9, 9, 14, 4, 0, 29, 7, 31, 5, 11, 27, 21, 3, 19, 2, 17, 15, 15, 11, 2, 5],
        opacity=0.2,
        color='#DC143C',
        name='input',
    )

PointSpiralMaze = go.Mesh3d(
        # 8 vertices of a cube
x = np.array([24.,  8., 24.,  8., 24., 24.,  4.,  4.,  4.,  4., 24., 24., 24.,
       24., 12., 12., 12., 12., 20., 20., 20., 20.,  8.,  8., 28., -0.,
       28.,  0., 28., 28., -0.,  0.]),
y = np.array([ 8.,  8.,  8.,  8.,  4.,  4.,  4.,  4., 24., 24., 24., 24., 12.,
       12., 12., 12., 16., 16., 16., 16., 20., 20., 20., 20.,  0.,  0.,
        0.,  0., 28., 28., 28., 28.]),
z = np.array([2., 2., 0., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2.,
       0., 2., 0., 2., 0., 2., 0., 2., 2., 0., 0., 2., 0., 2., 0.]),
        i=[0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 28, 29, 30, 31, 25, 27, 1, 3, 0, 24, 25, 25, 30, 30, 28, 28, 18, 16, 16, 14, 14, 12, 12, 28, 21, 23, 23, 3, 3, 2, 5, 7, 7, 9, 9, 11, 11, 13, 13, 2],
        j=[1, 1, 0, 0, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16, 16, 18, 18, 20, 20, 25, 25, 24, 24, 28, 28, 30, 30, 22, 22, 4, 4, 4, 6, 6, 8, 8, 10, 20, 20, 22, 22, 1, 1, 0, 0, 19, 19, 17, 17, 15, 15, 26, 26, 27, 27, 31, 31, 29, 29, 26, 26],
        k=[2, 3, 5, 2, 7, 5, 9, 7, 11, 9, 13, 11, 15, 13, 17, 15, 19, 17, 21, 19, 23, 21, 26, 27, 29, 26, 31, 29, 27, 31, 3, 23, 24, 25, 6, 30, 8, 28, 10, 12, 16, 22, 14, 1, 12, 0, 28, 24, 23, 17, 3, 15, 2, 13, 7, 27, 9, 31, 11, 29, 13, 26, 2, 5],
        opacity=0.2,
        color='#DC143C',
        name='input',
    )

fig = plotly.subplots.make_subplots(rows=1, cols=1, subplot_titles=("PointUMaze"), specs=[[{"type": "scatter3d"}]])
fig.add_trace(PointUMaze, row=1, col=1)


fig.update_layout(
        width=1600,
        height=1400,
        autosize=False,
        scene=dict(
            xaxis=dict(nticks=4, range=[-30, 30], ),
            yaxis=dict(nticks=4, range=[-30, 30], ),
            zaxis=dict(nticks=4, range=[-15, 15], ),
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='manual'
        ),
    )
fig.show()

fig = plotly.subplots.make_subplots(rows=1, cols=1, subplot_titles=("PointNMaze"), specs=[[{"type": "scatter3d"}]])
fig.add_trace(PointNMaze, row=1, col=1)


fig.update_layout(
        width=1600,
        height=1400,
        autosize=False,
        scene=dict(
            xaxis=dict(nticks=4, range=[-30, 30], ),
            yaxis=dict(nticks=4, range=[-30, 30], ),
            zaxis=dict(nticks=4, range=[-15, 15], ),
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='manual'
        ),
    )
fig.show()


fig = plotly.subplots.make_subplots(rows=1, cols=1, subplot_titles=("PointSpiralMaze"), specs=[[{"type": "scatter3d"}]])
fig.add_trace(PointSpiralMaze, row=1, col=1)


fig.update_layout(
        width=1600,
        height=1400,
        autosize=False,
        scene=dict(
            xaxis=dict(nticks=4, range=[-30, 30], ),
            yaxis=dict(nticks=4, range=[-30, 30], ),
            zaxis=dict(nticks=4, range=[-15, 15], ),
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='manual'
        ),
    )
fig.show()