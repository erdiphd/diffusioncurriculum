"""Value function analyse
Put break points before normalization (before this line  inputs_norm_tensor = torch.hstack((obs[:, -1, :8], self.diffusion_goals)))
"""


fig = plotly.subplots.make_subplots(rows=1, cols=1, subplot_titles=("2.5k"), specs=[[{"type": "scatter3d"}]])
fig.add_trace(go.Mesh3d(
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
    ), row=1, col=1)


fig.update_layout(
        width=1600,
        height=1400,
        autosize=False,
        scene=dict(
            xaxis=dict(nticks=4, range=[-10, 15], ),
            yaxis=dict(nticks=4, range=[-10, 15], ),
            zaxis=dict(nticks=4, range=[-10, 15], ),
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='manual'
        ),
    )


#get the first observation value
obs_debug = critic_input_tensor[0,:]
#get the first action value
action_debug = action[0,:]
x = torch.linspace(6,8,100)
y = torch.linspace(15,18,100)
X, Y = torch.meshgrid(x, y)
x_input_debug = X.reshape(-1,1)
y_input_debug = Y.reshape(-1,1)
virtual_diffusion_goals_debug = torch.hstack((x_input_debug,y_input_debug)).to("cuda:0")
repeated_state_debug = torch.tile(obs_debug,[len(virtual_diffusion_goals_debug),1])
repeated_state_debug[:,-2:] = virtual_diffusion_goals_debug
repeated_state_debug = agent.normalize_obs(repeated_state_debug, agent.env_name, "diffusion")

repeated_action_debug = torch.tile(action_debug,[len(virtual_diffusion_goals_debug),1])
critic_value_debug = agent.critic(repeated_state_debug, repeated_action_debug)[0]
surface_plot = go.Surface(x=X, y=Y, z=np.array(critic_value_debug.detach().cpu()).reshape(X.shape), colorscale='Viridis')
fig.add_trace((surface_plot), row=1, col=1)

diffusion_goals_tmp = np.array(self.diffusion_goals.tolist())

fig.add_trace(go.Scatter3d(x=diffusion_goals_tmp[:, 0], y=diffusion_goals_tmp[:, 1], z=diffusion_goals_tmp[:, 1] * 0 , name='pool',mode='markers', marker=dict(size=8,color='brown',opacity=0.8)), row=1, col=1)

# fig.show()
# fig.write_html("/media/erdi/erdihome_hdd/Codes/tmp_folder/outpace_plotly/25k.html")

# "Aim reward anaylse"


# x = torch.linspace(-2,15,100)
# y = torch.linspace(-2,15,100)
# X, Y = torch.meshgrid(x, y)
# x_input_debug = X.reshape(-1,1)
# y_input_debug = Y.reshape(-1,1)
# virtual_diffusion_goals_debug = torch.hstack((x_input_debug,y_input_debug)).to("cuda:0")




# inputs_norm_tensor_tmp = torch.hstack((virtual_diffusion_goals_debug,repeated_state_debug[:,8:10]))

# inputs_norm_tensor_tmp = agent.normalize_obs(inputs_norm_tensor_tmp, agent.env_name, "diffusion")
# aim_reward = agent.aim_discriminator.forward(inputs_norm_tensor_tmp)
# surface_plot = go.Surface(x=X, y=Y, z=np.array(aim_reward.detach().cpu()).reshape(X.shape), colorscale='spectral')
# fig.add_trace((surface_plot), row=1, col=1)
fig.show()
# fig.write_html("/media/erdi/erdihome_hdd/Codes/tmp_folder/outpace_plotly/pointN/value_aim55000.html")