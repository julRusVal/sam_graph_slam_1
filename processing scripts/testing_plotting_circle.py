import numpy as np
from mayavi import mlab
from tf.transformations import quaternion_matrix

# %%
center = [1, 1, 0]
radius = 2
rot_mat = quaternion_matrix([-0.7070727, 0.7071408, -0.0000026, -0.0000026])[:3, :3]
rot_mat = quaternion_matrix([0, 0, 0, 1])[:3, :3]

# Generate point of the circle
theta = np.linspace(0, 2 * np.pi, num=100)
x = center[0] * np.ones_like(theta)
y = center[1] + radius * np.sin(theta)
z = center[2] + radius * np.cos(theta)



points_orig = np.stack([x, y, z], axis=1)
points = np.dot(points_orig, rot_mat.T)

# %% Dot test
test_point = points_orig[0, :].reshape((3, 1))
test_output = np.dot(rot_mat, test_point)

# %%
mlab.plot3d(points[:, 0], points[:, 1], points[:, 2], tube_radius=0.05)
mlab.points3d(0, 0, 0, color=(1, 1, 1), scale_factor=0.25)
mlab.show()
