# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# load image
image_path = "/Users/julian/KTH/Degree project/sam_slam/processing scripts/new_img.jpg"
img = Image.open(image_path)
img = np.array(img)/255

# Define start and end x,y coordinates and height range
start_x = -4
start_y = 0
end_x = -4
end_y = 5
height_range = [0, -7.5]

# Define the resolution of the meshgrid
res_x = img.shape[1]
res_h = img.shape[0]

# Create the meshgrid
x_linspace = np.linspace(start_x, end_x, res_x)
y_linspace = np.linspace(start_y, end_y, res_x)
h_linspace = np.linspace(height_range[0], height_range[1], res_h)

X, Z = np.meshgrid(x_linspace, h_linspace)
Y, _ = np.meshgrid(y_linspace, h_linspace)

# %%

# Plot the surface using the plot_surface function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot_surface(X, Y, Z, rstride=5, cstride=5,
                facecolors=img)

# Add axes labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
