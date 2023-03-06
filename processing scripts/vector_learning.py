#!/usr/bin/env python3
import numpy as np

# Define the vectors that define the plane
v1 = np.array([0, 1, 0])
v2 = np.array([0, 0, -1])

# Define the point on the plane and the normal vector
Q = np.array([-5, 4, 0])
n = np.cross(v1, v2)

# Define the line
P = np.array([-4, 3, 0])
d = np.array([1, 0, 0])

# Calculate the intersection point
t = -np.dot(n, (P - Q)) / np.dot(n, d)
intersection_point = P + t * d

# Express the intersection point as a linear combination of v1 and v2 using the pseudoinverse
A = np.vstack([v1, v2]).T
b = intersection_point - Q
x = np.linalg.pinv(A).dot(b)
s, t = x
intersection_point_linear_combination = f"{s}*v1 + {t}*v2 + {Q}"

# Print the result
print("The intersection point expressed as a linear combination of v1 and v2 is:", intersection_point_linear_combination)