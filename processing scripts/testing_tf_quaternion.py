import tf.transformations
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import numpy as np
import gtsam
from tf.transformations import quaternion_matrix

if __name__ == '__main__':
    # RPY to convert: 90deg, 0, -90deg
    q = quaternion_from_euler(-2.865817, 0.0050101, 2.034287)

    print(f"The quaternion representation is {q[0]} {q[1]} {q[2]} {q[3]}.")

    rpy = tf.transformations.euler_from_quaternion(q, 'sxyz')
    print('Done!!')

# %%
# dr [x, y, z, w]
# gt [x, y, z, w]
# # test one
# dr_q = [-0.728381, -0.667729, 0.122363, 0.092872]
# gt_q = [0.988453, -0.042860, -0.021235, -0.143783]
#
# # test two
# dr_q = [-0.782495, -0.608467, 0.091317, 0.095558]
# gt_q = [0.983543, -0.123174, 0.003016, -0.132142]

"""
The above tests are not correct as they don't take into account the frame differences of the odometries
The correction needs to be applied after they are transformed to the same frame, 
in the end this will not need to be used.
"""

# # pose_graph # 80
# dr_q = [-0.982961933951889, 0.12770378978294, -0.003634188683638, 0.13215283254342]
# gt_q = [0.090850223918956, 0.09604647469177, 0.785557448969203, 0.604500459786555]
#
#
# # pose_graph # 80  r_q = [2 ** (1 / 2) / 2, 2 ** (1 / 2) / 2, 0, 0]
# dr_q = [-0.132152566866684, -0.003622293193536, -0.127608402070231, -0.982974401387772]
# gt_q = [0.0908637962712, 0.096031723026671, 0.785459536741498, 0.604627980531455]


# pose_graph # 80  r_q = [0, 0, 0, 1]
dr_q = [0.785359264884362, 0.604758833406886, -0.090876404582151, -0.096015923506774]
gt_q = [0.0908637962712, 0.096031723026671, 0.785459536741498, 0.604627980531455]

# dr
dr_q_inv = dr_q.copy()
dr_q_inv[3] = -dr_q_inv[3]
dr_inv = tf.transformations.quaternion_inverse(dr_q)

# Find the rotation gt_q = rotation * dr_q
uncorrected_rpy = euler_from_quaternion([dr_q[0], dr_q[1], dr_q[2], dr_q[3]])
corrected_y = np.pi - uncorrected_rpy[2]
corrected_q = quaternion_from_euler(uncorrected_rpy[0], uncorrected_rpy[1], corrected_y)

r_q_calc = tf.transformations.quaternion_multiply(gt_q, dr_inv)

q_out = tf.transformations.quaternion_multiply(r_q_calc, dr_q)

rot3 = gtsam.Rot3.Quaternion(dr_q[3], dr_q[0], dr_q[1], dr_q[2])
rot3_yaw_dr = rot3.yaw()

rot3 = gtsam.Rot3.Quaternion(q_out[3], q_out[0], q_out[1], q_out[2])
rot3_yaw_check = rot3.yaw()

rot3 = gtsam.Rot3.Quaternion(gt_q[3], gt_q[0], gt_q[1], gt_q[2])
rot3_yaw_gt = rot3.yaw()

# %% q to r.mat
rot_mat = quaternion_matrix([-0.7070727, 0.7071408, -0.0000026, -0.0000026])[:3,:3]