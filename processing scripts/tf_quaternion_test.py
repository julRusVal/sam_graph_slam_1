import tf.transformations
from tf.transformations import quaternion_from_euler

if __name__ == '__main__':
    # RPY to convert: 90deg, 0, -90deg
    q = quaternion_from_euler(-2.865817, 0.0050101, 2.034287)

    print(f"The quaternion representation is {q[0]} {q[1]} {q[2]} {q[3]}.")

    rpy = tf.transformations.euler_from_quaternion(q, 'sxyz')
    print('Done!!')