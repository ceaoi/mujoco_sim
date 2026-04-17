import numpy as np

def quat_conjugate(q):
    # q: [w, x, y, z]
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_rotate(q, v):
    # q: [w, x, y, z], v: [x, y, z]
    qv = np.concatenate([[0], v])
    return quat_mult(quat_mult(q, qv), quat_conjugate(q))[1:]

def quat_rotate_inverse(q, v):
    # 等价于 IsaacGym 的 quat_rotate_inverse：把 world 向量旋转到 body 坐标
    return quat_rotate(quat_conjugate(q), v)

def pd_ctrl(err, err_dot, kp, kd):
    return kp * err + kd * err_dot

if __name__ == "__main__":
    # test quat_rotate and quat_rotate_inverse
    q = np.array([0.7071068, 0, 0.7071068, 0.0])  # 90 deg around y-axis
    v = np.array([1, 0, 0])
    print("original v:", v)
    rotated_v = quat_rotate(q, v)
    print("rotated_v:", rotated_v)  # should be close to [0, 1, 0]
    inv_rotated_v = quat_rotate_inverse(q, rotated_v)
    print("inv_rotated_v:", inv_rotated_v)  # should be close to [1, 0, 0]