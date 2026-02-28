# pose_utils.py

def quat_normalize(q):
    """归一化四元数 q = (x, y, z, w)。"""
    x, y, z, w = q
    n = (x * x + y * y + z * z + w * w) ** 0.5
    return (0.0, 0.0, 0.0, 1.0) if n == 0 else (x / n, y / n, z / n, w / n)


def rotate_vec_by_quat(v, q):
    """
    使用四元数 q 旋转向量 v。
    v: (vx, vy, vz)
    q: (x, y, z, w) —— Unity 样式
    """
    x, y, z, w = quat_normalize(q)
    vx, vy, vz = v
    ux, uy, uz = x, y, z

    dot_uv = ux * vx + uy * vy + uz * vz
    cross_x = uy * vz - uz * vy
    cross_y = uz * vx - ux * vz
    cross_z = ux * vy - uy * vx
    s2_minus_u2 = w * w - (ux * ux + uy * uy + uz * uz)

    rx = 2 * dot_uv * ux + s2_minus_u2 * vx + 2 * w * cross_x
    ry = 2 * dot_uv * uy + s2_minus_u2 * vy + 2 * w * cross_y
    rz = 2 * dot_uv * uz + s2_minus_u2 * vz + 2 * w * cross_z
    return (rx, ry, rz)
