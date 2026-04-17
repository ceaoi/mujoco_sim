import argparse
import numpy as np
import mujoco
from deploy.deploy_mujoco.utils.deploy_func import quat_conjugate, quat_mult, quat_rotate, quat_rotate_inverse
from legged_gym import LEGGED_GYM_ROOT_DIR

xml_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/M20_mjcf/mjcf/M20.xml"
base_name = "base_link"

def assert_close(name, a, b, atol=1e-6):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.allclose(a, b, atol=atol)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print("  got     :", a)
    print("  expected:", b)
    if not ok:
        print("  abs diff:", np.abs(a - b))
    return ok


def print_body_names(model):
    print("\n=== body names ===")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"{i:3d}: {name}")


def test_basic_quaternion_math():
    print("\n=== test_basic_quaternion_math ===")

    # 绕 y 轴 90 度
    theta = np.pi / 2
    q = np.array([np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0], dtype=np.float64)
    v = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    v_rot = quat_rotate(q, v)
    v_back = quat_rotate_inverse(q, v_rot)

    # 右手系下 x 轴绕 y 轴 +90deg -> -z
    assert_close("rotate around y 90deg", v_rot, np.array([0.0, 0.0, -1.0]), atol=1e-5)
    assert_close("inverse rotate back", v_back, v, atol=1e-5)

    prod = quat_mult(q, quat_conjugate(q))
    assert_close("q * q_conj = identity", prod, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-5)

    v2 = np.array([2.0, -1.0, 3.0], dtype=np.float64)
    v2_rot = quat_rotate(q, v2)
    assert_close("rotation preserves norm", np.linalg.norm(v2_rot), np.linalg.norm(v2), atol=1e-5)


def debug_current_base_state(model, data, base_body_id):
    print("\n=== debug_current_base_state ===")

    base_quat = data.qpos[3:7].copy().astype(np.float64)  # [w, x, y, z]
    quat_norm = np.linalg.norm(base_quat)

    omega_world = data.qvel[3:6].copy().astype(np.float64)
    omega_body = quat_rotate_inverse(base_quat, omega_world)
    omega_world_reconstructed = quat_rotate(base_quat, omega_body)

    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    gravity_body = quat_rotate_inverse(base_quat, gravity_world)
    gravity_world_reconstructed = quat_rotate(base_quat, gravity_body)

    xmat = data.xmat[base_body_id].reshape(3, 3).copy().astype(np.float64)
    gravity_body_from_xmat = xmat.T @ gravity_world
    omega_body_from_xmat = xmat.T @ omega_world

    print("base_quat:", base_quat)
    print("quat_norm:", quat_norm)
    print("omega_world:", omega_world)
    print("omega_body:", omega_body)
    print("gravity_world:", gravity_world)
    print("gravity_body:", gravity_body)
    print("xmat (body->world):\n", xmat)

    assert_close("omega round-trip", omega_world_reconstructed, omega_world, atol=1e-5)
    assert_close("gravity round-trip", gravity_world_reconstructed, gravity_world, atol=1e-5)
    assert_close("gravity body vs xmat.T @ gravity_world", gravity_body, gravity_body_from_xmat, atol=1e-5)
    assert_close("omega body vs xmat.T @ omega_world", omega_body, omega_body_from_xmat, atol=1e-5)

    print("\nInterpretation:")
    print("- quat_norm 应接近 1")
    print("- 如果机器人直立且无 pitch/roll，gravity_body 应接近 [0, 0, -1]")
    print("- yaw 旋转不会改变 gravity_body 的 z 主方向")
    print("- 你自己的 quat_rotate_inverse 结果应和 xmat.T @ vec_world 一致")


def test_known_pose_gravity(model, data, base_body_id):
    print("\n=== test_known_pose_gravity ===")

    qpos_backup = data.qpos.copy()
    qvel_backup = data.qvel.copy()

    try:
        # 设置 base freejoint 姿态：绕 y 轴 +90 度
        theta = np.pi / 2
        q = np.array([np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0], dtype=np.float64)

        data.qvel[:] = 0.0
        data.qpos[3:7] = q

        mujoco.mj_forward(model, data)

        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        gravity_body = quat_rotate_inverse(q, gravity_world)

        # 用 mujoco xmat 对拍
        xmat = data.xmat[base_body_id].reshape(3, 3).copy().astype(np.float64)
        gravity_body_xmat = xmat.T @ gravity_world

        print("set quat:", q)
        print("gravity_body from quat:", gravity_body)
        print("gravity_body from xmat:", gravity_body_xmat)

        # world z- 向量经 body inverse rotate 后应落到 body x+
        assert_close("known pose gravity (quat)", gravity_body, np.array([1.0, 0.0, 0.0]), atol=1e-5)
        assert_close("known pose gravity (xmat)", gravity_body_xmat, np.array([1.0, 0.0, 0.0]), atol=1e-5)

    finally:
        data.qpos[:] = qpos_backup
        data.qvel[:] = qvel_backup
        mujoco.mj_forward(model, data)


def test_known_pose_omega(model, data, base_body_id):
    print("\n=== test_known_pose_omega ===")

    qpos_backup = data.qpos.copy()
    qvel_backup = data.qvel.copy()

    try:
        # 设置 base freejoint 姿态：绕 z 轴 +90 度
        theta = np.pi / 2
        q = np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)], dtype=np.float64)

        data.qpos[3:7] = q

        # 世界系角速度 x+
        omega_world = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        data.qvel[:] = 0.0
        data.qvel[3:6] = omega_world

        mujoco.mj_forward(model, data)

        omega_body = quat_rotate_inverse(q, omega_world)
        xmat = data.xmat[base_body_id].reshape(3, 3).copy().astype(np.float64)
        omega_body_xmat = xmat.T @ omega_world

        print("set quat:", q)
        print("omega_world:", omega_world)
        print("omega_body from quat:", omega_body)
        print("omega_body from xmat:", omega_body_xmat)

        # world x 在 body frame 下应约为 body y-
        assert_close("known pose omega quat", omega_body, np.array([0.0, -1.0, 0.0]), atol=1e-5)
        assert_close("known pose omega xmat", omega_body_xmat, np.array([0.0, -1.0, 0.0]), atol=1e-5)

    finally:
        data.qpos[:] = qpos_backup
        data.qvel[:] = qvel_backup
        mujoco.mj_forward(model, data)


def test_joint_reorder(data, leg_joint_idx):
    if leg_joint_idx is None:
        return

    print("\n=== test_joint_reorder ===")
    qj = data.qpos[7:][leg_joint_idx]
    dqj = data.qvel[6:][leg_joint_idx]

    print("leg_joint_idx:", leg_joint_idx)
    print("reordered joint pos:", qj)
    print("reordered joint vel:", dqj)
    print("This test is only checking indexing shape/order visibility.")


def parse_leg_joint_idx(s):
    if s is None or s == "":
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def main():
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print_body_names(model)

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_name)
    if base_body_id == -1:
        raise ValueError(f"Base body '{base_name}' not found.")

    leg_joint_idx = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]

    test_basic_quaternion_math()
    debug_current_base_state(model, data, base_body_id)
    test_known_pose_gravity(model, data, base_body_id)
    test_known_pose_omega(model, data, base_body_id)
    test_joint_reorder(data, leg_joint_idx)

    print("\nAll tests finished.")


if __name__ == "__main__":
    main()