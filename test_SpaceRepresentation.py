import numpy as np
import pytest
from numpy import cos, pi, sin, sqrt
from space_representation import (
    AxisAngle,
    EulerAnglesZYZ,
    Quaternion,
    RotationMatrix,
    TransformationMatrix,
)


def test_RotationMatrix():
    R = RotationMatrix.rotation_x(pi / 2)

    assert R.inv() == np.linalg.inv(R.matrix)
    assert np.linalg.det(R.matrix) == 1

    Rtest = np.array([[-1 / 2, 0, -sqrt(3) / 2], [0, 1, 0], [-sqrt(3) / 2, 0, -1 / 2]])
    assert RotationMatrix.check_rotation_matrix(Rtest) is False
    assert Rtest[0, 0] == -1 / 2

    Rtest = np.array([[-1 / 2, 0, -sqrt(3) / 2], [0, 1, 0], [sqrt(3) / 2, 0, -1 / 2]])
    assert RotationMatrix.check_rotation_matrix(Rtest) is True

    Vtest = np.array([0, sqrt(3), 1])
    Rx30 = RotationMatrix.rotation_x(30 * pi / 180)
    assert Rx30 @ Vtest == np.array([0, 1, sqrt(3)])

    Mattest = np.array(
        [
            [1, 0, 0],
            [2, 1, 0],
            [3, 2, 1],
        ]
    )

    assert Rx30 @ Mattest == Rx30.matrix @ Mattest
    assert np.allclose(Rx30 @ Mattest, Rx30.matrix @ Mattest)


def test_TransformationMatrix():
    Ttrasl1 = TransformationMatrix.translation_pure(
        traslation_vector=np.array([6, -2, 10])
    )
    Trotx90 = TransformationMatrix.rotation_pure("X", 90 * pi / 180)

    aTb = Ttrasl1.matrix @ Trotx90.matrix
    print(aTb)


def test_RollPitchYaw():
    mat = np.array(
        [
            [-0.5, -0.557, 0.663],
            [0.866, -0.321, 0.383],
            [0, 0.766, 0.643],
        ]
    )
    R = RotationMatrix(mat)

    R_rpy_positive = R.to_roll_pitch_yaw()
    R_rpt_negative = R.to_roll_pitch_yaw(negative=True)

    rpy_to_matrix = R_rpy_positive.to_rotation_matrix()
    rpy_negative_to_matrix = R_rpt_negative.to_rotation_matrix()

    assert np.allclose(np.round(rpy_to_matrix, 3), R.matrix)
    assert np.allclose(np.round(rpy_negative_to_matrix, 3), R.matrix)


def test_AxisAngle():
    mat = np.array(
        [
            [0.926, -0.304, 0.226],
            [0.363, 0.881, -0.304],
            [-0.107, 0.363, 0.926],
        ]
    )

    R = RotationMatrix(mat)

    axis_angle = R.to_axis_angle()
    axis_angle_negative = R.to_axis_angle(negative=True)

    axis_angle_to_matrix = axis_angle.to_rotation_matrix()
    axis_angle_negative_to_matrix = axis_angle_negative.to_rotation_matrix()

    assert np.allclose(np.round(axis_angle_to_matrix, 3), R.matrix)
    assert np.allclose(np.round(axis_angle_negative_to_matrix, 3), R.matrix)

    v = np.array([3, 1, 1])
    theta = 30 * pi / 180
    axis = np.array([2, 1, 2])

    test = AxisAngle(theta, axis)
    test_matrix = test.to_rotation_matrix()

    v_rotated = test.rotate_vector(v)

    assert np.allclose(v_rotated, np.array([2.699, 1.667, 0.967]), atol=1e-2)

    v_rotated1 = test_matrix @ v

    assert np.allclose(v_rotated1, v_rotated)


def test_EulerAngleZYZ():
    mat = np.array(
        [
            [-0.5, -0.557, 0.663],
            [0.866, -0.321, 0.383],
            [0, 0.766, 0.643],
        ]
    )

    R = RotationMatrix(mat)

    euler_angles_ZYZ = R.to_euler_angles("ZYZ")

    assert np.allclose(
        np.round(
            [
                euler_angles_ZYZ.angle1 * 180 / pi,
                euler_angles_ZYZ.angle2 * 180 / pi,
                euler_angles_ZYZ.angle3 * 180 / pi,
            ],
            1,
        ),
        [30, 50, 90],
    )

    euler_angles_ZYZ_to_matrix = euler_angles_ZYZ.to_rotation_matrix()
    assert np.allclose(euler_angles_ZYZ_to_matrix, R, atol=1e-2)


def test_Quaternion():
    quat = Quaternion.rotation(60 * pi / 180, [2, 0, 0])

    assert np.allclose(quat.w, cos(60 * pi / 180 / 2), atol=1e-2)
    assert np.allclose(
        quat.axis, sin(60 * pi / 180 / 2) * np.array([1, 0, 0]), atol=1e-2
    )

    quat_inv = quat.inv()
    assert np.allclose(quat_inv.w, cos(60 * pi / 180 / 2), atol=1e-2)
    assert np.allclose(
        quat_inv.axis, -sin(60 * pi / 180 / 2) * np.array([1, 0, 0]), atol=1e-2
    )

    mat = np.array(
        [
            [0.321, -0.117, 0.940],
            [0.683, 0.716, -0.145],
            [-0.656, 0.688, 0.310],
        ]
    )

    R = RotationMatrix(mat)

    quat = R.to_quaternion()

    print(quat)

    quat_to_matrix = quat.to_rotation_matrix()
    print()
    print(R)
    print(quat_to_matrix)
    assert np.allclose(quat_to_matrix, R, atol=1e-2)

    quat_to_axe_angle = quat.to_axis_angle()
    axe_angle_to_matrix = quat_to_axe_angle.to_rotation_matrix()
    print(axe_angle_to_matrix)
    assert np.allclose(axe_angle_to_matrix, R, atol=1e-2)

    quat1 = Quaternion(3, np.array([1, -2, 1]))
    quat2 = Quaternion(2, np.array([-1, 2, 3]))

    assert np.allclose(quat1 * quat2, Quaternion(8, np.array([-9, -2, 11])))

    p = np.array([3, 5, 2])
    theta = 60 * pi / 180
    axis = np.array([2, 0, 0])

    quat3 = Quaternion.rotation(theta, axis)

    p_rotated = quat3.rotate_vector(p)
    print(p_rotated)

    assert np.allclose(p_rotated, np.array([3, 0.768, 5.33]), atol=1e-2)

    p_rotated1 = quat3.to_rotation_matrix() @ p
    print(p_rotated1)
    assert np.allclose(p_rotated1, np.array([3, 0.768, 5.33]), atol=1e-2)
