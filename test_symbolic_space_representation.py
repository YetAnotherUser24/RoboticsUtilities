import pytest
import sympy as sp
from sympy import pi, simplify
import numpy as np
from symbolic_space_representation import (
    SymbolicRotationMatrix,
    SymbolicTransformationMatrix,
    SymbolicEulerAnglesZYZ,
    SymbolicRollPitchYaw,
    SymbolicAxisAngle,
    SymbolicQuaternion,
)


def test_symbolic_rotation_matrix_basic():
    # Test initialization
    R = SymbolicRotationMatrix()
    assert R.matrix == sp.eye(3)

    # Test rotation matrices with symbolic angle
    theta = sp.Symbol("theta")
    R_x = SymbolicRotationMatrix.rotation_x(theta)
    R_y = SymbolicRotationMatrix.rotation_y(theta)
    R_z = SymbolicRotationMatrix.rotation_z(theta)

    # Test matrix multiplication
    R_combined = R_x @ R_y @ R_z
    assert isinstance(R_combined, SymbolicRotationMatrix)


def test_symbolic_rotation_matrix_composition():
    # Test composition of rotations with symbolic angles
    alpha, beta, gamma = sp.symbols("alpha beta gamma")

    R_x = SymbolicRotationMatrix.rotation_x(alpha)
    R_y = SymbolicRotationMatrix.rotation_y(beta)
    R_z = SymbolicRotationMatrix.rotation_z(gamma)

    # XYZ composition
    R_xyz = R_x @ R_y @ R_z

    # Verify that the result is still symbolic
    assert all(isinstance(elem, sp.Expr) for elem in R_xyz.matrix)


def test_symbolic_euler_angles():
    # Test Euler angles with symbolic variables
    phi1, phi2, phi3 = sp.symbols("phi1 phi2 phi3")

    # Create ZYZ Euler angles
    euler = SymbolicEulerAnglesZYZ(phi1, phi2, phi3)
    R = euler.to_rotation_matrix()

    # Convert back to Euler angles
    euler_back = R.to_euler_angles("ZYZ")

    # Note: Direct comparison might not work due to multiple solutions
    # Instead, verify that converting back to rotation matrix gives same result
    R_back = euler_back.to_rotation_matrix()

    # The matrices should be equivalent (though expressions might look different)
    diff_matrix = simplify(R.matrix - R_back.matrix)
    assert all(simplify(elem) == 0 for elem in diff_matrix)


def test_symbolic_roll_pitch_yaw():
    # Test Roll-Pitch-Yaw with symbolic angles
    roll, pitch, yaw = sp.symbols("roll pitch yaw")

    rpy = SymbolicRollPitchYaw(roll, pitch, yaw)
    R = rpy.to_rotation_matrix()

    # Convert back to roll-pitch-yaw
    rpy_back = R.to_roll_pitch_yaw()
    R_back = rpy_back.to_rotation_matrix()

    # Verify that the matrices are equivalent
    diff_matrix = simplify(R.matrix - R_back.matrix)
    assert all(simplify(elem) == 0 for elem in diff_matrix)


def test_symbolic_axis_angle():
    # Test axis-angle with symbolic angle
    theta = sp.Symbol("theta")
    axis = [0, 0, 1]  # z-axis

    aa = SymbolicAxisAngle(theta, axis)
    R = aa.to_rotation_matrix()

    # Test vector rotation
    v = sp.Matrix([1, 0, 0])
    v_rotated = aa.rotate_vector(v)

    # For rotation around z-axis, the result should be [cos(theta), sin(theta), 0]
    assert simplify(v_rotated[0] - sp.cos(theta)) == 0
    assert simplify(v_rotated[1] - sp.sin(theta)) == 0
    assert simplify(v_rotated[2]) == 0


def test_symbolic_quaternion():
    # Test quaternion with symbolic angle
    theta = sp.Symbol("theta")

    # Create a quaternion representing rotation around z-axis
    q = SymbolicQuaternion.rotation(theta, [0, 0, 1])

    # Convert to rotation matrix
    R = q.to_rotation_matrix()

    # Test vector rotation - compare effect of rotation instead of angles
    test_vector = sp.Matrix([1, 0, 0])
    v1 = R.matrix @ test_vector
    v2 = q.rotate_vector(test_vector)

    # The rotated vectors should be the same
    diff = simplify(v1 - v2)
    assert all(simplify(x) == 0 for x in diff)

    # Test quaternion multiplication
    q2 = SymbolicQuaternion.rotation(theta / 2, [0, 1, 0])
    q_combined = q * q2
    assert isinstance(q_combined, SymbolicQuaternion)


def test_symbolic_transformation_matrix():
    # Test transformation matrix with symbolic variables
    x, y, theta = sp.symbols("x y theta")

    # Create a transformation with translation and rotation
    trans = SymbolicTransformationMatrix.translation_pure(translation_vector=[x, y, 0])
    rot = SymbolicTransformationMatrix.rotation_pure("Z", theta)

    # Combine transformation
    transform = trans @ rot

    # Check matrix structure
    assert transform.matrix.shape == (4, 4)
    assert transform.matrix[3, 3] == 1

    # Translation should be preserved
    assert transform.translation_vector[0] == x
    assert transform.translation_vector[1] == y
    assert transform.translation_vector[2] == 0


def test_symbolic_numeric_conversion():
    # Test conversion between symbolic and numeric values
    theta = sp.Symbol("theta")
    R_symbolic = SymbolicRotationMatrix.rotation_z(theta)

    # Substitute a numeric value
    R_numeric = R_symbolic.matrix.subs(theta, pi / 4)

    # Compare with expected numeric values
    expected = sp.Matrix(
        [
            [1 / sp.sqrt(2), -1 / sp.sqrt(2), 0],
            [1 / sp.sqrt(2), 1 / sp.sqrt(2), 0],
            [0, 0, 1],
        ]
    )

    diff = simplify(R_numeric - expected)
    assert all(simplify(x) == 0 for x in diff)
