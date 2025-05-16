from symbolic_space_representation import (
    SymbolicEulerAnglesZYZ,
    SymbolicQuaternion,
    SymbolicRotationMatrix,
    SymbolicTransformationMatrix,
)
import sympy as sp


def basic_rotations_example():
    """Example of basic rotation matrix operations"""
    # Create symbolic variables
    theta = sp.Symbol("theta")
    phi = sp.Symbol("phi")
    psi = sp.Symbol("psi")

    # Create rotation matrices
    R_z = SymbolicRotationMatrix.rotation_z(theta)
    R_y = SymbolicRotationMatrix.rotation_y(phi)
    R_x = SymbolicRotationMatrix.rotation_x(psi)

    print("Basic rotation matrices:")
    print("\nRotation around Z:")
    print(R_z)
    print("\nRotation around Y:")
    print(R_y)
    print("\nRotation around X:")
    print(R_x)

    # Combine rotations (ZYX Euler sequence)
    R_combined = R_z @ R_y @ R_x
    print("\nCombined rotation (ZYX sequence):")
    print(R_combined)


def euler_angles_example():
    """Example of Euler angles conversions"""
    # Create symbolic variables
    phi1, phi2, phi3 = sp.symbols("phi1 phi2 phi3")

    # Create ZYZ Euler angles and convert to rotation matrix
    euler = SymbolicEulerAnglesZYZ(phi1, phi2, phi3)
    R = euler.to_rotation_matrix()

    print("ZYZ Euler angles to rotation matrix:")
    print(R)

    # Convert back to Euler angles
    euler_back = R.to_euler_angles("ZYZ")
    print("\nEuler angles extracted from matrix:")
    print(f"phi1 = {euler_back.angle1}")
    print(f"phi2 = {euler_back.angle2}")
    print(f"phi3 = {euler_back.angle3}")


def quaternion_example():
    """Example of quaternion operations"""
    # Create symbolic angle
    theta = sp.Symbol("theta")

    # Create quaternion for rotation around Z axis
    q = SymbolicQuaternion.rotation(theta, [0, 0, 1])
    print("Quaternion for Z-axis rotation:")
    print(f"w = {q.w}")
    print(f"axis = {q.axis}")

    # Convert to rotation matrix
    R = q.to_rotation_matrix()
    print("\nCorresponding rotation matrix:")
    print(R)

    # Rotate a vector
    v = sp.Matrix([1, 0, 0])
    v_rotated = q.rotate_vector(v)
    print("\nRotating vector [1, 0, 0]:")
    print(sp.simplify(v_rotated))


def transformation_matrix_example():
    """Example of transformation matrix operations"""
    # Create symbolic variables
    x, y, theta = sp.symbols("x y theta")

    # Create a translation
    trans = SymbolicTransformationMatrix.translation_pure(translation_vector=[x, y, 0])
    print("Pure translation matrix:")
    print(trans)

    # Create a rotation
    rot = SymbolicTransformationMatrix.rotation_pure("Z", theta)
    print("\nPure rotation matrix:")
    print(rot)

    # Combine transformation
    transform = trans @ rot
    print("\nCombined transformation matrix:")
    print(transform)


if __name__ == "__main__":
    print("=== Basic Rotations Example ===")
    basic_rotations_example()
    print("\n=== Euler Angles Example ===")
    euler_angles_example()
    print("\n=== Quaternion Example ===")
    quaternion_example()
    print("\n=== Transformation Matrix Example ===")
    transformation_matrix_example()
