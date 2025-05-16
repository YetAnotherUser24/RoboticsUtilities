from symbolic_space_representation import *
import sympy as sp


def rotation_vector_example():
    """Example showing pretty-printed vector rotations"""
    # Create symbolic angle and vector
    theta = sp.Symbol("theta")
    v = sp.Matrix([1, 0, 0])

    # Create rotation matrices
    R_z = SymbolicRotationMatrix.rotation_z(theta)

    print("=== Vector Rotation Example ===")
    # Rotate vector and see pretty-printed output
    v_rotated = R_z.rotate_vector(v)

    # Show simplified result
    print("\nSimplified result:")
    print(sp.pretty(sp.simplify(v_rotated)))


def transformation_example():
    """Example showing pretty-printed transformation matrices"""
    # Create symbolic variables
    x, y, theta = sp.symbols("x y theta")

    print("=== Transformation Matrix Example ===")
    # Create a translation
    trans = SymbolicTransformationMatrix.translation_pure(translation_vector=[x, y, 0])
    print("\nTranslation matrix:")
    print(trans)

    # Create a rotation around Z
    rot = SymbolicTransformationMatrix.rotation_pure("Z", theta)
    print("\nRotation matrix:")
    print(rot)

    # Combine them
    transform = trans @ rot
    print("\nCombined transformation:")
    print(transform)


def quaternion_example():
    """Example showing pretty-printed quaternions"""
    # Create symbolic angle
    theta = sp.Symbol("theta")

    print("=== Quaternion Example ===")
    # Create a quaternion for rotation around Z axis
    q = SymbolicQuaternion.rotation(theta, [0, 0, 1])
    print("\nQuaternion for Z rotation by theta:\n")
    print(q)

    # Show the equivalent rotation matrix
    R = q.to_rotation_matrix()
    print("\nEquivalent rotation matrix:")
    print(R)

    # Convert to axis-angle
    aa = q.to_axis_angle()
    print("\nAs axis-angle:")
    print(aa)


if __name__ == "__main__":
    rotation_vector_example()
    print("\n" + "=" * 50 + "\n")
    transformation_example()
    print("\n" + "=" * 50 + "\n")
    quaternion_example()
