import sympy as sp
from sympy import cos, sin, atan2, sqrt
import numpy as np


class SymbolicRotationMatrix:
    """A symbolic 3x3 rotation matrix class using SymPy for symbolic computations.

    This class represents rotation matrices with symbolic variables, allowing for:
    - Symbolic matrix operations
    - Conversion between different rotation representations
    - Analytical computations of rotations

    Attributes:
        matrix (sp.Matrix): The 3x3 SymPy matrix representing the rotation

    Examples:
        >>> theta = sp.Symbol('theta')
        >>> R_z = SymbolicRotationMatrix.rotation_z(theta)  # Create z-rotation
        >>> R_z  # Pretty-printed matrix
        Matrix([
        [cos(θ), -sin(θ), 0],
        [sin(θ),  cos(θ), 0],
        [    0,       0,  1]])
    """

    def __init__(self, matrix=None):
        if matrix is None:
            self.matrix = sp.eye(3)
        else:
            self.matrix = matrix if isinstance(matrix, sp.Matrix) else sp.Matrix(matrix)

    def __mul__(self, other):
        if isinstance(other, (int, float, sp.Symbol)):
            return SymbolicRotationMatrix(self.matrix * other)

        if isinstance(other, (np.ndarray, list)):
            return SymbolicRotationMatrix(self.matrix * sp.Matrix(other))

        if isinstance(other, SymbolicRotationMatrix):
            return SymbolicRotationMatrix(self.matrix * other.matrix)

        raise TypeError

    def __matmul__(self, other):
        if isinstance(other, (np.ndarray, list)):
            return SymbolicRotationMatrix(self.matrix @ sp.Matrix(other))

        if isinstance(other, SymbolicRotationMatrix):
            return SymbolicRotationMatrix(self.matrix @ other.matrix)

        raise TypeError

    def __str__(self):
        return sp.pretty(self.matrix)

    def __repr__(self):
        return sp.pretty(self.matrix)

    @classmethod
    def rotation_x(cls, angle):
        rotation_matrix = sp.Matrix(
            [[1, 0, 0], [0, cos(angle), -sin(angle)], [0, sin(angle), cos(angle)]]
        )
        return cls(rotation_matrix)

    @classmethod
    def rotation_y(cls, angle):
        rotation_matrix = sp.Matrix(
            [[cos(angle), 0, sin(angle)], [0, 1, 0], [-sin(angle), 0, cos(angle)]]
        )
        return cls(rotation_matrix)

    @classmethod
    def rotation_z(cls, angle):
        rotation_matrix = sp.Matrix(
            [[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]]
        )
        return cls(rotation_matrix)

    def inv(self):
        return SymbolicRotationMatrix(self.matrix.transpose())

    def to_euler_angles(self, axes):
        R = self.matrix
        if axes == "ZYZ":
            phi2 = atan2(sp.sqrt(R[0, 2] ** 2 + R[1, 2] ** 2), R[2, 2])
            phi1 = atan2(R[1, 2] / sin(phi2), R[0, 2] / sin(phi2))
            phi3 = atan2(R[2, 1] / sin(phi2), -R[2, 0] / sin(phi2))
            return SymbolicEulerAnglesZYZ(phi1, phi2, phi3)

    def to_roll_pitch_yaw(self):
        R = self.matrix
        pitch = atan2(-R[2, 0], sp.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        roll = atan2(R[1, 0] / cos(pitch), R[0, 0] / cos(pitch))
        yaw = atan2(R[2, 1] / cos(pitch), R[2, 2] / cos(pitch))
        return SymbolicRollPitchYaw(roll, pitch, yaw)

    def compute_derivative(self, var):
        """Compute the derivative of the rotation matrix with respect to a variable.

        Args:
            var (sp.Symbol): The variable to differentiate with respect to

        Returns:
            SymbolicRotationMatrix: The derivative matrix wrapped in a SymbolicRotationMatrix
        """
        derivative = self.matrix.diff(var)
        print(f"\nDerivative with respect to {sp.pretty(var)}:")
        print(sp.pretty(derivative))
        return SymbolicRotationMatrix(derivative)

    def rotate_vector(self, vector):
        """Rotate a vector using this rotation matrix with pretty output.

        Args:
            vector (list or sp.Matrix): The vector to rotate

        Returns:
            sp.Matrix: The rotated vector
        """
        v = sp.Matrix(vector) if not isinstance(vector, sp.Matrix) else vector
        result = self.matrix @ v
        print(f"\nRotating vector:\n{sp.pretty(v)}")
        print(f"Result:\n{sp.pretty(result)}")
        return result


class SymbolicTransformationMatrix:
    """A symbolic 4x4 homogeneous transformation matrix using SymPy.

    This class represents homogeneous transformations with symbolic variables, combining:
    - 3x3 rotation matrix
    - 3x1 translation vector

    Attributes:
        matrix (sp.Matrix): The 4x4 SymPy matrix representing the transformation
        rotation_matrix (sp.Matrix): The 3x3 rotation part
        translation_vector (sp.Matrix): The 3x1 translation part

    Examples:
        >>> x, theta = sp.symbols('x theta')
        >>> T = SymbolicTransformationMatrix.translation_pure(translation_vector=[x, 0, 0])
        >>> R = SymbolicTransformationMatrix.rotation_pure('Z', theta)
        >>> T_combined = T @ R  # Combine translation and rotation
    """

    def __init__(self, matrix=None, rotation_matrix=None, translation_vector=None):
        self.matrix = sp.zeros(4, 4)
        self.matrix[3, 3] = 1
        self.rotation_matrix = SymbolicRotationMatrix().matrix
        self.translation_vector = sp.Matrix([0, 0, 0])

        if rotation_matrix is not None:
            if isinstance(rotation_matrix, SymbolicRotationMatrix):
                self.rotation_matrix = rotation_matrix.matrix
            else:
                self.rotation_matrix = sp.Matrix(rotation_matrix)

        if translation_vector is not None:
            self.translation_vector = sp.Matrix(translation_vector)

        self.matrix[0:3, 0:3] = self.rotation_matrix
        self.matrix[0:3, 3] = self.translation_vector

        if matrix is not None:
            self.matrix = sp.Matrix(matrix)
            self.rotation_matrix = self.matrix[0:3, 0:3]
            self.translation_vector = self.matrix[0:3, 3]

    def __str__(self):
        return sp.pretty(self.matrix)

    def __repr__(self):
        return sp.pretty(self.matrix)

    @classmethod
    def rotation_pure(cls, axis, angle):
        if axis == "X":
            return cls(rotation_matrix=SymbolicRotationMatrix.rotation_x(angle))
        elif axis == "Y":
            return cls(rotation_matrix=SymbolicRotationMatrix.rotation_y(angle))
        elif axis == "Z":
            return cls(rotation_matrix=SymbolicRotationMatrix.rotation_z(angle))
        raise ValueError("Axis must be one of the following: X,Y,Z")

    @classmethod
    def translation_pure(cls, axis=None, distance=None, translation_vector=None):
        if translation_vector is not None:
            return cls(translation_vector=translation_vector)

        if axis == "X":
            return cls(translation_vector=[distance, 0, 0])
        elif axis == "Y":
            return cls(translation_vector=[0, distance, 0])
        elif axis == "Z":
            return cls(translation_vector=[0, 0, distance])
        raise ValueError("Axis must be one of the following: X,Y,Z")

    def update_translation_vector(self):
        """Update translation vector from matrix"""
        self.translation_vector = self.matrix[0:3, 3]
        return self.translation_vector

    def __matmul__(self, other):
        if isinstance(other, SymbolicTransformationMatrix):
            result = self.matrix @ other.matrix
            tm = SymbolicTransformationMatrix(matrix=result)
            tm.update_translation_vector()
            return tm
        raise TypeError(
            "Can only multiply SymbolicTransformationMatrix with another SymbolicTransformationMatrix"
        )

    def __mul__(self, other):
        if isinstance(other, (int, float, sp.Symbol)):
            return SymbolicTransformationMatrix(matrix=self.matrix * other)
        raise TypeError("Can only multiply SymbolicTransformationMatrix with scalars")


class SymbolicEulerAnglesZYZ:
    """Symbolic representation of ZYZ Euler angles.

    Represents rotation as three successive rotations:
    1. Rotation around Z axis by angle1 (phi1)
    2. Rotation around Y axis by angle2 (phi2)
    3. Rotation around Z axis by angle3 (phi3)

    Attributes:
        angle1 (sp.Expr): First rotation angle (around Z)
        angle2 (sp.Expr): Second rotation angle (around Y)
        angle3 (sp.Expr): Third rotation angle (around Z)

    Examples:
        >>> phi1, phi2, phi3 = sp.symbols('phi1 phi2 phi3')
        >>> euler = SymbolicEulerAnglesZYZ(phi1, phi2, phi3)
        >>> R = euler.to_rotation_matrix()
    """

    def __init__(self, angle1, angle2, angle3):
        self.angle1 = angle1
        self.angle2 = angle2
        self.angle3 = angle3

    def __str__(self):
        return f"SymbolicEulerAnglesZYZ({self.angle1}, {self.angle2}, {self.angle3})"

    def to_rotation_matrix(self):
        phi1 = self.angle1
        phi2 = self.angle2
        phi3 = self.angle3

        rotation_matrix = sp.Matrix(
            [
                [
                    cos(phi1) * cos(phi2) * cos(phi3) - sin(phi1) * sin(phi3),
                    -sin(phi1) * cos(phi3) - cos(phi1) * cos(phi2) * sin(phi3),
                    cos(phi1) * sin(phi2),
                ],
                [
                    sin(phi1) * cos(phi2) * cos(phi3) + cos(phi1) * sin(phi3),
                    cos(phi1) * cos(phi3) - sin(phi1) * cos(phi2) * sin(phi3),
                    sin(phi1) * sin(phi2),
                ],
                [-sin(phi2) * cos(phi3), sin(phi2) * sin(phi3), cos(phi2)],
            ]
        )

        return SymbolicRotationMatrix(rotation_matrix)


class SymbolicRollPitchYaw:
    def __init__(self, angle1, angle2, angle3):
        self.angle1 = angle1  # Roll angle
        self.angle2 = angle2  # Pitch angle
        self.angle3 = angle3  # Yaw angle

    def __str__(self):
        return f"SymbolicRollPitchYaw({self.angle1}, {self.angle2}, {self.angle3})"

    def to_rotation_matrix(self):
        roll = self.angle1
        pitch = self.angle2
        yaw = self.angle3

        rotation_matrix = sp.Matrix(
            [
                [
                    cos(roll) * cos(pitch),
                    cos(roll) * sin(pitch) * sin(yaw) - sin(roll) * cos(yaw),
                    cos(roll) * sin(pitch) * cos(yaw) + sin(roll) * sin(yaw),
                ],
                [
                    sin(roll) * cos(pitch),
                    sin(roll) * sin(pitch) * sin(yaw) + cos(roll) * cos(yaw),
                    sin(roll) * sin(pitch) * cos(yaw) - cos(roll) * sin(yaw),
                ],
                [-sin(pitch), cos(pitch) * sin(yaw), cos(pitch) * cos(yaw)],
            ]
        )

        return SymbolicRotationMatrix(rotation_matrix)


class SymbolicAxisAngle:
    def __init__(self, angle, axis):
        self.angle = angle
        self.axis = sp.Matrix(axis)

    def __str__(self):
        return f"Rotation by angle\n{sp.pretty(self.angle)}\naround axis\n{sp.pretty(self.axis)}"

    def to_rotation_matrix(self):
        theta = self.angle
        u = self.axis / self.axis.norm()
        u_anti_symmetric = sp.Matrix(
            [[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]
        )

        rotation_matrix = (
            sp.eye(3)
            + u_anti_symmetric * sin(theta)
            + u_anti_symmetric * u_anti_symmetric * (1 - cos(theta))
        )

        return SymbolicRotationMatrix(rotation_matrix)

    def rotate_vector(self, v):
        theta = self.angle
        u = self.axis / self.axis.norm()
        v = sp.Matrix(v)

        v_rotated = (
            v * cos(theta) + u.cross(v) * sin(theta) + u * (u.dot(v)) * (1 - cos(theta))
        )

        return v_rotated


class SymbolicQuaternion:
    """Symbolic quaternion representation for 3D rotations.

    Represents rotation using a quaternion with symbolic components:
    q = w + xi + yj + zk

    Attributes:
        w (sp.Expr): Scalar part of the quaternion
        axis (sp.Matrix): Vector part [x, y, z]

    Examples:
        >>> theta = sp.Symbol('theta')
        >>> q = SymbolicQuaternion.rotation(theta, [0, 0, 1])  # Z-axis rotation
        >>> R = q.to_rotation_matrix()
    """

    def __init__(self, w=None, axis=None):
        if w is None or axis is None:
            self.w = 1
            self.axis = sp.Matrix([0, 0, 0])
        else:
            self.w = w
            self.axis = sp.Matrix(axis)

    def __mul__(self, other):
        if isinstance(other, SymbolicQuaternion):
            result_w = self.w * other.w - self.axis.dot(other.axis)
            result_axis = (
                self.w * other.axis + other.w * self.axis + self.axis.cross(other.axis)
            )
            return SymbolicQuaternion(result_w, result_axis)
        raise TypeError

    def __str__(self):
        x, y, z = self.axis
        i, j, k = sp.symbols("i j k")
        return f"Quaternion:\n{sp.pretty(self.w + (x * i + y * j + z * k))}"

    def inv(self):
        return SymbolicQuaternion(self.w, -self.axis)

    @classmethod
    def rotation(cls, w, axis):
        axis = sp.Matrix(axis)
        u = axis / axis.norm()
        return cls(cos(w / 2), u * sin(w / 2))

    def to_rotation_matrix(self):
        w = self.w
        x, y, z = self.axis

        rotation_matrix = sp.Matrix(
            [
                [2 * (w**2 + x**2) - 1, 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 2 * (w**2 + y**2) - 1, 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 2 * (w**2 + z**2) - 1],
            ]
        )

        return SymbolicRotationMatrix(rotation_matrix)

    def to_axis_angle(self):
        w = self.w
        axis = self.axis
        theta = 2 * atan2(axis.norm(), w)
        u = axis / axis.norm()
        return SymbolicAxisAngle(theta, u)

    def rotate_vector(self, v):
        v_quat = SymbolicQuaternion(0, v)
        v_rotated = self * v_quat * self.inv()
        return v_rotated.axis
