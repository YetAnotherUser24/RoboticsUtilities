import numpy as np
import sympy as sp
from numpy import cos, sin, atan2, sqrt, pi
from numpy import ndarray


class RotationMatrix:
    def __init__(self, matrix=None):
        if matrix is None:
            self.matrix = np.eye(3)
        else:
            self.matrix = matrix

    def __add__(self, other):
        if isinstance(other, ndarray):
            return RotationMatrix(self.matrix + other)

        if isinstance(other, RotationMatrix):
            return RotationMatrix(self.matrix + other.matrix)

        raise TypeError

    def __sub__(self, other):
        if isinstance(other, ndarray):
            return RotationMatrix(self.matrix - other)

        if isinstance(other, RotationMatrix):
            return RotationMatrix(self.matrix - other.matrix)

        raise TypeError

    def __mul__(self, other):
        if isinstance(other, float):
            return RotationMatrix(self.matrix * other)

        if isinstance(other, ndarray):
            return RotationMatrix(self.matrix * other)

        if isinstance(other, RotationMatrix):
            return RotationMatrix(self.matrix * other.matrix)

        raise TypeError

    def __matmul__(self, other):
        if isinstance(other, ndarray):
            return RotationMatrix(self.matrix @ other)

        if isinstance(other, RotationMatrix):
            return RotationMatrix(self.matrix @ other.matrix)

        raise TypeError

    def __truediv__(self, other):
        if isinstance(other, float):
            return RotationMatrix(self.matrix / other)

        if isinstance(other, ndarray):
            return RotationMatrix(self.matrix / other)

        if isinstance(other, RotationMatrix):
            return RotationMatrix(self.matrix / other.matrix)

        raise TypeError

    def __radd__(self, other):
        if isinstance(other, ndarray):
            return RotationMatrix(self.matrix + other)

        if isinstance(other, RotationMatrix):
            return RotationMatrix(self.matrix + other.matrix)

        raise TypeError

    def __rsub__(self, other):
        if isinstance(other, ndarray):
            return RotationMatrix(self.matrix - other)

        if isinstance(other, RotationMatrix):
            return RotationMatrix(self.matrix - other.matrix)

        raise TypeError

    def __rmul__(self, other):
        if isinstance(other, float):
            return RotationMatrix(self.matrix * other)

        if isinstance(other, ndarray):
            return RotationMatrix(self.matrix * other)

        if isinstance(other, RotationMatrix):
            return RotationMatrix(self.matrix * other.matrix)

        raise TypeError

    def __rmatmul__(self, other):
        if isinstance(other, ndarray):
            return RotationMatrix(other @ self.matrix)

        if isinstance(other, RotationMatrix):
            return RotationMatrix(other.matrix @ self.matrix)

        raise TypeError

    def __array__(self):
        return self.matrix

    def __eq__(self, other):
        if isinstance(other, ndarray):
            return np.allclose(self.matrix, other)

        if isinstance(other, RotationMatrix):
            return np.allclose(self.matrix, other.matrix)

        raise TypeError

    def __repr__(self):
        return str(self.matrix)

    def __str__(self):
        return str(self.matrix)

    def __getitem__(self, index):
        return self.matrix[index]

    def __setitem__(self, index, value):
        self.matrix[index] = value

    @classmethod
    def rotation_x(cls, angle):
        rotation_matrix = np.array(
            [
                [1, 0, 0],
                [0, cos(angle), -sin(angle)],
                [0, sin(angle), cos(angle)],
            ]
        )
        return cls(rotation_matrix)

    @classmethod
    def rotation_y(cls, angle):
        rotation_matrix = np.array(
            [
                [cos(angle), 0, sin(angle)],
                [0, 1, 0],
                [-sin(angle), 0, cos(angle)],
            ]
        )
        return cls(rotation_matrix)

    @classmethod
    def rotation_z(cls, angle):
        rotation_matrix = np.array(
            [
                [cos(angle), -sin(angle), 0],
                [sin(angle), cos(angle), 0],
                [0, 0, 1],
            ]
        )
        return cls(rotation_matrix)

    @staticmethod
    def check_rotation_matrix(matrix):
        result = True

        if isinstance(matrix, RotationMatrix):
            matrix = matrix.matrix

        if isinstance(matrix, ndarray):
            if matrix.shape != (3, 3):
                result = False

            if not np.allclose(np.linalg.det(matrix), 1.0):
                result = False

            if not np.allclose(np.linalg.inv(matrix), np.transpose(matrix)):
                result = False

            return result

        raise TypeError

    def is_rotation_matrix(self):
        return RotationMatrix.is_rotation_matrix(self.matrix)

    def inv(self):
        return RotationMatrix(np.linalg.matrix_transpose(self.matrix))

    def to_euler_angles(self, axes, negative=False):
        R = self.matrix
        if axes == "ZYZ":
            phi2 = atan2(sqrt(R[0, 2] ** 2 + R[1, 2] ** 2), R[2, 2])
            if negative:
                phi2 = atan2(-sqrt(R[0, 2] ** 2 + R[1, 2] ** 2), R[2, 2])

            if sin(phi2) == 0:
                raise ValueError("No rotation on axis Y")

            phi1 = atan2(R[1, 2] / sin(phi2), R[0, 2] / sin(phi2))
            phi3 = atan2(R[2, 1] / sin(phi2), -R[2, 0] / sin(phi2))

            return EulerAnglesZYZ(phi1, phi2, phi3)

    def to_roll_pitch_yaw(self, negative=False):
        R = self.matrix

        pitch = atan2(-R[2, 0], sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        if np.allclose(cos(pitch), 0):
            raise ValueError("No rotation on axis Y")

        if negative:
            pitch = atan2(-R[2, 0], -sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))

        row = atan2(R[1, 0] / cos(pitch), R[0, 0] / cos(pitch))
        yaw = atan2(R[2, 1] / cos(pitch), R[2, 2] / cos(pitch))

        return RollPitchYaw(row, pitch, yaw)

    def to_axis_angle(self, negative=False):
        R = self.matrix
        R_vect = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

        theta = atan2(np.linalg.norm(R_vect) * 0.5, (np.trace(R) - 1) * 0.5)

        if sin(theta) == 0:
            raise ValueError("Sin(theta) == 0!")

        u = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (
            2 * sin(theta)
        )

        if negative:
            u = -u
            theta = -theta

        return AxisAngle(theta, u)

    def to_quaternion(self):
        R = self.matrix

        w = sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)

        return Quaternion(w, [x, y, z])


class TransformationMatrix:
    def __init__(self, matrix=None, rotation_matrix=None, traslation_vector=None):
        self.matrix = np.zeros((4, 4))
        self.matrix[3, 3] = 1
        self.rotation_matrix = RotationMatrix(np.eye(3)).matrix
        self.traslation_vector = np.array([0, 0, 0])

        if rotation_matrix is not None:
            if isinstance(rotation_matrix, RotationMatrix):
                self.rotation_matrix = rotation_matrix.matrix
            else:
                self.rotation_matrix = rotation_matrix
        if traslation_vector is not None:
            self.traslation_vector = traslation_vector

        self.matrix[0:3, 0:3] = self.rotation_matrix
        self.matrix[0:3, 3] = self.traslation_vector

        if matrix is not None:
            self.matrix = matrix
            self.rotation_matrix = self.matrix[0:3, 0:3]
            self.traslation_vector = self.matrix[0:3, 3]

    def __array__(self):
        return self.matrix

    def __add__(self, other):
        if isinstance(other, ndarray):
            return TransformationMatrix(matrix=self.matrix + other)

        if isinstance(other, TransformationMatrix):
            return TransformationMatrix(matrix=self.matrix + other.matrix)

        raise TypeError

    def __sub__(self, other):
        if isinstance(other, ndarray):
            return TransformationMatrix(matrix=self.matrix - other)

        if isinstance(other, TransformationMatrix):
            return TransformationMatrix(matrix=self.matrix - other.matrix)

        raise TypeError

    def __mul__(self, other):
        if isinstance(other, ndarray):
            return TransformationMatrix(matrix=self.matrix * other)

        if isinstance(other, TransformationMatrix):
            return TransformationMatrix(matrix=self.matrix * other.matrix)

        raise TypeError

    def __matmul__(self, other):
        if isinstance(other, ndarray):
            return TransformationMatrix(matrix=self.matrix @ other)

        if isinstance(other, TransformationMatrix):
            return TransformationMatrix(matrix=self.matrix @ other.matrix)

        raise TypeError

    def __truediv__(self, other):
        if isinstance(other, ndarray):
            return TransformationMatrix(self.matrix / other)

        if isinstance(other, TransformationMatrix):
            return TransformationMatrix(self.matrix / other.matrix)

        raise TypeError

    def __eq__(self, other):
        if isinstance(other, ndarray):
            return np.allclose(self.matrix, other)

        if isinstance(other, TransformationMatrix):
            return np.allclose(self.matrix, other.matrix)

        raise TypeError

    def __repr__(self):
        return str(self.matrix)

    def __str__(self):
        return str(self.matrix)

    def __getitem__(self, index):
        return self.matrix[index]

    def __setitem__(self, index, value):
        self.matrix[index] = value

    @classmethod
    def rotation_pure(cls, axis, angle, rotation_matrix=None):
        if rotation_matrix is not None:
            return cls(rotation_matrix=rotation_matrix)

        if axis == "X":
            return cls(rotation_matrix=RotationMatrix.rotation_x(angle))
        elif axis == "Y":
            return cls(rotation_matrix=RotationMatrix.rotation_y(angle))
        elif axis == "Z":
            return cls(rotation_matrix=RotationMatrix.rotation_z(angle))

        raise ValueError("Axis must be one of the following: X,Y,Z")

    @classmethod
    def translation_pure(cls, axis=None, distance=None, traslation_vector=None):
        if traslation_vector is not None:
            return cls(traslation_vector=traslation_vector)

        if axis == "X":
            return cls(traslation_vector=np.array([distance, 0, 0]))
        elif axis == "Y":
            return cls(traslation_vector=np.array([0, distance, 0]))
        elif axis == "Z":
            return cls(traslation_vector=np.array([0, 0, distance]))

        raise ValueError("Axis must be one of the following: X,Y,Z")


class EulerAngles:
    def __init__(self, axes: str, angles: list[float]):
        axis1 = axes[0].upper()
        axis2 = axes[1].upper()
        axis3 = axes[2].upper()

        if axis1 not in ["X", "Y", "Z"]:
            raise ValueError("Axis1 must be one of the following: X,Y,Z")

        if axis2 not in ["X", "Y", "Z"]:
            raise ValueError("Axis2 must be one of the following: X,Y,Z")

        if axis3 not in ["X", "Y", "Z"]:
            raise ValueError("Axis3 must be one of the following: X,Y,Z")

        self.axis1: str = axis1
        self.axis2: str = axis2
        self.axis3: str = axis3

        self.angle1 = angles[0]
        self.angle2 = angles[1]
        self.angle3 = angles[2]

        self.return_sub_class(self)

    def return_sub_class(self):
        if [self.axis1, self.axis2, self.axis3] == "ZYZ":
            return EulerAnglesZYZ(self.angle1, self.angle2, self.angle3)
        if [self.axis1, self.axis2, self.axis3] == "ZYX":
            return RollPitchYaw(self.angle1, self.angle2, self.angle3)


class RollPitchYaw(EulerAngles):
    def __init__(self, angle1, angle2, angle3):
        self.angle1 = angle1  # Roll angle
        self.angle2 = angle2  # Pitch angle
        self.angle3 = angle3  # Yaw angle

    def to_rotation_matrix(self):
        row = self.angle1
        pitch = self.angle2
        yaw = self.angle3

        rotation_matrix = np.array(
            [
                [
                    cos(row) * cos(pitch),
                    cos(row) * sin(pitch) * sin(yaw) - sin(row) * cos(yaw),
                    cos(row) * sin(pitch) * cos(yaw) + sin(row) * sin(yaw),
                ],
                [
                    sin(row) * cos(pitch),
                    sin(row) * sin(pitch) * sin(yaw) + cos(row) * cos(yaw),
                    sin(row) * sin(pitch) * cos(yaw) - cos(row) * sin(yaw),
                ],
                [
                    -sin(pitch),
                    cos(pitch) * sin(yaw),
                    cos(pitch) * cos(yaw),
                ],
            ]
        )
        return RotationMatrix(rotation_matrix)

    def to_str_degrees(self):
        return f"RollPitchYaw({self.angle1 * 180 / pi}, {self.angle2 * 180 / pi}, {self.angle3 * 180 / pi})"

    def __str__(self):
        return f"RollPitchYaw({self.angle1}, {self.angle2}, {self.angle3})"

    def __repr__(self):
        return f"RollPitchYaw({self.angle1}, {self.angle2}, {self.angle3})"


class EulerAnglesZYZ(EulerAngles):
    def __init__(self, angle1, angle2, angle3):
        self.angle1 = angle1
        self.angle2 = angle2
        self.angle3 = angle3

    def __str__(self):
        return f"EulerAnglesZYZ({self.angle1}, {self.angle2}, {self.angle3})"

    def to_rotation_matrix(self):
        phi1 = self.angle1
        phi2 = self.angle2
        phi3 = self.angle3

        rotation_matrix = np.array(
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
                [
                    -sin(phi2) * cos(phi3),
                    sin(phi2) * sin(phi3),
                    cos(phi2),
                ],
            ]
        )

        return RotationMatrix(rotation_matrix)


class AxisAngle:
    def __init__(self, angle, axis):
        self.axis = axis
        self.angle = angle

    def to_str_degrees(self, round=3):
        return f"AxisAngle({np.round(self.angle * 180 / pi, round)}, {np.round(self.axis, round)})"

    def __str__(self):
        return f"AxisAngle({self.angle}, {self.axis})"

    def __repr__(self):
        return f"AxisAngle({self.angle}, {self.axis})"

    def __array__(self):
        return np.array([self.angle, self.axis[0], self.axis[1], self.axis[2]])

    def to_rotation_matrix(self):
        theta = self.angle
        u = self.axis / np.linalg.norm(self.axis)
        u_anti_simmetric = np.array(
            [
                [0, -u[2], u[1]],
                [u[2], 0, -u[0]],
                [-u[1], u[0], 0],
            ]
        )

        rotation_matrix = (
            np.eye(3)
            + u_anti_simmetric * sin(theta)
            + u_anti_simmetric @ u_anti_simmetric * (1 - cos(theta))
        )

        return RotationMatrix(rotation_matrix)

    def rotate_vector(self, v):
        theta = self.angle
        u = self.axis / np.linalg.norm(self.axis)
        v_rotated = (
            v * cos(theta)
            + np.cross(u, v) * sin(theta)
            + u * (np.dot(u, v)) * (1 - cos(theta))
        )

        return v_rotated


class Quaternion:
    def __init__(self, w=None, axis=None):
        # Constructor principal: Quaternion(w, x, y, z)
        if w is None or axis is None:
            w = 1
            axis = [0, 0, 0]

        self.w = w
        self.axis = axis

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            result_w = self.w * other.w - np.transpose(self.axis) @ other.axis
            result_axis = (
                self.w * other.axis
                + other.w * self.axis
                + np.cross(self.axis, other.axis)
            )
            return Quaternion(result_w, result_axis)

        raise TypeError

    def to_str_degree(self):
        return f"Quaternion({self.w * 180 / pi}, {self.axis})"

    def __str__(self):
        return f"Quaternion({self.w}, {self.axis})"

    def __repr__(self):
        return f"Quaternion({self.w}, {self.axis})"

    def __array__(self):
        return np.array([self.w, self.axis[0], self.axis[1], self.axis[2]])

    def inv(self):
        return Quaternion(self.w, self.axis * -1)

    def identity(self):
        return Quaternion(1, [0, 0, 0])

    @classmethod
    def rotation(cls, w, axis):
        u = axis / np.linalg.norm(axis)
        return Quaternion(cos(w / 2), u * sin(w / 2))

    def to_rotation_matrix(self):
        w = self.w
        [x, y, z] = self.axis

        rotation_matrix = np.array(
            [
                [2 * (w**2 + x**2) - 1, 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 2 * (w**2 + y**2) - 1, 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 2 * (w**2 + z**2) - 1],
            ]
        )

        return RotationMatrix(rotation_matrix)

    def to_axis_angle(self):
        w = self.w
        axis = self.axis

        theta = 2 * atan2(np.linalg.norm(axis), w)
        u = axis / np.linalg.norm(axis)

        return AxisAngle(theta, u)

    def rotate_vector(self, v):
        v_rotated = self * Quaternion(w=0, axis=v) * self.inv()
        v_rotated = v_rotated.axis

        return v_rotated
