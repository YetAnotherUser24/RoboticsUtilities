from space_representation import (
    RotationMatrix,
    RollPitchYaw,
    EulerAngles,
    TransformationMatrix,
    Quaternion,
)
import numpy as np
from numpy import pi


RotX = RotationMatrix.rotation_x
RotY = RotationMatrix.rotation_y

print(RotX(90 * pi / 180))

print(RotX(90 * pi / 180))
print("------------1----------------")
print("a)")
Ttrasl = TransformationMatrix.translation_pure(traslation_vector=np.array([6, 4, 4]))
Trot = TransformationMatrix.rotation_pure("Z", 90 * pi / 180)

T0_1 = Ttrasl @ Trot
print(np.round(T0_1, 3))

print("b)")
Ttrasl = TransformationMatrix.translation_pure(traslation_vector=np.array([-2, 9, 7]))

T1_2 = Ttrasl
print(np.round(T1_2, 3))

print("c)")
Ttrasl = TransformationMatrix.translation_pure(traslation_vector=np.array([2, 3, -4]))
Trot = TransformationMatrix.rotation_pure(
    "X", 180 * pi / 180
) @ TransformationMatrix.rotation_pure("Y", -90 * pi / 180)

T2_3 = Ttrasl @ Trot
print(np.round(T2_3, 3))

print("d)")
Ttrasl = TransformationMatrix.translation_pure(traslation_vector=np.array([7, 0, 2]))
Trot = TransformationMatrix.rotation_pure(
    "X", 90 * pi / 180
) @ TransformationMatrix.rotation_pure("Z", 180 * pi / 180)

T3_4 = Ttrasl @ Trot
print(np.round(T3_4, 3))

print("e)")
T0_4 = T0_1 @ T1_2 @ T2_3 @ T3_4
print(np.round(T0_4, 3))

print("f)")
T1_4 = T1_2 @ T2_3 @ T3_4
R1_4 = T1_4.rotation_matrix
print(np.round(R1_4, 3))
q1_4 = RotationMatrix(R1_4).to_quaternion()
print(np.round(q1_4, 3))

print()
print("------------2----------------")
print("a)")
p5 = np.array([2, 1, 0])
q5 = Quaternion(0.966, np.array([0, 0, 0.259]))
aa5 = q5.to_axis_angle()
print(aa5.to_str_degrees())

p10 = np.array([3, 3, 0])
q10 = Quaternion(0.866, np.array([0, 0, 0.5]))
aa10 = q10.to_axis_angle()
print(aa10.to_str_degrees())

print("b)")
Ttrasl = TransformationMatrix(traslation_vector=p10)
Trot = TransformationMatrix(rotation_matrix=q10.to_rotation_matrix())

Ti_t10 = Ttrasl @ Trot
print(np.round(Ti_t10, 3))
