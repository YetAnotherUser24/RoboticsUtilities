from symbolic_space_representation import *
import sympy as sp


def derivative_example():
    """Example showing symbolic derivatives with pretty printing"""
    try:
        # Create symbolic variables
        theta = sp.Symbol("theta")
        omega = sp.Symbol("omega")
        t = sp.Symbol("t")

        print("=== Symbolic Derivatives Example ===")

        # Create a time-varying rotation
        theta_t = omega * t  # Angular position as a function of time
        R = SymbolicRotationMatrix.rotation_z(theta_t)

        print("\nRotation matrix R(t):")
        print(R)

        # Compute derivatives
        print("\nComputing time derivative dR/dt:")
        dR_dt = R.compute_derivative(t)  # Derivative with respect to time

        print("\nComputing angular velocity derivative dR/dω:")
        dR_dw = R.compute_derivative(
            omega
        )  # Derivative with respect to angular velocity

        # Show how we can compose rotations symbolically
        print("\nComposing two rotations:")
        R2 = SymbolicRotationMatrix.rotation_y(theta)
        R_composed = R @ R2
        print("\nComposed rotation matrix:")
        print(R_composed)

        # Compute the derivative of the composed rotation
        print("\nComputing derivative of composed rotation:")
        dR_composed = R_composed.compute_derivative(t)

        return dR_dt, dR_dw, dR_composed

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise


if __name__ == "__main__":
    derivative_example()
