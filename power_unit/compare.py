import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Define system matrices
A = np.array([[1.1, 0.1], [0, 1]])
B = np.array([[0.1], [1]])

# Define cost matrices
Q = np.eye(2)
R = np.eye(1)

# Horizon length
N = 10

# Define the reference trajectory (constant for simplicity)
x_ref = np.array([10, 0])

def simulate_cvxp(x0, x_ref, num_steps):
    # Initialize history lists
    x_history = [x0]
    u_history = []

    for _ in range(num_steps):
        # Define optimization variables
        x = [cp.Variable(2) for _ in range(N+1)]
        u = [cp.Variable(1) for _ in range(N)]

        # Define the cost function
        cost = 0
        constraints = []

        for k in range(N):
            cost += cp.quad_form(x[k] - x_ref, Q) + cp.quad_form(u[k], R)
            constraints.append(x[k+1] == A @ x[k] + B @ u[k])

        # Initial condition constraint
        constraints.append(x[0] == x_history[-1])

        # Solve the optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        # Get optimal control input
        u_opt = u[0].value

        # Update state
        x_next = A @ x_history[-1] + B @ u_opt
        x_history.append(x_next)
        u_history.append(u_opt)

    return np.array(x_history), np.array(u_history)

# Initial state
x0 = np.array([1, 0])

# Number of simulation steps
num_steps = 20

# Simulate using CVXPY-based MPC
x_history_cvxpy, u_history_cvxpy = simulate_cvxp(x0, x_ref, num_steps)

# Plotting the state trajectory and control inputs
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(x_history_cvxpy[:, 0], label='$x_1$ (CVXPY)')
plt.plot(x_history_cvxpy[:, 1], label='$x_2$ (CVXPY)')
plt.axhline(y=x_ref[0], color='r', linestyle='--', label='$x_{1,ref}$')
plt.axhline(y=x_ref[1], color='g', linestyle='--', label='$x_{2,ref}$')
plt.xlabel('Time step')
plt.ylabel('State')
plt.title('State Trajectory with CVXPY-Based Control')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(u_history_cvxpy, label='Control input (CVXPY)')
plt.xlabel('Time step')
plt.ylabel('Control input')
plt.title('Control Input with CVXPY-Based Control')
plt.legend()

plt.tight_layout()
plt.show()
