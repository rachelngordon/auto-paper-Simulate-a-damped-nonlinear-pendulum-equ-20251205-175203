# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

def pendulum_deriv(state, b, g=9.81, L=1.0):
    theta, omega = state
    dtheta = omega
    domega = -b * omega - (g / L) * np.sin(theta)
    return np.array([dtheta, domega])

def rk4_step(state, dt, b):
    k1 = pendulum_deriv(state, b)
    k2 = pendulum_deriv(state + 0.5 * dt * k1, b)
    k3 = pendulum_deriv(state + 0.5 * dt * k2, b)
    k4 = pendulum_deriv(state + dt * k3, b)
    return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

def simulate(b, t_max=20.0, dt=0.01, theta0=0.5, omega0=0.0):
    N = int(t_max / dt) + 1
    t = np.linspace(0, t_max, N)
    state = np.array([theta0, omega0])
    theta = np.empty(N)
    omega = np.empty(N)
    for i in range(N):
        theta[i] = state[0]
        omega[i] = state[1]
        state = rk4_step(state, dt, b)
    return t, theta, omega

def exp1():
    b = 0.1
    t, theta, omega = simulate(b)
    # Angle vs time plot
    plt.figure()
    plt.plot(t, theta)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Damped Pendulum Angle vs Time')
    plt.grid(True)
    plt.savefig('angle_vs_time.png')
    plt.close()
    # Phase space plot
    plt.figure()
    plt.plot(theta, omega)
    plt.xlabel('Angle (rad)')
    plt.ylabel('Angular velocity (rad/s)')
    plt.title('Phase Space')
    plt.grid(True)
    plt.savefig('phase_space.png')
    plt.close()
    # Estimate decay time (time to reach 1/e of initial amplitude)
    theta0 = theta[0]
    target = theta0 * np.exp(-1)
    idx = np.where(np.abs(theta) <= target)[0]
    if idx.size > 0:
        decay_time = t[idx[0]]
    else:
        decay_time = np.nan
    return decay_time

def exp2():
    damping_values = [0.0, 0.05, 0.1, 0.2]
    plt.figure()
    for b in damping_values:
        t, theta, _ = simulate(b)
        plt.plot(t, theta, label=f'b={b}')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Angle vs Time for Different Damping Coefficients')
    plt.legend()
    plt.grid(True)
    plt.savefig('damping_vs_angle_time.png')
    plt.close()

def main():
    decay_time = exp1()
    exp2()
    print('Answer:', decay_time)

if __name__ == '__main__':
    main()

