import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Systemparametre
M = 1.0; m = 0.1; l = 1.0; g = 9.81

# Linearisert system og LQR-gain fra oppg. 3a (Q=I, R=1)
A = np.array([[0,0,1,0],[0,0,0,1],[0,-m*g/M,0,0],[0,(M+m)*g/(M*l),0,0]])
B = np.array([[0],[0],[1/M],[-1/(M*l)]])
Q = np.eye(4); R = np.array([[1.0]])
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
A_cl = A - B @ K

# Ulineaer modell: loser for x_ddot og theta_ddot fra ligning (1) og (2)
def nonlinear_dynamics(t, state):
    x, theta, xd, thetad = state
    F = (-K @ state).item()                 # LQR padrag
    c = np.cos(theta); s = np.sin(theta)
    det = l * (m * c**2 - (M + m))          # Determinant fra 2x2-systemet
    x_dd  = (m*l*c*g*s - l*(F + m*l*s*thetad**2)) / det
    th_dd = (c*(F + m*l*s*thetad**2) - (M+m)*g*s) / det
    return [xd, thetad, x_dd, th_dd]

# Lineaer modell (lukket sloyfe)
def linear_dynamics(t, state):
    return A_cl @ state

t_eval = np.linspace(0, 10, 5000)

# --- 4b: Sammenligning ved 3 grader ---
theta0 = 3 * np.pi / 180
x0 = np.array([0, theta0, 0, 0])
sol_nl = solve_ivp(nonlinear_dynamics, (0,10), x0, t_eval=t_eval, max_step=0.01)
sol_lin = solve_ivp(linear_dynamics, (0,10), x0, t_eval=t_eval)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Oppgave 4b: Lineaer vs ulineaer, theta0 = 3 grader')
labels = ['x [m]', 'theta [deg]', 'dx/dt [m/s]', 'dtheta/dt [deg/s]']
scales = [1, 180/np.pi, 1, 180/np.pi]
titles = ['Vognposisjon', 'Pendelvinkel', 'Vognhastighet', 'Vinkelhastighet']
for i, ax in enumerate(axes.flat):
    ax.plot(sol_lin.t, sol_lin.y[i]*scales[i], label='Lineaer')
    ax.plot(sol_nl.t, sol_nl.y[i]*scales[i], '--', label='Ulineaer')
    ax.set_ylabel(labels[i]); ax.set_xlabel('t [s]')
    ax.set_title(titles[i]); ax.grid(True); ax.legend()
plt.tight_layout(); plt.savefig('oppg4b.png', dpi=150); plt.close()

# --- 4c: Sammenligning for storre vinkler ---
# Stopp-hendelse: avbryt hvis pendelen velter (|theta| > 180 deg)
def theta_limit(t, state):
    return np.abs(state[1]) - np.pi
theta_limit.terminal = True

angles = [10, 30, 45, 60, 65]
fig, axes = plt.subplots(len(angles), 2, figsize=(12, 4*len(angles)))
fig.suptitle('Oppgave 4c: Lineaer vs ulineaer for ulike startvinkler', y=1.0)
for j, deg in enumerate(angles):
    theta0 = deg * np.pi / 180
    x0 = np.array([0, theta0, 0, 0])
    sol_nl = solve_ivp(nonlinear_dynamics, (0,10), x0, t_eval=t_eval,
                       max_step=0.01, events=theta_limit)
    sol_lin = solve_ivp(linear_dynamics, (0,10), x0, t_eval=t_eval)
    for k, (ylabel, title_prefix) in enumerate(
        [('theta [deg]', 'Pendelvinkel'), ('x [m]', 'Vognposisjon')]):
        idx = 1 if k == 0 else 0
        sc = 180/np.pi if k == 0 else 1
        axes[j,k].plot(sol_lin.t, sol_lin.y[idx]*sc, label='Lineaer')
        axes[j,k].plot(sol_nl.t, sol_nl.y[idx]*sc, '--', label='Ulineaer')
        axes[j,k].set_ylabel(ylabel); axes[j,k].set_xlabel('t [s]')
        axes[j,k].set_title(f'{title_prefix}, theta0 = {deg} deg')
        axes[j,k].grid(True); axes[j,k].legend()
plt.tight_layout(); plt.savefig('oppg4c.png', dpi=150); plt.close()