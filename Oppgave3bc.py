import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Systemparametre
M = 1.0    # Masse vogn [kg]
m = 0.1    # Masse pendel [kg]
l = 1.0    # Lengde stag [m]
g = 9.81   # Tyngdeakselerasjon [m/s^2]

# Tilstandsrom-matriser fra linearisering (oppg. 2)
# Tilstand: x = [x, theta, x_dot, theta_dot]^T, padrag: u = F
A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, -m*g/M, 0, 0],
    [0, (M+m)*g/(M*l), 0, 0]
])

B = np.array([[0], [0], [1/M], [-1/(M*l)]])

# --- 3a) LQR med Q = I, R = 1 ---
Q_a = np.eye(4)
R_a = np.array([[1.0]])

# Los algebraisk Riccati-ligning og beregn gain
P_a = solve_continuous_are(A, B, Q_a, R_a)
K_a = np.linalg.inv(R_a) @ B.T @ P_a
print("3a) K =", np.round(K_a.flatten(), 4))

# --- 3b) Simulering med theta0 = 30 grader ---
A_cl_a = A - B @ K_a                          # Lukket sloyfe A-matrise
theta0 = 30 * np.pi / 180                     # 30 grader i radianer
x0 = np.array([0, theta0, 0, 0])              # Initialtilstand

t_eval = np.linspace(0, 10, 5000)
sol_a = solve_ivp(lambda t, x: A_cl_a @ x, (0, 10), x0, t_eval=t_eval)

# Beregn padrag u = -Kx for hvert tidssteg
u_a = np.array([-K_a @ sol_a.y[:, i] for i in range(len(sol_a.t))]).flatten()

# Plott 3b
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle('Oppgave 3b: LQR med Q=I, R=1, theta0=30 grader', fontsize=14)

axes[0,0].plot(sol_a.t, sol_a.y[0])
axes[0,0].set_ylabel('x [m]'); axes[0,0].set_xlabel('t [s]')
axes[0,0].set_title('Vognposisjon'); axes[0,0].grid(True)

axes[0,1].plot(sol_a.t, sol_a.y[1] * 180/np.pi)
axes[0,1].set_ylabel('theta [deg]'); axes[0,1].set_xlabel('t [s]')
axes[0,1].set_title('Pendelvinkel'); axes[0,1].grid(True)

axes[1,0].plot(sol_a.t, sol_a.y[2])
axes[1,0].set_ylabel('dx/dt [m/s]'); axes[1,0].set_xlabel('t [s]')
axes[1,0].set_title('Vognhastighet'); axes[1,0].grid(True)

axes[1,1].plot(sol_a.t, sol_a.y[3] * 180/np.pi)
axes[1,1].set_ylabel('dtheta/dt [deg/s]'); axes[1,1].set_xlabel('t [s]')
axes[1,1].set_title('Vinkelhastighet'); axes[1,1].grid(True)

axes[2,0].plot(sol_a.t, u_a)
axes[2,0].set_ylabel('F [N]'); axes[2,0].set_xlabel('t [s]')
axes[2,0].set_title('Paadrag'); axes[2,0].grid(True)

axes[2,1].axis('off')
plt.tight_layout()
plt.savefig('oppg3b.png', dpi=150)
plt.close()

# --- 3c) Tunet LQR for aa oppfylle designkrav ---
# Hoy vekt paa theta (10000) gir rask innsving
# Moderat vekt paa x (30) begrenser vognens utslag
Q_c = np.diag([30, 10000, 1, 1])
R_c = np.array([[1.0]])

P_c = solve_continuous_are(A, B, Q_c, R_c)
K_c = np.linalg.inv(R_c) @ B.T @ P_c
print("3c) K =", np.round(K_c.flatten(), 4))

# Simuler lukket sloyfe med tunet regulator
A_cl_c = A - B @ K_c
sol_c = solve_ivp(lambda t, x: A_cl_c @ x, (0, 10), x0, t_eval=t_eval)
u_c = np.array([-K_c @ sol_c.y[:, i] for i in range(len(sol_c.t))]).flatten()

# Beregn ytelsesmaal
theta_deg = sol_c.y[1] * 180/np.pi
threshold = 0.02 * 30  # 2% av 30 grader
settled = np.where(np.abs(theta_deg) > threshold)[0]
ts = sol_c.t[settled[-1]] if len(settled) > 0 else 0

print(f"  Innsvingningstid theta: {ts:.2f} s")
print(f"  Max |x|: {np.max(np.abs(sol_c.y[0])):.3f} m")
print(f"  Max |F|: {np.max(np.abs(u_c)):.1f} N")

# Plott 3c
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle('Oppgave 3c: Tunet LQR, Q=diag(30,10000,1,1), R=1', fontsize=14)

axes[0,0].plot(sol_c.t, sol_c.y[0])
axes[0,0].set_ylabel('x [m]'); axes[0,0].set_xlabel('t [s]')
axes[0,0].set_title('Vognposisjon'); axes[0,0].grid(True)

axes[0,1].plot(sol_c.t, sol_c.y[1] * 180/np.pi)
axes[0,1].set_ylabel('theta [deg]'); axes[0,1].set_xlabel('t [s]')
axes[0,1].set_title('Pendelvinkel'); axes[0,1].grid(True)

axes[1,0].plot(sol_c.t, sol_c.y[2])
axes[1,0].set_ylabel('dx/dt [m/s]'); axes[1,0].set_xlabel('t [s]')
axes[1,0].set_title('Vognhastighet'); axes[1,0].grid(True)

axes[1,1].plot(sol_c.t, sol_c.y[3] * 180/np.pi)
axes[1,1].set_ylabel('dtheta/dt [deg/s]'); axes[1,1].set_xlabel('t [s]')
axes[1,1].set_title('Vinkelhastighet'); axes[1,1].grid(True)

axes[2,0].plot(sol_c.t, u_c)
axes[2,0].set_ylabel('F [N]'); axes[2,0].set_xlabel('t [s]')
axes[2,0].set_title('Paadrag'); axes[2,0].grid(True)

axes[2,1].axis('off')
plt.tight_layout()
plt.savefig('oppg3c.png', dpi=150)
plt.close()
