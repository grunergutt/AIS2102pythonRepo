import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Systemparametre (nominelle)
M = 1.0; m = 0.1; l = 1.0; g = 9.81
theta_ref = 30 * np.pi / 180

# Foroverkobling (nominell likevektsakselerasjon)
v_ff = g * np.tan(theta_ref)

# Lineariserte koeffisienter rundt theta_ref
# delta_omega_dot = a * delta_theta - b * delta_v
a = g / (l * np.cos(theta_ref))
b = np.cos(theta_ref) / l

# PID-parametre via polplassering: (s+p)^3
# s^3 + b*Kd*s^2 + (b*Kp - a)*s + b*Ki = s^3 + 3p*s^2 + 3p^2*s + p^3
p = 3.0
Kp = (3*p**2 + a) / b
Kd = 3*p / b
Ki = p**3 / b

# Sanne parametre (simulerer parameterusikkerhet)
m_true = 0.12   # 20% avvik
l_true = 0.9    # 10% avvik

def pid_nonlinear(t, state):
    theta, omega, xi = state
    delta_theta = theta - theta_ref
    # PID-regulator med foroverkobling
    delta_v = Kp * delta_theta + Kd * omega + Ki * xi
    v = v_ff + delta_v
    # Ulineaer pendeldynamikk med sanne parametre
    theta_ddot = (g * np.sin(theta) - np.cos(theta) * v) / l_true
    xi_dot = delta_theta
    return [omega, theta_ddot, xi_dot]

# Simuler med theta0 = 20 grader
theta0 = 20 * np.pi / 180
x0 = [theta0, 0, 0]
t_eval = np.linspace(0, 10, 5000)

sol = solve_ivp(pid_nonlinear, (0, 10), x0, t_eval=t_eval, max_step=0.005)

# Beregn padrag
v_arr = np.zeros(len(sol.t))
for i in range(len(sol.t)):
    delta_theta = sol.y[0, i] - theta_ref
    v_arr[i] = v_ff + Kp * delta_theta + Kd * sol.y[1, i] + Ki * sol.y[2, i]

# Vognposisjon via numerisk integrasjon av v = x_ddot
dt = np.diff(sol.t)
x_dot = np.cumsum(np.concatenate([[0], v_arr[:-1] * dt]))
x_pos = np.cumsum(np.concatenate([[0], x_dot[:-1] * dt]))

# Plott
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Oppgave 5e: PID med integralvirkning, theta_ref=30, theta0=20 deg')

axes[0,0].plot(sol.t, sol.y[0]*180/np.pi)
axes[0,0].axhline(y=30, color='r', linestyle='--', label='Referanse')
axes[0,0].set_ylabel('theta [deg]'); axes[0,0].set_xlabel('t [s]')
axes[0,0].set_title('Pendelvinkel'); axes[0,0].grid(True); axes[0,0].legend()

axes[0,1].plot(sol.t, sol.y[1]*180/np.pi)
axes[0,1].set_ylabel('omega [deg/s]'); axes[0,1].set_xlabel('t [s]')
axes[0,1].set_title('Vinkelhastighet'); axes[0,1].grid(True)

axes[1,0].plot(sol.t, v_arr)
axes[1,0].axhline(y=v_ff, color='r', linestyle='--', label='v_ff (nominell)')
axes[1,0].set_ylabel('v [m/s^2]'); axes[1,0].set_xlabel('t [s]')
axes[1,0].set_title('Akselerasjon vogn'); axes[1,0].grid(True); axes[1,0].legend()

axes[1,1].plot(sol.t, x_pos)
axes[1,1].set_ylabel('x [m]'); axes[1,1].set_xlabel('t [s]')
axes[1,1].set_title('Vognposisjon (ubegrenset)'); axes[1,1].grid(True)

plt.tight_layout(); plt.savefig('oppg5e.png', dpi=150); plt.close()