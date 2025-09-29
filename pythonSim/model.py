
from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


alpha = 0.05 # Seebeck Coefficient [V/K]
R = 2 # Peltier Resistance [Ohm]
K = 0.1  # Thermal Conductance [W/K]
C = 20  # Thermal Capacitance [J/K]
G = 0.5  # Ambient Conductance [W/K]
Tinf = 20  # Ambient Temperature [C]

T1 = ca.MX.sym("T1")
T2 = ca.MX.sym("T2")
pwm_value = ca.MX.sym("pwm_value")
V_max = 10

# The duty cycle is the fraction of time the pulse is on
duty_cycle_abs = ca.fabs(pwm_value) / 100.0

# The peak voltage depends on the sign of the PWM input
peak_voltage = ca.sign(pwm_value) * V_max

# The current that flows when the pulse is on
I_peak = (peak_voltage - alpha * (T2 - T1)) / R

# The average current over a full cycle (for Peltier effect)
I_avg = duty_cycle_abs * I_peak

# The mean square of the current (for Joule heating)
I_rms_sq = duty_cycle_abs * (I_peak ** 2)

T1_dot = (1/C) * (K * (T2 - T1) - (alpha * T1 * I_avg) + (0.5 * R * I_rms_sq) + G * (Tinf - T1))
T2_dot = (1/C) * (K * (T1 - T2) + (alpha * T2 * I_avg) + (0.5 * R * I_rms_sq) + G * (Tinf - T2))

x = ca.vertcat(T1, T2)
xdot = ca.MX.sym('xdot', 2)
f_expl = ca.vertcat(T1_dot, T2_dot)
f_impl = xdot - f_expl

model = AcadosModel()
model.name = "Peltier"
model.x = x
model.xdot = xdot
model.u = pwm_value
model.f_impl_expr = f_impl
model.f_expl_expr = f_expl

sim = AcadosSim()
sim.model  = model

Tf = 0.1
nx = sim.model.x.rows()
N_sim = 6000

# set simulation time
sim.solver_options.T = Tf
# set options
sim.solver_options.integrator_type = 'ERK'
sim.solver_options.num_stages = 3
sim.solver_options.num_steps = 3
sim.solver_options.newton_iter = 3  # for implicit integrator
sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"

# create
acados_integrator = AcadosSimSolver(sim)

x0 = np.array([20, 20])
u0 = np.array([-50])
xdot_init = np.zeros((nx,))

simX = np.zeros((N_sim + 1, nx))
simX[0, :] = x0

for i in range(N_sim):
    # Note that xdot is only used if an IRK integrator is used
    u = u0
    if i > 3000:
        u = 50
    simX[i + 1, :] = acados_integrator.simulate(x=simX[i, :], u=u)

tgrid = np.linspace(0, Tf * N_sim, N_sim + 1)

plt.figure(figsize=(8, 5))
plt.plot(tgrid, simX[:, 0], label="T1 (Cold Side)")
plt.plot(tgrid, simX[:, 1], label="T2 (Hot Side)")
plt.xlabel("Time [s]")
plt.ylabel("Temperature [°C]")
plt.title("Peltier Thermal Dynamics (acados simulation)")
plt.legend()
plt.grid(True)
plt.show()


#T1_dot = 1/C * (K * (T2 - T1) + (0.5 * R * I ** 2) - (alpha * T1 * I) + G * (Tinf - T1))
#T2_dot = 1/C * (K * (T1 - T2) + (0.5 * R * I ** 2) + (alpha * T2 * I) + G * (Tinf - T2))

#T1_dot = 1/C * (K * (T2 - T1) + (0.5 * R * I ** 2) - (alpha * T1 * I) + G * (Tinf - T1))
#T2_dot = 1/C * (-K * (T2 - T1) + (0.5 * R * I ** 2) + (alpha * T2 * I) + G * (Tinf - T2))

#T1_dot = 1/C * ((alpha * T1 * I) - (0.5 * R * I ** 2) + K * (T1 - T2) + G * (Tinf - T1))
#T2_dot = 1/C * ((alpha * T2 * I) + (0.5 * R * I ** 2) + K * (T1 - T2) + G * (Tinf - T2))



#T1_dot = 1/C * (K * (T2 - T1) - (0.5 * R * I ** 2) + (alpha * T1 * I) + G * (Tinf - T1))
#T2_dot = 1/C * (K * (T2 - T1) + (0.5 * R * I ** 2) + (alpha * T2 * I) + G * (Tinf - T2))