import numpy as np
from matplotlib import pyplot as plt


def peltier_dyn(T: np.ndarray, I_avg: float) -> np.ndarray:
    dTdt = np.zeros(2)
    T1 = T[0]
    T2 = T[1]
    dTdt[0] = (1 / C) * (K * (T2 - T1) - (alpha * T1 * I_avg) + (0.5 * R * I_avg ** 2) + G * (Tinf - T1))
    dTdt[1] = (1 / C) * (K * (T1 - T2) + (alpha * T2 * I_avg) + (0.5 * R * I_avg ** 2) + G * (Tinf - T2))
    return dTdt


def sensor_dyn(T_s: float, T_p: float):
    dTsdt = (1 / tau) * (T_p - T_s)
    return dTsdt


def rk4_peltier(state: np.ndarray, u: float, dt: float):
    k1 = peltier_dyn(state, u)
    k2 = peltier_dyn(state + (k1 * dt / 2), u)
    k3 = peltier_dyn(state + (k2 * dt / 2), u)
    k4 = peltier_dyn(state + (k3 * dt), u)

    next_state = state + dt * ((1 / 6) * k1 + (1 / 3) * k2 + (1 / 3) * k3 + (1 / 6) * k4)

    return next_state

def rk4_sensor(T_s: float, T_p: float, dt: float):
    k1 = sensor_dyn(T_s, T_p)
    k2 = sensor_dyn(T_s + (k1 * dt / 2), T_p)
    k3 = sensor_dyn(T_s + (k2 * dt / 2), T_p)
    k4 = sensor_dyn(T_s + (k3 * dt), T_p)

    next_state = T_s + dt * ((1 / 6) * k1 + (1 / 3) * k2 + (1 / 3) * k3 + (1 / 6) * k4)

    return next_state


if __name__ == '__main__':
    # parameters known
    R = 2  # Peltier Resistance [Ohm]
    Tinf = 25.3  # Ambient Temperature [C]

    # parameters estimate
    alpha = 0.07  # Seebeck Coefficient [V/K]
    K = 0.1  # Thermal Conductance [W/K]
    C = 5  # Thermal Capacitance [J/K]
    G = 0.1  # Ambient Conductance [W/K]
    tau = 5  # Sensor delay

    dt = 0.01
    num_steps = 4500
    true_temp = np.array([Tinf, Tinf])
    mes_temp = np.array([Tinf, Tinf])

    I_avg = 0
    time = 0.0

    time_arr = np.zeros(num_steps)
    state_arr = np.zeros((num_steps, 2))
    state_mes_arr = np.zeros((num_steps, 2))

    for i in range(num_steps):

        if time> 5 and time < 30:
            I_avg = 1
        else:
            I_avg = 0


        true_temp = rk4_peltier(true_temp, I_avg, dt)
        state_arr[i, :] = true_temp

        mes_temp[0] = rk4_sensor(mes_temp[0], true_temp[0], dt)
        mes_temp[1] = rk4_sensor(mes_temp[1], true_temp[1], dt)
        state_mes_arr[i, :] = mes_temp

        time = i * dt
        time_arr[i] = time

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time_arr, state_arr[:, 0], label='T1', color=[0, 1, 0])
    ax.plot(time_arr, state_arr[:, 1], label='T2', color=[0, 0.5, 0])
    ax.plot(time_arr, state_mes_arr[:, 0], label='T1_mes', color=[0, 1, 0], linestyle='--')
    ax.plot(time_arr, state_mes_arr[:, 1], label='T2_mes', color=[0, 0.5, 0], linestyle='--')
    plt.show()

