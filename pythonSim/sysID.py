import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from dataclasses import dataclass
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import aerosandbox.tools.pretty_plots as p
from trame_server.utils.logger import initial_state


@dataclass
class PeltierParams:
    C: float
    K: float
    R: float
    n: float
    m: float
    alpha: float
    G: float
    Tinf: float
    tau: float


@dataclass
class SimParams:
    T1_init: float
    T2_init: float
    sim_length: float
    dt: float
    allow_current: bool


@dataclass
class ModelResults:
    pelt_params: PeltierParams
    sim_params: SimParams
    time: np.ndarray
    temp_true: np.ndarray
    temp_measure: np.ndarray


class sysID:
    df_current: pd.DataFrame
    df_temp: pd.DataFrame
    start_time: float
    init_params: PeltierParams

    start_current_step: float
    end_current_step: float
    current_value: float
    run_length: float
    init_Tinf: float  # initial temperature

    def __init__(self, file_path: str, init_params: PeltierParams):
        self.file_path = file_path
        self.init_params = init_params
        self.get_data_frames()
        self.compensate_sensor_lag()
        self.get_current_times()

        self.tau = 5

    def get_data_frames(self):
        try:
            df_raw = pd.read_csv(self.file_path)
        except FileNotFoundError:
            print("CSV file not found. Run the logger first!")
            exit()

        self.df_current = df_raw[df_raw['Type'] == 'I'].copy()
        self.df_temp = df_raw[df_raw['Type'] == 'T'].copy()

        self.df_temp = self.df_temp.rename(columns={'Setpoint': 'Temp1', 'Current': 'Temp2'})
        self.df_temp = self.df_temp.drop(columns=['PWM'])

        # 4. Convert timestamps to relative time (using the very first timestamp in the file)
        self.start_time = df_raw['Timestamp'].iloc[0]
        self.df_current['Time_Sec'] = self.df_current['Timestamp'] - self.start_time
        self.df_temp['Time_Sec'] = self.df_temp['Timestamp'] - self.start_time

    def compensate_sensor_lag(self, tau: float = 5.0, window_len: int = 30, poly_order: int = 2):
        # Convert data to numpy
        t = self.df_temp['Time_Sec'].to_numpy()
        T1_raw = self.df_temp['Temp1'].to_numpy()
        T2_raw = self.df_temp['Temp2'].to_numpy()

        # Step 1: Smooth the raw data
        # Savitzky-Golay preserves the 'peak' values better than a moving average
        T1_smooth = savgol_filter(T1_raw, window_length=window_len, polyorder=poly_order)
        T2_smooth = savgol_filter(T2_raw, window_length=window_len, polyorder=poly_order)

        # Step 2: Calculate the derivative of the smoothed signal
        # This represents the physical rate of change (C/s)
        dT1_dt = np.gradient(T1_smooth, t)
        dT2_dt = np.gradient(T2_smooth, t)

        # Step 3: Apply the Inverse Model (Lead Compensation)
        # T_reconstructed = T_measured + (tau * dT/dt)
        T1_recon = T1_smooth + (tau * dT1_dt)
        T2_recon = T2_smooth + (tau * dT2_dt)

        # Save to the dataframe for plotting
        self.df_temp['T1_Smooth'] = T1_smooth
        self.df_temp['T1_Recon'] = T1_recon
        self.df_temp['T2_Smooth'] = T2_smooth
        self.df_temp['T2_Recon'] = T2_recon

        return T1_recon, T2_recon

    def get_current_times(self):
        current_arr = np.array(self.df_current['Setpoint'])
        time_arr = np.array(self.df_current['Time_Sec'])

        indices = np.nonzero(current_arr)[0]

        first_index = indices[0]
        last_index = indices[-1]

        self.start_current_step = float(time_arr[first_index])
        self.end_current_step = float(time_arr[last_index])
        self.current_value = float(current_arr[first_index + 1] / 1000)  # convert from mA to A
        self.run_length = float(time_arr[-1])
        self.init_Tinf = (self.df_temp['Temp1'].iloc[0] + self.df_temp['Temp2'].iloc[0]) / 2

    def param_est_1(self):

        t_start = self.end_current_step + 5.0
        df_relax = self.df_temp[self.df_temp['Time_Sec'] >= t_start].copy()
        dt = 0.1

        t_raw = df_relax['Time_Sec'].to_numpy()
        t = t_raw - t_raw[0]
        sim_length = t[-1] - t[0]
        T1 = df_relax['T1_Recon'].to_numpy()
        T2 = df_relax['T2_Recon'].to_numpy()

        T1_init = T1[0]
        T2_init = T2[0]

        def objective(x: np.ndarray) -> float:
            a1, a2, a3, a4, n = float(x[0]), 0, 0, float(x[1]), float(x[2])
            m = 1

            C = 1
            K = a1 * C
            R = a2 * C
            alpha = a3 * C
            G = a4 * C
            Tinf = self.init_Tinf

            # Run simulation
            pelt_params = PeltierParams(C=C, K=K, R=R, n=n, m=m, alpha=alpha, G=G, Tinf=Tinf, tau=self.tau)
            sim_params = SimParams(T1_init=T1_init, T2_init=T2_init, dt=dt, sim_length=sim_length, allow_current=False)
            res = self.run_sim(pelt_params, sim_params)

            sim_temp_1 = res.temp_true[:, 0]
            sim_temp_2 = res.temp_true[:, 1]
            sim_time = res.time

            sim_temp_1_interp = np.interp(t, sim_time, sim_temp_1)
            sim_temp_2_interp = np.interp(t, sim_time, sim_temp_2)

            T1_residuals = sim_temp_1_interp - T1
            T2_residuals = sim_temp_2_interp - T2

            least_squares_diff = np.sum(T1_residuals ** 2) + np.sum(T2_residuals ** 2)
            """
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(t, sim_temp_1_interp, label='T1', color=[1, 0.8, 0])
            ax.plot(t, sim_temp_2_interp, label='T2', color=[1, 0.5, 0])
            ax.set_ylabel('Temperature (°C)')
            ax.set_xlabel('Time (seconds)')
            ax.grid(True, alpha=0.3)
            p.show_plot(rotate_axis_labels=False)
            """
            return least_squares_diff

        x0 = np.array([0.001, 0.05, 2])
        bounds = [(0.0001, 1), (0.01, 1), (1.2, 2.5)]
        result = minimize(
            objective,
            x0=x0,
            bounds=bounds,
            method='L-BFGS-B',
            tol=1e-3,
            options={
                'ftol': 1e-4,
                'gtol': 1e-4,
                'disp': True,
            }
        )
        print(result)

        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t, T1, label='T1', color=[0, 0.8, 0])
        ax.plot(t, T2, label='T2', color=[0, 0.5, 0])
        ax.set_ylabel('Temperature (°C)')
        ax.set_xlabel('Time (seconds)')
        ax.grid(True, alpha=0.3)
        p.show_plot(rotate_axis_labels=False)
        """

        opt_a1 = result.x[0]
        opt_a4 = result.x[1]
        opt_n = result.x[2]
        return opt_a1, opt_a4, opt_n

    def find_K_G_least_squares(self):
        pass

    def run_sim(self, pelt_params: PeltierParams, sim_params: SimParams) -> ModelResults:
        dt = sim_params.dt
        sim_length = sim_params.sim_length
        num_steps = int(sim_length / dt)
        time_arr = np.arange(num_steps) * dt

        current_profile = np.where(
            (time_arr >= self.start_current_step) & (time_arr <= self.end_current_step),
            self.current_value,
            0.0
        )

        initial_state = np.array([sim_params.T1_init, sim_params.T2_init])

        state_arr = np.full((num_steps, 2), initial_state)
        state_mes_arr = np.full((num_steps, 2), initial_state)

        for i in range(1, num_steps):
            I_avg = current_profile[i - 1]

            # Step Peltier Dynamics
            state_arr[i] = self.rk4_peltier(state_arr[i - 1], I_avg, dt, pelt_params)

            # Step Sensor Dynamics (First Order Lag)
            state_mes_arr[i, 0] = self.rk4_sensor(state_mes_arr[i - 1, 0], state_arr[i, 0], dt, pelt_params)
            state_mes_arr[i, 1] = self.rk4_sensor(state_mes_arr[i - 1, 1], state_arr[i, 1], dt, pelt_params)

        results = ModelResults(pelt_params, sim_params, time_arr, state_arr, state_mes_arr)
        return results

    def draw_recon(self):
        t = self.df_temp['Time_Sec'].to_numpy()
        T1 = self.df_temp['Temp1'].to_numpy()
        T2 = self.df_temp['Temp2'].to_numpy()
        T1_Recon = self.df_temp['T1_Recon'].to_numpy()
        T2_Recon = self.df_temp['T2_Recon'].to_numpy()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t, T1_Recon, label='T1', color=[0, 0.8, 0])
        ax.plot(t, T2_Recon, label='T2', color=[0, 0.5, 0])
        ax.plot(t, T1, label='T1', color=[0, 0.8, 0], linestyle='--')
        ax.plot(t, T2, label='T2', color=[0, 0.5, 0], linestyle='--')

        ymin, ymax = ax.get_ylim()
        ax.plot([self.start_current_step, self.start_current_step], [ymin, ymax], linestyle='--', color='k', alpha=0.5)
        ax.plot([self.end_current_step, self.end_current_step], [ymin, ymax], linestyle='--', color='k', alpha=0.5)
        ax.set_ylim(ymin, ymax)

        ax.set_ylabel('Temperature (°C)')
        ax.set_xlabel('Time (seconds)')
        ax.grid(True, alpha=0.3)
        p.show_plot(rotate_axis_labels=False)

    def draw_results(self, results: ModelResults, overlay=True, show_true=False):
        time_arr = results.time
        state_arr = results.temp_true
        state_mes_arr = results.temp_measure

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if show_true:
            ax.plot(time_arr, state_arr[:, 0], label='T1', color=[0, 0.8, 0])
            ax.plot(time_arr, state_arr[:, 1], label='T2', color=[0, 0.5, 0])

        ax.plot(time_arr, state_mes_arr[:, 0], label='T1_mes', color=[0, 0.8, 0], linestyle='--')
        ax.plot(time_arr, state_mes_arr[:, 1], label='T2_mes', color=[0, 0.5, 0], linestyle='--')

        if overlay:
            ax.plot(self.df_temp['Time_Sec'], self.df_temp['Temp1'], label='Sensor 1', color=[0.5, 0.8, 0])
            #ax.plot(self.df_temp['Time_Sec'], self.df_temp['T1_Recon'], label='Sensor 1', color=[0.5, 0.8, 0])
            ax.plot(self.df_temp['Time_Sec'], self.df_temp['Temp2'], label='Sensor 2', color=[0.5, 0.5, 0])
            #ax.plot(self.df_temp['Time_Sec'], self.df_temp['T2_Recon'], label='Sensor 2', color=[0.5, 0.5, 0])

        ymin, ymax = ax.get_ylim()
        ax.plot([self.start_current_step, self.start_current_step], [ymin, ymax], linestyle='--', color='k', alpha=0.5)
        ax.plot([self.end_current_step, self.end_current_step], [ymin, ymax], linestyle='--', color='k', alpha=0.5)
        ax.set_ylim(ymin, ymax)
        #ax.set_xlim(0, 60)

        ax.set_ylabel('Temperature (°C)')
        ax.set_xlabel('Time (seconds)')
        ax.grid(True, alpha=0.3)

        p.show_plot(rotate_axis_labels=False)

    def rk4_peltier(self, state: np.ndarray, u: float, dt: float, params: PeltierParams):
        k1 = self.peltier_dyn(state, u, params)
        k2 = self.peltier_dyn(state + (k1 * dt / 2), u, params)
        k3 = self.peltier_dyn(state + (k2 * dt / 2), u, params)
        k4 = self.peltier_dyn(state + (k3 * dt), u, params)

        next_state = state + dt * ((1 / 6) * k1 + (1 / 3) * k2 + (1 / 3) * k3 + (1 / 6) * k4)

        return next_state

    def rk4_sensor(self, T_s: float, T_p: float, dt: float, params: PeltierParams):
        k1 = self.sensor_dyn(T_s, T_p, params)
        k2 = self.sensor_dyn(T_s + (k1 * dt / 2), T_p, params)
        k3 = self.sensor_dyn(T_s + (k2 * dt / 2), T_p, params)
        k4 = self.sensor_dyn(T_s + (k3 * dt), T_p, params)

        next_state = T_s + dt * ((1 / 6) * k1 + (1 / 3) * k2 + (1 / 3) * k3 + (1 / 6) * k4)

        return next_state

    @staticmethod
    def peltier_dyn(T: np.ndarray, I_avg: float, params: PeltierParams) -> np.ndarray:
        dTdt = np.zeros(2)
        T1 = T[0]
        T2 = T[1]
        dT_internal = T2 - T1

        dT1_diff = params.Tinf - T1
        dT2_diff = params.Tinf - T2
        G1_flow = params.G * np.sign(dT1_diff) * (np.abs(dT1_diff) ** params.m)
        G2_flow = params.G * np.sign(dT2_diff) * (np.abs(dT2_diff) ** params.m)

        # Modeling heat flow as Q = K * sign(dT) * |dT|^n
        K_flow = params.K * np.sign(dT_internal) * (np.abs(dT_internal) ** params.n)

        dTdt[0] = (1 / params.C) * (K_flow - (params.alpha * T1 * I_avg) +
                                    (0.5 * params.R * I_avg ** 2) + params.G * G1_flow)

        # For T2, the internal flow is simply the negative of the T1 flow
        dTdt[1] = (1 / params.C) * (-K_flow + (params.alpha * T2 * I_avg) +
                                    (0.5 * params.R * I_avg ** 2) + params.G * G2_flow)

        return dTdt

    @staticmethod
    def sensor_dyn(T_s: float, T_p: float, params: PeltierParams):
        dTsdt = (1 / params.tau) * (T_p - T_s)
        return dTsdt


if __name__ == '__main__':
    # TODO: do nonlinear curve fit to the dynamics without current (get n, K/C, and G/C)
    # TODO: After that find alpha and C 

    # parameters known
    R = 2  # Peltier Resistance [Ohm]
    Tinf = 24.3  # Ambient Temperature [C]

    # parameters estimate
    alpha = 0.075  # Seebeck Coefficient [V/K]
    K = 0.005  # Thermal Conductance [W/K]
    C = 5  # Thermal Capacitance [J/K]
    n = 1.8
    m = 1
    G = 0.019  # Ambient Conductance [W/K]
    tau = 5  # Sensor delay

    #params = PeltierParams(C=C, C_atm=C_atm, K_atm=K_atm, K=K, R=R, alpha=alpha, G=G, Tinf=Tinf, tau=tau, J=J)
    params = PeltierParams(C=C, K=K, R=R, n=n, m=m, alpha=alpha, G=G, Tinf=Tinf, tau=tau)
    file_path = 'peltier_run_3.csv'
    sys = sysID(file_path, params)
    sys.param_est_1()
    #sys.find_C_alpha(R)
    #sys.find_K_G_least_squares(C)
    #res = sys.run_sim(params)
    #sys.draw_results(res, overlay=True, show_true=True)
    #sys.draw_recon()

    """
    dt = 0.01
    num_steps = 4500
    true_temp = np.array([Tinf, Tinf])
    mes_temp = np.array([Tinf, Tinf])

    I_avg = 0.0
    time = 0.0

    time_arr = np.zeros(num_steps)
    state_arr = np.zeros((num_steps, 2))
    state_mes_arr = np.zeros((num_steps, 2))

    for i in range(num_steps):

        if time > 5 and time < 30:
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
    
        def find_exp(self):
        # 1. Extract the data
        # We only fit the data during the 'Active' window where the step occurs
        post_step_start = self.end_current_step + 10.0

        # Create the mask: From post_step_start to the end of the data
        mask = (self.df_temp['Time_Sec'] >= post_step_start)

        df_active = self.df_temp[mask].copy()

        # Shift time so it starts at 0 for the fit
        t_data = (df_active['Time_Sec'] - self.start_current_step).to_numpy()
        y_data = (df_active['Temp1'] - df_active['Temp2']).to_numpy()

        # 2. Define the model function: y = a * t^n
        # We add 'a' because 't^n' alone usually won't match the magnitude of temp diff
        def power_model(t, a, n):
            return a * (t ** n)

        # Initial guess: a=1, n=0.5 (square root behavior is common in thermal diffusion)
        popt, pcov = curve_fit(power_model, t_data, y_data, p0=[1, 0.5])
        a_fit, n_fit = popt

        print(f"Fit Results: y = {a_fit:.4f} * t^{n_fit:.4f}")
        print(f"Calculated exponent (n): {n_fit:.4f}")

        t_fit = np.linspace(50, t_data[-1], 100)
        y_fit = a_fit * (t_fit ** n_fit)

        plt.figure()
        plt.scatter(t_data, y_data, label='Experimental Diff (T1-T2)', s=2, alpha=0.5)
        plt.plot(t_fit, y_fit, color='red', label=f'Fit: t^{n_fit:.2f}')
        plt.legend()
        plt.title("Power Law Fit for Thermal Growth")
        plt.show()

        return n_fit
        
            #T_avg_plates = (T1 + T2) / 2

        # params.J: How fast the local air pocket equalizes with the plates
        # params.J_env: How fast the air pocket loses heat to the room (Tinf)
        #J = 1
        #J_env = 1.0
        #dTdt[2] = J * (T_avg_plates - Tamb) + J_env * (params.Tinf - Tamb)

        #dTdt[2] = (1 / params.C_atm) * (params.G * (T1 - Tatm1) +
        #                                params.K_atm * (Tatm2 - Tatm1) +
        #                                params.J * (params.Tinf - Tatm1))

        # Atmospheric Node 2
        #dTdt[3] = (1 / params.C_atm) * (params.G * (T2 - Tatm2) +
        #                                params.K_atm * (Tatm1 - Tatm2) +
        #                                params.J * (params.Tinf - Tatm2))
    
    """

    """
    derivative_length = 4
    time_after_step = 2
    derivative_start = self.start_current_step + time_after_step
    derivative_end = derivative_start + derivative_length

    mask = (self.df_temp['Time_Sec'] >= derivative_start) & (self.df_temp['Time_Sec'] <= derivative_end)
    df_slice = self.df_temp.loc[mask]

    # 2. Extract and convert to numpy arrays
    time_slice = df_slice['Time_Sec'].to_numpy()
    temp1_slice = df_slice['T1_Recon'].to_numpy()
    temp2_slice = df_slice['T2_Recon'].to_numpy()

    dT_dt1 = np.gradient(temp1_slice, time_slice)
    dT_dt2 = np.gradient(temp2_slice, time_slice)

    avg_dT_dt1 = np.mean(dT_dt1)
    avg_dT_dt2 = np.mean(dT_dt2)
    T1_avg = np.mean(temp1_slice)
    T2_avg = np.mean(temp2_slice)
    I = self.current_value

    A = np.array([[0.5 * R * I**2, -I * T1_avg], [0.5 * R * I**2, I * T2_avg]])
    b = np.array([avg_dT_dt1, avg_dT_dt2])
    solution = np.linalg.solve(A, b)
    x = solution[0]
    y = solution[1]

    C = 1/x
    alpha = -C*y

    print(1)
    
    
# 1. Define the relaxation window (10s after power off to allow sensors to settle)
        t_start = self.end_current_step + 10.0
        df_relax = self.df_temp[self.df_temp['Time_Sec'] >= t_start].copy()

        t = df_relax['Time_Sec'].to_numpy()
        T1 = df_relax['Temp1'].to_numpy()
        T2 = df_relax['Temp2'].to_numpy()
        Tinf = self.init_params.Tinf

        # 2. Calculate smooth derivatives (dT/dt)
        # Using the smoothing method we discussed to avoid noise spikes
        from scipy.signal import savgol_filter
        T1_smooth = savgol_filter(T1, window_length=15, polyorder=2)
        dT1_dt = np.gradient(T1_smooth, t)

        # 3. Construct the A matrix (N rows x 2 columns)
        # Column 0: Internal conductance term (T2 - T1)
        # Column 1: Ambient loss term (Tinf - T1)
        col0 = (T2 - T1)
        col1 = (Tinf - T1)

        A = np.column_stack((col0, col1))

        # 4. Construct the B vector (N rows)
        B = dT1_dt

        # 5. Solve using Least Squares
        # sol[0] = K/C,  sol[1] = G/C
        sol, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)

        # 6. Extract physical parameters
        K_fitted = sol[0] * C_fixed
        G_fitted = sol[1] * C_fixed

        print(f"--- Relaxation Analysis (I=0) ---")
        print(f"Fixed C used: {C_fixed:.4f}")
        print(f"Fitted K: {K_fitted:.6f} W/K")
        print(f"Fitted G: {G_fitted:.6f} W/K")

        return K_fitted, G_fitted
    """
