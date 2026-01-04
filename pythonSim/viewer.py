import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data
# We read the raw CSV first
try:
    df_raw = pd.read_csv('peltier_run.csv')
    #df_raw = pd.read_csv('peltier_data.csv')
except FileNotFoundError:
    print("CSV file not found. Run the logger first!")
    exit()

# 2. Separate into two DataFrames
# Filter for Inner Loop (Current) data
df_current = df_raw[df_raw['Type'] == 'I'].copy()
# Filter for Temperature data
df_temp = df_raw[df_raw['Type'] == 'T'].copy()

# 3. Clean and Rename columns for clarity
# For Temperature, 'Setpoint' is actually Temp1 and 'Current' is Temp2
df_temp = df_temp.rename(columns={'Setpoint': 'Temp1', 'Current': 'Temp2'})
# Drop the PWM column for temperature as it's empty
df_temp = df_temp.drop(columns=['PWM'])

# 4. Convert timestamps to relative time (using the very first timestamp in the file)
start_time = df_raw['Timestamp'].iloc[0]
df_current['Time_Sec'] = df_current['Timestamp'] - start_time
df_temp['Time_Sec'] = df_temp['Timestamp'] - start_time

# 5. Create the Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# TOP PLOT: Current Loop
ax1.set_title('Peltier Inner Loop (Current Control)')
ax1.plot(df_current['Time_Sec'], df_current['Setpoint'], 'r--', label='Current Setpoint (mA)', alpha=0.7)
ax1.plot(df_current['Time_Sec'], df_current['Current'], 'b-', label='Actual Current (mA)')
ax1.set_ylabel('Current (mA)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Add PWM on a twin axis for the top plot
ax1_pwm = ax1.twinx()
ax1_pwm.plot(df_current['Time_Sec'], df_current['PWM'], 'g-', label='PWM Output', alpha=0.3)
ax1_pwm.set_ylabel('PWM Value', color='green')
ax1_pwm.tick_params(axis='y', labelcolor='green')

# BOTTOM PLOT: Temperature
ax2.set_title('Temperature Sensor Readings')
ax2.plot(df_temp['Time_Sec'], df_temp['Temp1'], label='Sensor 1', color='orange')
ax2.plot(df_temp['Time_Sec'], df_temp['Temp2'], label='Sensor 2', color='purple')
ax2.set_ylabel('Temperature (°C)')
ax2.set_xlabel('Time (seconds)')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('peltier_analysis.png')
plt.show()

# 6. Print Stats
print(f"--- Data Summary ---")
print(f"Current Samples: {len(df_current)}")
print(f"Temp Samples:    {len(df_temp)}")
if not df_current.empty:
    rmse = ((df_current['Setpoint'] - df_current['Current'])**2).mean()**0.5
    print(f"Current Loop RMSE: {rmse:.2f} mA")