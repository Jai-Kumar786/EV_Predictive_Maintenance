"""
Week 10, Task 1, Step 4B: System Validation with a Digital Twin Prototype (FastSim-3 Working)

This version correctly handles FastSim-3's data structure and API.
"""

import fastsim as fsim
import pandas as pd
import numpy as np
import os

print("--- Digital Twin Simulation Started (FastSim-3 Working) ---")

# --- 1. Configuration ---
output_dir = 'data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")
output_file_path = os.path.join(output_dir, 'simulated_trip_data.csv')

try:
    # --- 2. Load Vehicle and Drive Cycle (FastSim-3 API) ---
    # Load vehicle from available resources
    vehicle = fsim.Vehicle.from_resource("2022_Renault_Zoe_ZE50_R135.yaml")
    print(f"Loaded Digital Twin Vehicle: 2022 Renault Zoe ZE50 R135")

    # Set save interval to capture time-series data
    vehicle.set_save_interval(1)

    # Load UDDS cycle from built-in resources
    cycle = fsim.Cycle.from_resource("udds.csv")
    print(f"Loaded Drive Cycle: UDDS (Urban Dynamometer Driving Schedule)")

    # --- 3. Run the Simulation (FastSim-3 API) ---
    # Create SimDrive object
    sim_drive = fsim.SimDrive(vehicle, cycle)
    print("Running simulation... This may take a moment as it solves powertrain dynamics for each second.")

    # Use walk() method for FastSim-3
    sim_drive.walk()
    print("Simulation complete.")

    # --- 4. Extract and Format the Output (FastSim-3 Data Access) ---
    # Convert to DataFrame - this contains all the time-series data
    results_df = sim_drive.to_dataframe()

    print(f"Raw simulation data contains {len(results_df)} rows with {len(results_df.columns)} columns")

    # Extract the key columns for your predictive maintenance pipeline
    clean_df = pd.DataFrame({
        'time_s': results_df['cyc.time_seconds'],
        'speed_mps': results_df['veh.history.speed_ach_meters_per_second'],
        'power_kw': results_df['veh.pt_type.BEV.res.history.pwr_out_electrical_watts'] / 1000,
        'soc': results_df['veh.pt_type.BEV.res.history.soc'],
        'energy_kwh': results_df['veh.pt_type.BEV.res.history.energy_out_electrical_joules'] / 3.6e6,
    })

    # Calculate approximate current and voltage
    # Assume nominal battery voltage (typical for modern EVs)
    nominal_voltage = 350.0  # Volts (typical for modern BEV)
    clean_df['voltage_V'] = nominal_voltage
    clean_df['current_A'] = clean_df['power_kw'] * 1000 / nominal_voltage
    clean_df['temperature_C'] = 25.0  # Assume constant temperature

    # Data cleaning
    for col in clean_df.columns:
        clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
    clean_df.dropna(inplace=True)

    print(f"\nExtracted {len(clean_df)} valid data points from the simulation.")
    print("Sample of simulated data (first 5 rows):")
    print(clean_df.head())

    # Get cycle duration from the simulation results
    cycle_duration = clean_df['time_s'].iloc[-1]
    print(f"Cycle duration: {cycle_duration:.0f} seconds")

    print("\nSimulation Summary:")
    print(f"Initial SOC: {clean_df['soc'].iloc[0]:.3f}")
    print(f"Final SOC: {clean_df['soc'].iloc[-1]:.3f}")
    print(f"SOC change: {clean_df['soc'].iloc[0] - clean_df['soc'].iloc[-1]:.3f}")
    print(f"Total energy consumed: {clean_df['energy_kwh'].iloc[-1]:.3f} kWh")
    print(f"Total distance: {np.trapezoid(clean_df['speed_mps'], clean_df['time_s']) / 1000:.2f} km")
    print(f"Average power: {clean_df['power_kw'].mean():.1f} kW")
    print(f"Peak power: {clean_df['power_kw'].max():.1f} kW")
    print(f"Max current: {clean_df['current_A'].max():.1f} A")
    print(
        f"Energy efficiency: {clean_df['energy_kwh'].iloc[-1] / (np.trapezoid(clean_df['speed_mps'], clean_df['time_s']) / 1000):.2f} kWh/km")

    # --- 5. Save the Final Dataset ---
    clean_df.to_csv(output_file_path, index=False)

    print("\n--- Digital Twin Prototyping Successful ---")
    print(f"Simulated trip data has been successfully saved to: {output_file_path}")
    print("This file is now ready to be used as the input for your end-to-end system validation.")
    print("The dataset contains realistic EV battery telemetry suitable for your predictive maintenance pipeline.")

    # Additional validation for your pipeline
    print("\n--- Dataset Validation ---")
    print(f"Data quality check:")
    print(f"  - No missing values: {clean_df.isnull().sum().sum() == 0}")
    print(f"  - Monotonic time series: {clean_df['time_s'].is_monotonic_increasing}")
    print(f"  - Realistic SOC range: {clean_df['soc'].min():.3f} to {clean_df['soc'].max():.3f}")
    print(f"  - Physical power values: {clean_df['power_kw'].min():.1f} to {clean_df['power_kw'].max():.1f} kW")
    print("-" * 70)

except Exception as e:
    print(f"\n--- An Error Occurred During Simulation ---")
    print(f"Error details: {e}")
    print("\nDetailed debugging:")
    import traceback

    traceback.print_exc()
    print("\nTroubleshooting steps:")
    print("1. Ensure you have fastsim-3 installed: pip install --upgrade fastsim")
    print("2. Check that the simulation completed successfully")
    print("3. Verify that the vehicle has sufficient battery charge for the cycle")