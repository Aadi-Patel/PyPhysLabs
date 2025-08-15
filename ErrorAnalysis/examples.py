from ErrorProp import Measurement, read_from_array, read_from_table
import numpy as np

# Basic measurements
mass = Measurement(1.23, 0.01)  # 1.23 ± 0.01 kg
velocity = Measurement(4.56, 0.02)  # 4.56 ± 0.02 m/s

# Calculate kinetic energy (1/2 * m * v^2)
kinetic_energy = (0.5 * mass) * (velocity * velocity)
print(f"Kinetic Energy: {kinetic_energy}")

# Example with arrays
temperatures = np.array([20.1, 20.3, 20.2, 20.1])
temp_uncertainties = np.array([0.1, 0.1, 0.1, 0.1])
temp_measurements = read_from_array(temperatures, temp_uncertainties)

# Calculate mean temperature
mean_temp = sum(m.value for m in temp_measurements) / len(temp_measurements)
mean_uncertainty = np.sqrt(sum(m.uncertainty**2 for m in temp_measurements)) / len(temp_measurements)
print(f"Mean Temperature: {Measurement(mean_temp, mean_uncertainty)}")
