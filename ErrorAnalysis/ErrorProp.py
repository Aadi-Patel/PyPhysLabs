import numpy as np
import scipy as sp

class Measurement:
    """
    A class to handle measurements with uncertainties and their error propagation.
    
    Supports basic mathematical operations (+, *, /) with automatic error propagation
    following standard uncertainty propagation rules.
    """
    
    def __init__(self, value: float, uncertainty: float):
        """
        Initialize a measurement with its value and uncertainty.
        
        Args:
            value: The measured value
            uncertainty: The uncertainty/error in the measurement
        """
        self.value = float(value)
        self.uncertainty = abs(float(uncertainty))  # Ensure uncertainty is positive
    
    def __add__(self, other):
        """
        Add two measurements or a measurement and a number.
        Error propagation: δ(a+b) = sqrt((δa)² + (δb)²)
        """
        if isinstance(other, (int, float)):
            return Measurement(self.value + other, self.uncertainty)
        value = self.value + other.value
        uncertainty = np.sqrt(self.uncertainty**2 + other.uncertainty**2)
        return Measurement(value, uncertainty)
    
    def __mul__(self, other):
        """
        Multiply two measurements or a measurement and a number.
        Error propagation: δ(a*b)/|a*b| = sqrt((δa/a)² + (δb/b)²)
        """
        if isinstance(other, (int, float)):
            return Measurement(self.value * other, self.uncertainty * abs(other))
        value = self.value * other.value
        uncertainty = abs(value) * np.sqrt((self.uncertainty/self.value)**2 + 
                                         (other.uncertainty/other.value)**2)
        return Measurement(value, uncertainty)
    
    def __rmul__(self, other):
        """
        Handle multiplication when number is on the left side.
        Example: 2 * Measurement(3, 0.1)
        """
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """
        Divide two measurements or a measurement and a number.
        Error propagation: δ(a/b)/|a/b| = sqrt((δa/a)² + (δb/b)²)
        """
        if isinstance(other, (int, float)):
            return Measurement(self.value / other, self.uncertainty / abs(other))
        value = self.value / other.value
        uncertainty = abs(value) * np.sqrt((self.uncertainty/self.value)**2 + 
                                         (other.uncertainty/other.value)**2)
        return Measurement(value, uncertainty)
    
    def __str__(self):
        """
        Return string representation of measurement in format: value ± uncertainty
        """
        return f"{self.value:.3g} ± {self.uncertainty:.3g}"

def read_from_array(values: np.ndarray, uncertainties: np.ndarray) -> list:
    """
    Create a list of Measurement objects from arrays of values and uncertainties.
    
    Args:
        values: Array of measured values
        uncertainties: Array of corresponding uncertainties
    
    Returns:
        List of Measurement objects
    
    Example:
        >>> vals = np.array([1.0, 2.0])
        >>> errs = np.array([0.1, 0.2])
        >>> measurements = read_from_array(vals, errs)
    """
    return [Measurement(v, u) for v, u in zip(values, uncertainties)]

def read_from_table(filename: str, value_col: int, uncertainty_col: int, 
                   delimiter: str = ',', skip_header: int = 1) -> list:
    """
    Read measurements from a CSV or similar table file.
    
    Args:
        filename: Path to the data file
        value_col: Column index for values
        uncertainty_col: Column index for uncertainties
        delimiter: Column separator in file (default: ',')
        skip_header: Number of header rows to skip (default: 1)
    
    Returns:
        List of Measurement objects
    
    Example:
        >>> measurements = read_from_table('data.csv', 0, 1)
    """
    data = np.genfromtxt(filename, delimiter=delimiter, skip_header=skip_header)
    return read_from_array(data[:,value_col], data[:,uncertainty_col])

# Example usage:
if __name__ == "__main__":
    # Direct measurements
    length = Measurement(10.0, 0.1)
    width = Measurement(5.0, 0.1)
    
    # Area calculation with propagated error
    area = length * width
    print(f"Area: {area}")
    
    # Reading from arrays
    values = np.array([1.0, 2.0, 3.0])
    errors = np.array([0.1, 0.2, 0.3])
    measurements = read_from_array(values, errors)

