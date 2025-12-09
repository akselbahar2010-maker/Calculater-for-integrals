import torch
import sympy as sp
import numpy as np
import tkinter as tk
from tkinter import messagebox
from scipy import integrate
from torchquad import MonteCarlo, Simpson, Trapezoid, Boole
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Helper functions for parsing and calculating
def parse_sympy_expression(expr_string):
    """Parse a string expression to a sympy expression."""
    try:
        # Replace 'e' with 'exp' for proper parsing
        expr_string = expr_string.replace('e', 'exp')
        expr = sp.sympify(expr_string)
        return expr
    except Exception as e:
        raise ValueError(f"Invalid expression: {expr_string}")

def expression_to_function(expr_string):
    """Convert a sympy expression to a Python function."""
    x = sp.Symbol('x')
    expr = parse_sympy_expression(expr_string)
    func = sp.lambdify(x, expr, 'numpy')
    return func

def safe_function(x, func):
    """Safely compute the function, ensuring valid values for log and sqrt."""
    # Ensure the input is within valid domain for log and sqrt
    if x <= 0:
        return np.nan  # Return NaN for invalid values of x
    
    try:
        # Compute the function and check if it's valid
        return func(x)
    except Exception as e:
        logging.error(f"Error computing function at x={x}: {e}")
        return np.nan

def calculate_definite_integral(expr_string, lower_bound, upper_bound, device='cpu'):
    """Calculate the definite integral of an expression from lower_bound to upper_bound."""
    x = sp.Symbol('x')
    try:
        expr = parse_sympy_expression(expr_string)
    except Exception as e:
        logging.error(f"Error parsing expression: {str(e)}")
        raise ValueError(f"Invalid expression: {expr_string}")
        
    try:
        func = expression_to_function(expr_string)
    except Exception as e:
        logging.error(f"Error creating function: {str(e)}")
        raise ValueError(f"Could not create function from expression: {expr_string}")
    
    # Check if infinity is in the bounds
    lower_is_infinite = lower_bound in [-float('inf'), float('inf')]
    upper_is_infinite = upper_bound in [-float('inf'), float('inf')]
    
    # Use TorchQuad for finite bounds, scipy.integrate for infinite bounds
    if not (lower_is_infinite or upper_is_infinite):
        try:
            # Try to use SymPy's symbolic integration first
            symbolic_result = sp.integrate(expr, (x, lower_bound, upper_bound))
            if symbolic_result.is_real and not symbolic_result.has(sp.Integral):
                result = float(symbolic_result.evalf())
                logging.debug(f"SymPy integration result: {result}")
                return result
            else:
                raise ValueError("Symbolic integration inconclusive, falling back to numerical")
        except Exception as sympy_error:
            logging.debug(f"SymPy integration failed: {str(sympy_error)}, trying TorchQuad")
            
            def torch_func(x):
                if isinstance(x, torch.Tensor):
                    try:
                        x_np = x.cpu().numpy()
                        result_np = safe_function(x_np, func)
                        return torch.tensor(result_np, dtype=torch.float32)
                    except Exception as e:
                        logging.error(f"Error in torch_func: {str(e)}")
                        return torch.tensor(float('nan'), dtype=torch.float32)
                else:
                    return safe_function(x, func)

            # Use TorchQuad methods with reduced grid size for faster results
            integration_domain = torch.tensor([[lower_bound, upper_bound]], device=device)
            try:
                integrator = Boole()
                result = integrator.integrate(torch_func, dim=1, N=1001, integration_domain=integration_domain)  # Reduced N for speed
                result = float(result.item())
                logging.debug(f"TorchQuad Boole integration result: {result}")
                return result
            except Exception as e:
                logging.debug(f"Boole's method failed: {str(e)}, trying Simpson's rule.")
                try:
                    integrator = Simpson()
                    result = integrator.integrate(torch_func, dim=1, N=1001, integration_domain=integration_domain)  # Reduced N for speed
                    result = float(result.item())
                    logging.debug(f"TorchQuad Simpson integration result: {result}")
                    return result
                except Exception as e:
                    logging.debug(f"Simpson's method failed: {str(e)}, trying Trapezoidal rule.")
                    try:
                        integrator = Trapezoid()
                        result = integrator.integrate(torch_func, dim=1, N=1001, integration_domain=integration_domain)  # Reduced N for speed
                        result = float(result.item())
                        logging.debug(f"TorchQuad Trapezoid integration result: {result}")
                        return result
                    except Exception as e:
                        logging.error(f"All TorchQuad methods failed, using SciPy: {str(e)}")
                        result, error = integrate.quad(func, lower_bound, upper_bound)
                        logging.debug(f"SciPy integration result: {result}")
                        return result
    else:
        # If bounds are infinite, fall back to SciPy
        try:
            result, error = integrate.quad(func, lower_bound, upper_bound)
            logging.debug(f"SciPy integration for infinite bounds: {result}")
            return result
        except Exception as e:
            logging.error(f"Integration failed for infinite bounds: {str(e)}")
            return None

# GUI using Tkinter
def create_ui():
    root = tk.Tk()
    root.title("Integral Calculator")

    def on_calculate():
        try:
            expr = expression_entry.get()
            lower = float(lower_bound_entry.get())
            upper = float(upper_bound_entry.get())
            
            # Select device (CPU or GPU)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Perform the integration
            result = calculate_definite_integral(expr, lower, upper, device=device)
            if result is None:
                messagebox.showerror("Error", "Integration failed. Check the bounds or expression.")
            else:
                result_label.config(text=f"Result: {result}")
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")

    # Create UI components
    expression_label = tk.Label(root, text="Enter the function (e.g., sin(x) + cos(x)): ")
    expression_label.pack()

    expression_entry = tk.Entry(root, width=50)
    expression_entry.pack()

    lower_bound_label = tk.Label(root, text="Enter lower bound: ")
    lower_bound_label.pack()

    lower_bound_entry = tk.Entry(root, width=50)
    lower_bound_entry.pack()

    upper_bound_label = tk.Label(root, text="Enter upper bound: ")
    upper_bound_label.pack()

    upper_bound_entry = tk.Entry(root, width=50)
    upper_bound_entry.pack()

    calculate_button = tk.Button(root, text="Calculate", command=on_calculate)
    calculate_button.pack()

    result_label = tk.Label(root, text="Result: ")
    result_label.pack()

    # Start the Tkinter event loop
    root.mainloop()

# Run the UI
if __name__ == "__main__":
    create_ui()
