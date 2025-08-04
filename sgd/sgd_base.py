import numpy as np
from scipy.optimize import approx_fprime
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logger
logger = logging.getLogger("SGDBase")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# Class for Stochastic Gradient Descent (SGD) optimization base class
class SGDBase:
    def __init__(self, learning_rate = 0.1, record_history = True):
        self.learning_rate = learning_rate
        self.history = {'iteration': [], 'loss': [], 'params': [], 'grads': []}
        self.record_history = record_history
        
        self.iterations = 0
        logger.info(f"SGDBase initialized with learning_rate={learning_rate}, record_history={record_history}")

    def step(self, params=None, loss = None, grads = None):
        logger.info(f"Starting SGD step {self.iterations + 1}")
        self.grads = grads
        self.loss = loss
        if params is None:
            params = self.params
        else:
            self.params = params


        if self.grads is None and self.loss is None:
            logger.error("Either loss or grads must be provided.")
            raise ValueError("Either loss or grads must be provided.")
        
        self.iterations += 1

        # take one step
        if self.grads is None:
            logger.info("Calculating gradients using finite difference method.")
            # calculate gradients
            # use scipy's finite difference method
            self.grads = approx_fprime(self.params, self.loss, epsilon=1e-8)

        if self.record_history and self.iterations == 0:
            logger.info("Logging first iteration to history.")
            # Log the first iteration
            self.log_history()

        logger.info(f"Updating parameters with learning_rate={self.learning_rate}")
        self.params = self.params - self.learning_rate * self.grads

        if self.record_history:
            logger.info("Logging current iteration to history.")
            # Log the current iteration
            self.log_history()

        # Return the updated parameters
        logger.info(f"Step {self.iterations} complete. Returning updated parameters.")
        return self.params

    def log_history(self):
        self.history['iteration'].append(self.iterations)
        self.history['loss'].append(self.loss(self.params))
        self.history['params'].append(self.params)
        self.history['grads'].append(self.grads)
        logger.info(f"History logged for iteration {self.iterations}.")

    def plot_loss_vs_iterations(self, figsize=(8, 5)):
        """
        Plots loss vs. iterations using matplotlib with professional styling.

        Parameters:
        - history: dict containing 'iteration' and 'loss' keys
        - figsize: tuple, size of the figure
        """

        # Use a professional seaborn color palette
        sns.set_palette("deep")
        sns.set_style("whitegrid")

        plt.figure(figsize=figsize, dpi=150)
        plt.plot(self.history['iteration'], self.history['loss'], linewidth=2, marker='o')

        # Set Calibri font and bold labels
        plt.xlabel('Iteration', fontname='Calibri', fontsize=14, fontweight='bold')
        plt.ylabel('Loss', fontname='Calibri', fontsize=14, fontweight='bold')
        plt.title('Loss vs. Iterations', fontname='Calibri', fontsize=16, fontweight='bold')

        plt.xticks(fontname='Calibri', fontsize=12)
        plt.yticks(fontname='Calibri', fontsize=12)

        plt.tight_layout()
        plt.show()




class SGDLineSearch(SGDBase):
    def __init__(self, learning_rate=0.1, record_history=True):
        super().__init__(learning_rate, record_history)
        logger.info("SGDLineSearch initialized.")

    def step(self, params=None, loss=None, grads=None):
        """
        Performs a single optimization step using line search.

        Parameters:
            params (np.ndarray, optional): Current parameter values. If None, uses self.params.
            loss (callable, optional): Loss function to minimize. Required if grads is not provided.
            grads (np.ndarray, optional): Gradient of the loss at params. If None, computed via finite differences.

        Returns:
            np.ndarray: Updated parameter values after the step.

        The update rule is:
            params_new = params - alpha_opt * grads
        where alpha_opt is found by minimizing phi(alpha) = loss(params - alpha * grads) using line search.
        """
        logger.info("Starting line search step.")
        self.loss = loss
        if params is None:
            params = self.params
        else:
            self.params = params

        # Compute gradient if not provided
        if grads is None:
            if self.loss is None:
                logger.error("Either loss or grads must be provided.")
                raise ValueError("Either loss or grads must be provided.")
            logger.info("Calculating gradients using finite difference method.")
            grads = approx_fprime(self.params, self.loss, epsilon=1e-8)
        self.grads = grads

        direction = -self.grads

        # Define 1D function for line search: phi(alpha) = loss(params + alpha * direction)
        def phi(alpha):
            return self.loss(self.params + alpha * direction)

        # Use scipy.optimize.minimize_scalar to find optimal alpha
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(phi, bounds=(0, 1), method='bounded')
        alpha_opt = res.x
        logger.info(f"Line search found optimal alpha: {alpha_opt}")

        # Update parameters
        self.params = self.params + alpha_opt * direction
        self.iterations += 1

        if self.record_history:
            self.log_history()

        logger.info(f"Line search step {self.iterations} complete. Returning updated parameters.")
        return self.params