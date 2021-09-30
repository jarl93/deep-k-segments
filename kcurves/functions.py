# libraries
import numpy as np

class Function():
    def __init__(self, interval, char_to_plot='o', color_to_plot='red'):
        """
        Class to define a function.
        Arguments:
            interval: pair [a, b] with the domain of the function.
            char_to_plot: character used to plot noisy data.
            color_to_plot: color used to plot the function.
        """
        self.interval = interval
        self.char_to_plot = char_to_plot
        self.color_to_plot = color_to_plot
        self.x = None
        self.y = None
        self.vec = None


class FunctionSin(Function):
    def __init__(self, amp, frec, interval, shift, char_to_plot, color_to_plot):
        """
        Class to define a sine wave function.
        Arguments:
            amp: amplitude of the function.
            frec: frequency of the function.
            interval: pair [a, b] with the domain of the function.
            shift: numerical value that shifts the function along the y axis.

        """
        super(FunctionSin, self).__init__(interval, char_to_plot, color_to_plot)
        self.amp = amp
        self.frec = frec
        self.shift = shift
        self.y_noisy = None
        self.epsilon = None

    def generate_data(self, num_samples=1000):
        """
        Generates the nosy data for the function, given the number of samples.
        Arguments:
            num_samples: number of points which will be uniformly
                 distributed in interval domain of the function.

        """
        # Get the extremes of the interval
        a, b = self.interval
        # Generate the points from the interval
        self.x = np.linspace(a, b, num=num_samples)

        # Generate some noise from a Gaussian distribution
        self.epsilon = np.random.normal(0, 1, num_samples)

        # Compute y = f(x) and the noisy data
        self.y = self.amp * np.sin(2 * np.pi * self.frec * self.x) + self.shift
        self.y_noisy = self.amp * np.sin(2 * np.pi * self.frec * self.x) + self.shift + self.epsilon

        self.vec = np.transpose(np.vstack((self.x, self.y_noisy)))

        return None