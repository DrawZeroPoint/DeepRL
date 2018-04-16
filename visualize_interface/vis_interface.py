import matplotlib
import matplotlib.pyplot as plt


class VisInterface:

    def __init__(self, title):
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title(title, fontsize=14)

    def plot_image(self, array):
        self.ax.imshow(array, interpolation='none')
        plt.pause(0.001)
        self.figure.show()


def plot_image(array):
    plt.figure()
    plt.ion()
    plt.imshow(array, interpolation='none')
    plt.title('Example extracted screen')
    plt.show()
