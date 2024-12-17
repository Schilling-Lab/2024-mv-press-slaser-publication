from abc import ABC, abstractmethod


class PlotterABC(ABC):
    def __init__(self):
        self.setup_figure()
        self.setup_interactive()
        self.show()

    @abstractmethod
    def setup_figure(self) -> None:
        """Sets up matplotlib figure and axes.

        Use plt.ioff() here to create the figure!
            import matplotlib.pyplot as plt

            with plt.ioff():
                self.fig, self.ax = plt.subplots()

        Creating the figure inside the 'with' statement ensures the plot is only
        rendered when we explicitly call plt.show()
        """
        pass

    @abstractmethod
    def update(self) -> None:
        """Plot-Update function used by self.setup_interactive()."""
        pass

    @abstractmethod
    def setup_interactive(self) -> None:
        """Sets up the interactive ipywidgets part."""
        pass

    @abstractmethod
    def show(self) -> None:
        """Command to render the figure."""
        pass
