import sys

class ProgressBar:
    def __init__(self, width, title='', divider=2):
        self._divider = divider
        toolbar_width = width // divider
        sys.stdout.write("%s [%s]" % (title, " " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width + 1))
        self._counter = 0

    def update(self):
        self._counter += 1
        if self._counter % self._divider == 0:
            sys.stdout.write("#")
            sys.stdout.flush()

    def fin(self):
        sys.stdout.write("\n")
        self._counter = 0

    def __del__(self):
        self.fin()


