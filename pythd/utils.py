class open_or_use:
    def __init__(self, fname, f, mode):
        if f is None:
            self.fname = fname
            self.needs_close = True
            self.mode = mode
        else:
            self.f = f
            self.needs_close = False
    def __enter__(self):
        if self.needs_close:
            self.f = open(self.fname, self.mode)
            return self.f
        else:
            return self.f
    def __exit__(self, type, value, traceback):
        if self.needs_close:
            self.f.close()
