from . import _hyperbolictanc as hyperbolictanc

class HypTan():    

    def __init__(self, x):
        self.x = x        

    def calculate(self):
        return hyperbolictanc.py_tanh_impl(self.x)


