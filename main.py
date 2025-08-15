import torch

class Test:
    def __init__(self):
        self.nn = torch.nn.Linear(1, 1)
    
    def compute(self, x):
        return self.nn(x)
    
    def sample(self, x):
        with torch.no_grad():
            x = self.compute(x)
        return x
    
t = Test()
x = torch.tensor([1]).float()
print(t.compute(x))
print(t.sample(x))
