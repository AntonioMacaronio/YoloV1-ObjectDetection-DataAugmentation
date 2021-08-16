import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

a = np.array([1, 2, 3])
b = torch.rand(3)
sns.lineplot(x = a, y = b)