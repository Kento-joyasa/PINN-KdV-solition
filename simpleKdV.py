import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import griddata

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = nn.Tanh()

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

def pde_loss(model, x):
    x.requires_grad = True
    y = model(x)
    u = y[:, 0:1]

    u_t = torch.autograd.grad(u, x, grad_outputs = torch.ones_like(u), create_graph = True)[0][:, 1:2]
    u_x = torch.autograd.grad(u, x, grad_outputs = torch.ones_like(u), create_graph = True)[0][:, 0:1]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs = torch.ones_like(u_x), create_graph = True)[0][:, 0:1]
    u_xxx = torch.autograd.grad(u_xx, x, grad_outputs = torch.ones_like(u_xx), create_graph = True)[0][:, 0:1]

    f_u = u_t + 6 * u * u_x + u_xxx
    return f_u

def boundary_cond1(x):
    T = x[:, 1:2]
    return 0.125 * (1 / torch.cosh(4.5 + 0.625 * T)) ** 2
def boundary_cond2(x):
    T = x[:, 1:2]
    return 0.125 * (1 / torch.cosh(5.5 - 0.624 * T)) ** 2
def initial_cond(x):
    X = x[:, 0:1]
    return 0.125 *(1 / torch.cosh(1.75 + 0.25 * X)) ** 2

x_lower, x_upper = -20.0, 20.0
t_lower, t_upper = -20.0, 20.0
x = np.linspace(x_lower, x_upper, num = 256)
t = np.linspace(t_lower, t_upper, num = 400)
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
X_star = torch.tensor(X_star, dtype = torch.float32)

layers = [2] + [30] * 5 + [1]

model = PINN(layers)

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)


epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()

    f_u = pde_loss(model, X_star)
    loss_pde = torch.mean(f_u ** 2)

    b_1_loss = torch.mean((model(X_star)[:, 0:1] - boundary_cond1(X_star)) ** 2)
    b_2_loss = torch.mean((model(X_star)[:, 0:1] - boundary_cond2(X_star)) ** 2)
    ini_loss = torch.mean((model(X_star)[:, 0:1] - initial_cond(X_star)) ** 2)

    loss_all = loss_pde + b_1_loss + b_2_loss + ini_loss
    loss_all.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss_all.item()}")
prediction = model(X_star).detach().numpy()
u = griddata(X_star.numpy(), prediction[:, 0], (X, T), method = "cubic")
# 保存模型
torch.save(model.state_dict(), 'pinn_model.pth')

k1 = 0.5
c = 1
exact = k1 / 2 * (1 / np.cosh(1 / 2 * (k1 * X + 0.5 * T)) + c) ** 2
np.savetxt("pred.txt", u)
np.savetxt("exact.txt", exact)






















