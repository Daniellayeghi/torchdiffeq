# Neural ODEs with running loss.

This library provides the modified version of neural ODEs where the running loss $L$ is applied per for every time step.

$$
L\big(\mathbf{z}(t_1)) = \mathbf{z}(t_0) + \int_{t_0}^{t_1}{L\big(f(\mathbf{z}(t), t, \theta)}\big)
$$

This results in the integeration over the adjoints to contain additional derivatives $L_x$ and $L_{\theta}$. **This modeification only supports the Euler solver!**
