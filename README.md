# Stochastic-Growth-with-Discrete-Choice
Stochastic Growth with Discrete Choice

### Model

Production: 

$$
Y_t = A_t K_t^\alpha L_t^{1-\alpha}.
$$

TFP envolution:

$$
\log A_{t+1} = \rho \log A_t + \epsilon_{t+1},\\
\epsilon_{t}\sim\mathcal{N}(0, \sigma^2).
$$

Period utility:

$$
u_t = \frac{c_t^{1-\gamma}}{1- \gamma} + B\frac{(\bar{L} - L_t)^{1-\eta}}{1- \eta}.
$$

Maximization objective:

$$
\sum\limits_{t=0}^\infty \beta^t u_t.
$$

Budget constraint:

$$
c_t + K_{t+1} = Y_t + (1-\delta)K_t.
$$

First-order condition:

$$
c_t^{-\gamma} = \beta\mathbb{E}_t\Bigg\{ c_{t+1}^{-\gamma} \Big[\alpha A_{t+1}(\frac{L_{t+1}}{K_{t+1}})^{1-\alpha} + 1 - \delta\Big] \Bigg\},\\
c_t^{-\gamma} (1-\alpha) A_t (\frac{K_t}{L_t})^{\alpha}= B(\bar{L}-L_t)^{-\eta}, \text{if L is continuous.}
$$

Steady state:

$$
Y=K^{\alpha}L^{1-\alpha}\\
c+\delta K = Y\\
\alpha (\frac{L}{K})^{1-\alpha} + 1 - \delta = \frac{1}{\beta}\\
c^{-\gamma} (1-\alpha) (\frac{K}{L})^{\alpha}= B(\bar{L}-L)^{-\eta}
$$