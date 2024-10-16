# Model based deep learning for successive convex approximation 
This is the official repository containing all the code to run the simulations and test the learning algorithm proposed in the master's thesis in Data Science  ***Deep-unrolled Successive Convex Approximation for
nonconvex sparse learning*** (a.y. 2023/24)

# Methodological background
+ Defined the sparse least squares problem with Difference-of-convex Sparsity Inducing Penalties:
  
$$\underset{\mathbf{x}}{\mathrm{min}} \  \frac{1}{2} \lVert \mathbf{y} - \mathbf{A}\mathbf{x}\rVert_2^2 + \lambda G(\mathbf{x})$$  
$$ G(\mathbf{x}) = \sum_{i=1}^m g(x_i) $$
$$ g(x_i) = g^+(x_i) - g^-(x_i) = \eta(\theta)|x_i| - [\eta(\theta)|x_i| - g(x_i)] $$

| **Penalty function** | **g(x)** | **η(θ)** |
|----------------------|----------|-----------|
| Exp [EXP]            | \(1 - e^{-\theta|x|}\) | θ |
| \(\ell_p (p < 0)\) [PNEG] | \(1 - (\theta|x| + 1)^p\) | \(-p\theta\) |
| SCAD [SCAD] | \(\begin{cases} \frac{2\theta}{a+1}|x|, & 0 \leq |x| \leq \frac{1}{\theta} \\ \frac{-\theta^2|x|^2 + 2a\theta|x| - 1}{a^2 - 1}, & \frac{1}{\theta} < |x| \leq \frac{a}{\theta} \\ 1, & |x| > \frac{a}{\theta} \end{cases}\) | \(\frac{2 \theta}{a + 1}\) |
| Log [LOG]            | \(\frac{\log(1 + \theta|x|)}{\log(1 + \theta)}\) | \(\frac{\theta}{\log(1 + \theta)}\) |

+ Solved it with Successive Convex Approximation, retrieving a fundamental functional form;
+ Reproduction of the state-of-the art for what concerns deep sparse coders;
+ Design of deep sparse coders hinging on Difference-of-convex Sparsity Inducing Penalties, proving the necessary condition of convergence and the optimal upper bound over the reconstruction error that enabled our proposed model, *Analytical Learnable Difference-of-Convex Iterative Soft Thresholding Algorithm* (AL-DC-ISTA)
