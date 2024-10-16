# Model based deep learning for successive convex approximation 
This is the official repository containing all the code to run the simulations and test the learning algorithm proposed in the master's thesis in Data Science  ***Deep-unrolled Successive Convex Approximation for
nonconvex sparse learning*** (a.y. 2023/24)

# Methodological background
+ Defined the sparse least squares problem with Difference-of-convex Sparsity Inducing Penalties:
  
$$\underset{\mathbf{x}}{\mathrm{min}} \  \frac{1}{2} \lVert \mathbf{y} - \mathbf{A}\mathbf{x}\rVert_2^2 + \lambda G(\mathbf{x})$$  
$$ G(\mathbf{x}) = \sum_{i=1}^m g(x_i) $$
$$ g(x_i) = g^+(x_i) - g^-(x_i) = \eta(\theta)|x_i| - [\eta(\theta)|x_i| - g(x_i)] $$

+ Solved it with Successive Convex Approximation, retrieving a fundamental functional form;
  $$\mathbf{x}^{k+1} = \mathcal{S}_{\frac{\lambda \eta(\theta)}{L}} \left[ \mathbf{x}^{k} - \frac{1}{L} \left( \mathbf{A}^T \mathbf{A} \mathbf{x}^{k} - \mathbf{A}^T \mathbf{y} + \lambda \Gamma_{\theta, \gamma}(\mathbf{x}^k)\right)\right]$$
+ Reproduction of the state-of-the art for what concerns deep sparse coders;
+ Design of deep sparse coders hinging on Difference-of-convex Sparsity Inducing Penalties, proving the necessary condition of convergence and the optimal upper bound over the reconstruction error that enabled our proposed model, *Analytical Learnable Difference-of-Convex Iterative Soft Thresholding Algorithm* (AL-DC-ISTA)
