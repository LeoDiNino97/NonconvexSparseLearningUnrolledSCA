# Model based deep learning for successive convex approximation 
This is the official repository containing all the code to run the simulations and test the learning algorithm proposed in the master's thesis in Data Science  ***Deep-unrolled Successive Convex Approximation for
nonconvex sparse learning*** (a.y. 2023/24)

# Methodological background
+ Defined the sparse least squares problem with Difference-of-convex Sparsity Inducing Penalties:
+ Solved it with Successive Convex Approximation, retrieving a fundamental functional form;
+ Reproduction of the state-of-the art for what concerns deep sparse coders;
+ Design of deep sparse coders hinging on Difference-of-convex Sparsity Inducing Penalties, proving the necessary condition of convergence and the optimal upper bound over the reconstruction error that enabled our proposed model, *Analytical Learnable Difference-of-Convex Iterative Soft Thresholding Algorithm* (AL-DC-ISTA):

# Results
+ Increased precision in support detection;
+ Increased robustness to measurement matrix conditioning;
+ Increased reconstruction performances

The model has been tested on real data in a naturale image denoising task over BSD500, outperforming ALISTA in terms of PSNR. 
  


