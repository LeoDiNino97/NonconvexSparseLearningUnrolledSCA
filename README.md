# Model based deep learning for successive convex approximation 
This is the official repository containing all the code to run the simulations and test the learning algorithm proposed in the master's thesis in Data Science  ***Deep-unrolled Successive Convex Approximation for
nonconvex sparse learning*** (a.y. 2023/24)

# Methodological background
+ Defined the sparse least squares problem with Difference-of-convex Sparsity Inducing Penalties:
+ Solved it with Successive Convex Approximation, retrieving a fundamental functional form;
+ Reproduction of the state-of-the art for what concerns deep sparse coders;
+ Design of deep sparse coders hinging on Difference-of-convex Sparsity Inducing Penalties, proving the necessary condition of convergence and the optimal upper bound over the reconstruction error that enabled our proposed model, *Analytical Learnable Difference-of-Convex Iterative Soft Thresholding Algorithm* (AL-DC-ISTA):

# Results

+ Increased precision in support detection;<br>
  <img src="https://github.com/user-attachments/assets/340f75a2-18ae-4d15-be65-0b91583e1f33" alt="Precision in Support Detection" width="400"/>

+ Increased robustness to measurement matrix conditioning;<br>
  <img src="https://github.com/user-attachments/assets/ff7c14b4-39e8-4716-ad31-9518acc1b26f" alt="Robustness to Measurement Matrix Conditioning" width="400"/>

+ Increased reconstruction performances;<br>
  <img src="https://github.com/user-attachments/assets/e41ff15f-7368-434b-958e-d3ad36046a15" alt="Reconstruction Performances" width="400"/>

The model has been tested on real data in a natural image denoising task over BSD500, outperforming ALISTA in terms of PSNR. 

<img src="https://github.com/user-attachments/assets/0b6540ca-feac-415c-9806-a7a65ab27dc1" alt="Natural Image Denoising Task" width="400"/>



