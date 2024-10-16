# Model based deep learning for successive convex approximation 
This is the official repository containing all the code to run the simulations and test the learning algorithm proposed in the master's thesis in Data Science  ***Deep-unrolled Successive Convex Approximation for
nonconvex sparse learning*** (a.y. 2023/24)

# Methodological background
+ Defined the sparse least squares problem with Difference-of-convex Sparsity Inducing Penalties:
  ![image](https://github.com/user-attachments/assets/7d409319-aa42-4591-84b5-3b46e7cb0a8e)
  ![image](https://github.com/user-attachments/assets/7846d473-c33e-4aad-9908-7b9c84ecbc64)
  ![image](https://github.com/user-attachments/assets/5326224f-56fc-49f8-afde-bcd5521274f7)


+ Solved it with Successive Convex Approximation, retrieving a fundamental functional form;
![image](https://github.com/user-attachments/assets/6b5a785a-d832-42a6-972e-4251650366c4)

+ Reproduction of the state-of-the art for what concerns deep sparse coders;
+ ![image](https://github.com/user-attachments/assets/93a0d2bb-92d2-4dea-b096-a3805a35a4b6)

+ Design of deep sparse coders hinging on Difference-of-convex Sparsity Inducing Penalties, proving the necessary condition of convergence and the optimal upper bound over the reconstruction error that enabled our proposed model, *Analytical Learnable Difference-of-Convex Iterative Soft Thresholding Algorithm* (AL-DC-ISTA):
  ![image](https://github.com/user-attachments/assets/cc5b3ebe-67ce-4447-8d9c-8e9452ec3ec1)

