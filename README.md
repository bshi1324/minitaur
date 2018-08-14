# minitaur
Pybullet Minitaur with Distributional Policy Gradients (D4PG).

Dependencies:
* ptan
* PyTorch
* gym
* pybullet

Using Maxim Lapan's baseline D4PG model, we implement Leaky RELU and a prioritized replay buffer for the minitaur pybullet environment. Exact hyperparameters for the buffer still need to be tuned.  Agent and experience source are implemented through ptan, a library of helpful wrappers for gym.
