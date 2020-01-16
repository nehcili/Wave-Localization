# Landscape Ground State Landscape Neural Net
## Goal
The goal of this project is to show case the effectiveness of the landscape function by comparing a simple machine learning architecture (CNN + dense) when trained on
1. potential $V$ only, or
2. landscape $u$ only, or
3. both $u$ and $V$.
to predict the lower ground state energy of a Schrodinger's operator. 

More precisely,
- Given a potential $V$ on $L^2(Q_L)$ where $Q_L = [0, L] \cap \mathbb{Z}$ with periodic boundary condition
- Find the ground state eigenvalue of the disrete Schrodinger Hamiltonian $-\Delta+V$ where $-\Delta$ is the discrite Laplacian on $\mathbb{Z}$.

### Architecture
- CNN + fully connected NN
- implemented by hand as python classes (not using tf.keras.Sequential), to facilitate future upgrades
- No optimization such as image augmentation/rotation/translation

## Conclusion
We observe that after training a simple CNN+dense layers net to predict the ground state eigenvalues $\lambda$,

1. The NN which trained on the landscape function $u$ performed with an percent relative error of about 2.5%.
2. If training is done for the original potential $V$, the percent relative error is about 15%
3. When $u$ and $V$ are both given as input, the result is about 3%.

It the landscape function itself out performed the random potential about 6 times as well. 

