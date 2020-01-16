# Wave-Localization
ML for Simons Collaboration: Wave Localization

## Landscape Ground State Landscape Neural Net (LGSNN)
The goal of this project is to show case the effectiveness of the landscape function by comparing a simple machine learning architecture (CNN + dense) on different input features.

We observe that after training to predict the ground state eigenvalues $\lambda$,

1. The NN which trained on the landscape function $u$ performed with an percent relative error of about 2.5%.
2. If training is done for the original potential $V$, the percent relative error is about 15%
3. When $u$ and $V$ are both given as input, the result is about 3%.

It the landscape function itself out performed the random potential about 6 times as well. 

