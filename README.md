# Wave-Localization
ML for Simons Collaboration: Wave Localization

## 1. The eigenvalue problem
The goal of this project is to show case the effectiveness of the landscape function by comparing the result of training a simple machine learning architecture (CNN + dense) on different input features ($1/u$ vs $V$).

### Mathematical Setting
The underlying Hilbert space is $L^2(Q_L)$ where $Q_L = [0, L] \cap \mathbb{Z}$ with periodic boundary condition. $V$ is a non-negative real valued potential on $Q_L$. We form the Hamiltonian
\[
	H = -\Delta + V
\]
where $-\Delta$ is the discrete Laplacian on $Q_L$. We would like to compute/approximate
- all (or as many as possible) eigenvalues of $H$ and
- the density of states, $N_V(E)$, of $H$.

### Data generation
FENICS

### LGSNN
<b>Architecture</b>
- CNN + fully connected NN
- implemented by hand as python classes (not using tf.keras.Sequential), to facilitate future upgrades
- No optimization such as image augmentation/rotation/translation

### LEVNN
<b>Architecture</b>
- Essentially the same as LGSNN
- CNN + fully connected NN, with output dimn = number of eigevanlues required


### Observations
<b>Training</b>
We trained LGSNN/LEVNN on the first/first 20 eigenvalues with 
- interval length: $L= 1000$ (LGSNN) or $L=1024$ (LEVNN)
- data size: 6000 ground states eigenvalues (LGSNN) or 5000 lists of first 20 eigenvalues
- epocsh: 40 (LGSNN) or 30 (LEVNN)
- batch size: 32 (LGSNN) or 16 (LEVNN)
- Random uniform potential with values in $[0,4]$ is used.

<b>Results</b>
1. The NN which trained on the landscape function $u$ performed with an percent relative error of about $O(1\%)$
2. If training is done for the original potential $V$, the percent relative error is about $O(15 \%)$.
3. In the LGSNN case, when $u$ and $V$ are both given as input, the result is about 3%.

It the landscape function itself out performed the random potential about 10 times as well. 


## 2. Density of state (DOS) computation
### Mathematical Setting
Same as the case for the eigenvalue problem

### Data generation
Same as the case for the eigenvalue problem

### DOSNN (v1)
<b>Architecture</b>
- Input = n
- creates $3n$ box counting functions which counts # of boxes such that
	- inf potential < E
	- sup potential < E
	- average potential < E
where each box has side length $1.6^n$.
- train a 3-layer dense NN with input [potential-E] and outputs 3n weights
- returns the dot product of the 2n weights and the 3n results from box counting


### Observations
<b>Training</b>
We trained DOSNN on the first eigenvalues
- interval length: $L=200$
- data size: 1000 lists of first 20 eigenvalues
- epocsh: 40 
- batch size: 16
- Random uniform potential with values in $[0,4]$ is used.
- target = i if input is $(E_i, V)$ where $E_i$ is the i-th eigenvalue of $-\Delta+V$.

<b>Results</b>
- The NN trained on both $1/u$ and $V$ performed equally well with a test mean squared error of 2 and train mean squared error < 1. 
- Training for the $1/u$ case seems to be a little unstable.
- Predictions of both NN seems to be concentrated near a few integers.

## 3. Data
I removed all the data due to their large size. But they can be easity generated simply by running the data generation jupyter notebook


## 4. Current direction
- Train the neural network on 2D data
- Train the neural network on various random potentials
- Scale up DOS and compare with the usual box counting
- Can one train $C_1$ and $C_2$?
- Improving DOS by learning the deciding function in box counting
- Studying regime in which the landscape starts to fail
	- x% of the V max?
	- what does the neural net actually learn about u that's better than V, if at all?
- General: be more model specific, capture more leading order terms.



