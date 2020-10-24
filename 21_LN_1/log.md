# Landscape Net 2.0

### Major Project structure update
- I can't find helpful tf.Data method to help me load data.
- I am writing generators for Model.fit



### Project structure
- benchmark.py has to be in both models and the main directory.
  - the file in the main directory is the newest. Every time it is updated, must also update the one in models.

### Data format
#### Notation
All notation follows python convention.
  - "...{name}..." means "...apple..." if name = apple
  - special symbols in string is proceeded by a slash (\\). i.e. "\n" for newline and "\\_ " for under line etc.

#### Data naming convention
The data_name of a data point is

data_name = "{domain_type}\_{domain_size}\_{counter}"

where
  - domain_type is one of "discrete" or "continuous"
  - domain_size is size of the domain (1000, 10000, etc...)
  - counter is an integer string that indexes the data point (counter = 0,1,2,3,...).

#### Data structure
A data point comes with an overall name [data-name]. The associate file in a data point is
1. "{data_name}_info.txt": contains Information of the potential (to ensure reproducibility). The file contains exactly the following lines
  - (string) potential type (uniform, bernoulli etc)
  - (int) domain size
  - (np.float32) pmax
  - (np.float32) argument passed to the potential (e.g. p = probability of getting 0 in a bernoulli distribution)
  - (int or np.float32) seed
  - (string) how was the potential generated? (matlab? python? which random number generator?)
2. "{data_name}_potential.txt": contains exactly 2 columns.  Each entry on each line is delimited by "," (comma). The landscape potential $u$ is required to be error checked and satisfy 1/pmax <= u <= 1/pmin.
  - First column = original potential (np.float32) the original potential V
  - Second column = W (np.float32) the landscape potential 1/u
3. "{data_name}_nve.txt": contains pairs of $E$ and associated eigenvalue counting $N_V(E)$.
  - This files contains $N$ lines, where $N$ is the number of pairs $(E, N_V(E))$.
  - Each line contains 2 numbers, delimited by "," (comma):
    - (np.float32) E
    - (np.float32) N_V(E) eigenvalue counting

#### Data size
- Each file contains 1 potential and
- 100 pairs $(E, N_V(E))$ of that potential
