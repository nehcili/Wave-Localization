class DOS_data_generator():
    # self, list of int, method, int/None, int/None, bool
    def __init__(self, size, V_gen=None, max_it=None, tol=None, periodic=True):
        self.max_it = max_it
        self.tol=tol
        self.periodic = True
        
        self.V_gen = V_gen
        if V_gen==None:
            self.V_gen = np.random.rand
            
        
        if type(size) == int:
            self.size = [1,size]
        else:
            self.size = size
            
        self.one = PETSc.Vec().createSeq(self.size[1]) 
        self.one[:] = np.ones(self.size[1])
        
    # self, method --> PETSc Mat
    # Makeing a periodic problem Hamiltonian -\Delta+V
    def makeHamiltonian(self, V):
        n = self.size[1]
        A = PETSc.Mat().create()
        A.setSizes([n, n])
        A.setUp()

        rstart, rend = A.getOwnershipRange()

        # first row
        if rstart == 0:
            A[0, :2] = [2, -1]
            rstart += 1
        # last row
        if rend == n:
            A[n-1, -2:] = [-1, 2]
            rend -= 1
        # other rows
        for i in range(rstart, rend):
            A[i, i-1:i+2] = [-1, 2+V[i], -1]
        # Periodic condition
        if self.periodic:
            A[rstart,rend-1] = -1
            A[rend-1, rstart] = -1

        A.assemble()

        return A

    # self, PETSc Mat --> gs_ev class
    # compute the ground state eigenvalue
    # return -1 if numerical solver is divergent
    def compute_gs_ev(self, Hamiltonian):
        E = SLEPc.EPS()
        E.create()

        E.setOperators(Hamiltonian)
        E.setProblemType(SLEPc.EPS.ProblemType.HEP)
        E.setTolerances(tol=self.tol, max_it=self.max_it)
        E.setWhichEigenpairs(E.Which.SMALLEST_REAL)

        E.solve()

        ev = -1
        if E.getConverged():
            # Create the results vectors
            vr, wr = Hamiltonian.getVecs()
            vi, wi = Hamiltonian.getVecs()
            ev = E.getEigenpair(0, vr, vi)
        
        
        return ev.real
    
    # self, PETSc.Mat, bool --> PETSc.Vec
    # use PETSc.Vec.getArray() to convert result to np.ndarray
    def compute_landscape(self, Hamiltonian, show=False):        
        # Create solution landscape function u
        u = PETSc.Vec().createSeq(self.size[1])
        
        # Initialize ksp solver.
        ksp = PETSc.KSP().create()
        ksp.setOperators(Hamiltonian)
        
        # Solve!
        ksp.solve(self.one, u)

        # # Use this to plot the solution (should look like a sinusoid).
        if show:
            pylab.plot(u.getArray())
            pylab.show()
            
        return u   
    
    
    # self, method --> np.ndarray, np.ndarray, np.ndarray
    def data_gen(self, V_gen=None):
        if V_gen == None:
            V_gen = self.V_gen
        
        #print(self.size)
        #print(V_gen)
        self.V = V_gen(*self.size)
        self.ev = np.empty(self.size[0], dtype=np.float32)
        self.u = np.empty(self.size, dtype=np.float32)
        
        
        
        for i in range(self.size[0]):
            Hamiltonian = self.makeHamiltonian(self.V[i])
            self.ev[i] = self.compute_gs_ev(Hamiltonian)
            self.u[i] = self.compute_landscape(Hamiltonian).getArray()
    
            
        return self.ev, self.u, self.V
    
    # self, method --> np.ndarray, np.ndarray, np.ndarray
    def data_gen_bc(self, V_gen=None):
        if V_gen == None:
            V_gen = self.V_gen
        
        #print(self.size)
        #print(V_gen)
        self.V = V_gen(*self.size).astype(np.float32)        
        self.ev = np.empty(self.size[0], dtype=np.float32)
        self.u = np.empty(self.size, dtype=np.float32)
        self.tp = np.empty(self.size[0], dtype=np.float32)
        
        
        for i in range(self.size[0]):
            Hamiltonian = self.makeHamiltonian(self.V[i])
            self.ev[i] = self.compute_gs_ev(Hamiltonian)
            self.u[i] = self.compute_landscape(Hamiltonian).getArray()
            self.tp[i] = CONV_CONST/np.max(self.u[i]) # CONV_CONST = np.pi**2 / 8

        return self.ev, self.tp, self.u, self.V
