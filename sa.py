import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import pdb
import os
import argparse
import json
import datetime
import functools
import time
            

def compute_mat_prod(mats):
    """ Compute Matrix product of matrices from left to right i.e mats[0] @ mats[1], ..., mats[n] 
    
        mats: list of np.array 
        return np.array (matrix prod)
    """
    if len(mats) == 1:
        return mats[0]
    tmp = functools.reduce(np.dot, mats)
    return tmp


def complex_norm_squared(cvec):
    """ Compute the squared norm of a complex vector using conjugation  """
    #return np.linalg.norm(cvec).astype('float64')**2
    return np.vdot(cvec, cvec).astype('float64')


def visualize_start_end(Q):
    """ Show start and target state of atom """
    plt.plot(np.linspace(-1,1,Q.start_state.size), Q.start_state, 'r-', label='start_wave')
    plt.plot(np.linspace(-1,1,Q.target_state.size), Q.target_state, 'g-', label='end_wave')
    plt.legend()
    plt.show()


def visualize_fidelity(Q, protocol):
    """ Visualize result of protocol """
    def plot_transform(x):
        return np.abs(x)**2
    unitaries = [Q.unitaries[k] for k in protocol]
    U = compute_mat_prod(unitaries)
    fig, ax_list = plt.subplots(1, 3, figsize=(32, 16))
    grid = Q.hamiltonian_grid_size
    ax = ax_list[0]
    ax.plot(np.linspace(-1, 1, grid), plot_transform(Q.start_state.ravel()), 'm--', label='start_state')
    ax.plot(np.linspace(-1, 1, grid), plot_transform(Q.target_state.ravel()), 'y--', label='target_state')
    ax.plot(np.linspace(-1, 1, grid), plot_transform((U @ Q.start_state).ravel()), 'g--', label='U @ start_state')
    ax.legend()
    ax.set_title('start state, end_state, and transformed start state')

    steps = 5
    step_size = int(Q.N/ steps) # assume that is an int
    axes = ax_list[1]
    axes.plot(np.linspace(-1, 1, grid), plot_transform(Q.start_state.ravel()), 'm--', label='start_state')
    axes.plot(np.linspace(-1, 1, grid), plot_transform(Q.target_state.ravel()), 'y--', label='target_state')
    for i in range(0, steps, 1):
        tmp = unitaries[-i * step_size:]
        Utmp = compute_mat_prod(tmp)
        axes.plot(np.linspace(-1, 1, grid), plot_transform((Utmp @ Q.start_state).ravel()), label='U{0} @ start_state'.format(i*step_size))
    axes.set_title('Wavefunction after every {0} steps'.format(step_size))
    axes.legend()

    ax3 = ax_list[2]
    start = 5
    Ustart = unitaries[start:]
    Up = compute_mat_prod(Ustart)
    ax3.plot(np.linspace(-1, 1, grid), plot_transform(Q.start_state.ravel()), 'm--', label='start_state')
    ax3.plot(np.linspace(-1, 1, grid), plot_transform(Q.target_state.ravel()), 'y--', label='target_state')
    for i in range(start-1, -1, -1):
        Up = unitaries[i] @ Up
        ax3.plot(np.linspace(-1, 1, grid), plot_transform((Up @ Q.start_state).ravel()), label='U{0} @ start_state'.format(start-i))
    ax3.legend()
    ax3.set_title('Wavefunction last 5 steps')
    plt.tight_layout()

    
class Quantum():
    
    def __init__(self, m=1.0, T=0.1, N=40, L=1.1, Alpha=160, Beta=130, sigma=1.0/8.0, tweezer_grid_size=128, hamiltonian_grid_size = 32, 
                fix_start=False,  superposition=False, super_combination=None):
        """
        Class for Quantum Moves BringHomeWater Game Stochastic Ascent algorithm

        args:
        m: float - mass (always 1 i think)
        T: float, time for protocol 
        N: int, number of steps in protocol
        L: float, tweezer position parameter
        Alpha: float - movable tweezer force
        Beta: float - Stuck tweezer force
        sigma: float - std of gausian attraction field
        tweezer_grid_size: int - number of tweezer positions to try
        hamiltonian_grid_size: int - grid for hamiltonian approximation
        start: bool, 
        stepsize = delta x
        optimal fidelity is F >= 0.999
        """
        self.m = m # set to one so ignored in the paper after the beginning
        self.T = T
        self.N = N
        self.delta_t = T/N
        print('Time Per Step is: {0}'.format(self.delta_t))
        self.x_start = L/2
        self.x_end = -L/2
        self.Alpha = Alpha
        self.Beta = Beta
        self.sigma = sigma
        self.fix_start = fix_start
        self.superposition = superposition
        self.super_combination = super_combination        
        self.tweezer_grid_size = tweezer_grid_size
        self.tweezer_grid = np.linspace(-1, 1, self.tweezer_grid_size)
        self.tweezer_spacing = 1.0 / ((self.tweezer_grid_size-1) / 2.0)
        self.hamiltonian_grid_size = hamiltonian_grid_size
        self.hamiltonian_grid = np.linspace(-1, 1, self.hamiltonian_grid_size)
        self.hamiltonian_spacing = 1.0 /((self.hamiltonian_grid_size - 1.0) / 2.0)
        # in numerical gradient check we use something like h = 0.00001
        
        (hs, t, b) = self.make_hamiltonians()
        self.hamiltonians = hs
        self.t_mat = t
        self.b_mat = b

        print('Made {0} Hamiltonians'.format(len(self.hamiltonians)))
        assert len(self.hamiltonians) == self.tweezer_grid_size
        self.start_state = self.compute_initial_state()
        self.target_state = self.compute_target_state()
        self.unitaries = self.make_unitaries()

    def get_params(self):
        """ Get params as dict """
        params = {'m': self.m, 'N': self.N, 'T': self.T, 'fix_start': self.fix_start, 
                  'superposition': self.superposition, 'super_combination': self.super_combination}
        return params

    def make_hamiltonians(self, Alpha=None):
        """ Compute H(x_k) for x_k in np.linspace(-1,1, self.hamiltonian_grid_size) 
        
        All tridiagonal. So we should be able to optimize stuff if need be
        """
        if Alpha is None:
            Alpha = self.Alpha
        
        grid_size = self.hamiltonian_grid_size
        print('Make hamiltonians - grid size, Alpha: ', grid_size, Alpha)
        ### T MATRIX ###
        print('Make T matrix for finite difference approximation of p^2: -1 2 -1: Tridiagonal matrix')
        t_mat = np.diag(2 * np.ones(grid_size), k=0) + np.diag(- np.ones(grid_size-1), k=-1) + np.diag(- np.ones(grid_size-1), k=+1)        
        scale = 0.5 / (self.hamiltonian_spacing ** 2)
        print('T Scale: 0.5 / h^2', scale)
        t_mat = scale * t_mat
        # print('tri diag',np.diag(t_mat,k=-1), np.diag(t_mat,k=0), np.diag(t_mat, k=1))
        ### B MATRIX ###
        print('Make B Matrix exp(-(x - start_state)^2/(2std^2)))')
        std2 = 2 * self.sigma ** 2 # should be something
        exp_xi = np.exp(-((self.hamiltonian_grid - self.x_start) ** 2) / std2)
        b_mat = np.diag(self.Beta * exp_xi)
        ### A MATRIX 
        a_mat_list = list()
        for pos in self.tweezer_grid:
            grid_exponents  = (self.hamiltonian_grid - pos)**2
            grid_gaussians = np.exp(- grid_exponents / std2)
            ak = np.diag(Alpha * grid_gaussians)
            a_mat_list.append(ak)
        Hks = [t_mat - b_mat - ak for ak in a_mat_list] 
        print('hamiltonians done')    
        return Hks, t_mat, b_mat    

    def compute_smallest_eigenvector(self, h):
        """ Compute the target state as the eigenvector of the smallest eigenvalue of hamiltonian """
        print('finding smallest eigenvalue', h.shape)
        # val, vec = scipy.sparse.linalg.eigsh(h, k=1, which='SA')
        # print('smallest eigenvalue', val, 'vec norm', scipy.linalg.norm(vec))        
        # return vec
        val, vec = scipy.linalg.eigh(h)
        print('largest eigenvalue', np.max(val))
        idx = np.argmin(val)        
        print('Smallest eigenvalues ', val[idx])
        return vec[:, idx].astype(np.complex).reshape(-1, 1)
        
    def compute_initial_state(self):
        """ Compute the target state as the eigenvector of the smallest eigenvalue of hamiltonian"""
        print('Compute start state vector -  H=p^2/2m - 130 exp(-(x-x_start)^2/2\sigma).')
        h = self.t_mat - self.b_mat
        start_state =  self.compute_smallest_eigenvector(h)
        print('start_state norm', complex_norm_squared(start_state), start_state.dtype)
        return start_state

    def compute_target_state(self):
        """ Compute the target state as the eigenvector of the smallest eigenvalue of H """
        print('Compute target state vector - H=p^2/2m-130 exp(-(x-x_end)^2/2\sigma).')
        std2 = 2 * self.sigma ** 2
        exp_xi = np.exp(-((self.hamiltonian_grid - self.x_end) ** 2) / std2)
        e_mat = np.diag(self.Beta * exp_xi)
        self.e_mat = e_mat
        h = self.t_mat - e_mat
        end_state = self.compute_smallest_eigenvector(h)
        if self.superposition:
            sqr = np.sqrt(2)/2
            if self.super_combination is None:
                c1 = sqr + 0j
                c2 = 0.0 + sqr * 1j
            else:
                c1 = self.super_combination[0]
                c2 = self.super_combination[1]
            target_state = c1 * self.start_state + c2 * end_state
            print('superposition target_state norm', complex_norm_squared(target_state))
            assert np.allclose(complex_norm_squared(target_state), 1.0)
            return target_state
        print('target_state norm', complex_norm_squared(end_state))
        return end_state


    def compute_unitaries(self, hamiltonians):
        """ Fastest unitary computation found """
        print('compute unitaries by eigendecomposition of tridiagoanal matrices')
        start_time = time.time()
        out = []
        const = -self.delta_t * 1j #tf.constant(-self.delta_t, dtype=_ftype)

        for h in hamiltonians:
            #s1 = time.time()
            #vals, vecs = scipy.linalg.eigh(h)
            #s2 = time.time()
            #print('eigh time', s2-s1)
            #s3 = time.time()
            vals, vecs = scipy.linalg.eigh_tridiagonal(np.diag(h), np.diag(h, k=1))
            #s4 = time.time()
            #print('tridiangoanl time', s4-s3)
            Q = vecs
            R = np.transpose(Q)
            L = np.diag(vals)
            B = (Q@L)@R 
            #print(B)
            #print(hamiltonians[0])
            #print('diff\n', np.linalg.norm(B-hamiltonians[0]))
            #print('diff\n', B-hamiltonians[0])
            
            vals_exp = np.diag(np.exp(const * vals))
            #h0 = (tf.complex(real=Q, imag=tf_zero) @ vals_exp) @ tf.complex(real=R, imag=tf_zero)
            h0 = (Q.astype('complex128') @ (vals_exp)) @ R.astype('complex128')
            #print(h0.dtype)
            #print(unitaries[0].dtype)
            #print(h0-unitaries[0])
            # print('change to eigendecomposition and compare')
            out.append(h0)
        end_time = time.time()
        print('Time to make unitaries eig', end_time-start_time)

        return out

    def make_unitaries(self):
        fast_unitaries = self.compute_unitaries(self.hamiltonians)
        return fast_unitaries
        #print('Compute unitaries', self.hamiltonians[0].shape)
        #start_time = time.time()
        #hk = self.hamiltonians
        #const = self.delta_t * (-1j)
        #unitaries = [scipy.linalg.expm(h * const) for h in hk]
        #norm_diffs = [np.linalg.norm(x-y) for (x, y) in zip(fast_unitaries, unitaries)]
        #print('norm_diffs max', np.max(norm_diffs))
        #end_time = time.time()
        #print(f'Time to make unitaries {end_time-start_time}')
        # return unitaries
  
    def fidelity(self, protocol, unitaries):
        """ Compute fidelity of protocol with given unitaries """
        unitaries = [unitaries[k] for k in protocol]
        U = compute_mat_prod(unitaries)
        fidel_vec = self.target_state.T.conj() @ (U @ self.start_state)        
        fidel = complex_norm_squared(fidel_vec)        
        return fidel
    
    def stochastic_ascent(self, unitaries=None, protocol=None, goal=0.999):
        """ Run stochastic Ascent 
        
        Args:
         unitary matrices to use
         protocol: list, starting protocol
         goal: float, stop if this fidelity is achieved
        Returns:
         best_score: float - final fidelity
         protocol: list - final protocol
         path: list - fidelity after each epoch
         all_fidelities: list, fidelitites computed after each update
        """
        
        if unitaries is None:
            unitaries = self.unitaries
        print('unitary dtype', unitaries[0].dtype)
        fix_start = self.fix_start
        path = []
        all_fidelities = []
        identity = np.eye(self.hamiltonian_grid_size)
        if protocol is None:
            protocol = np.random.randint(low=0, high=len(unitaries), size=self.N).astype('int32')
        else:
            protocol = np.asarray(protocol)
        start_pos = int(self.tweezer_grid_size * (0.45/2))
        if fix_start:
            protocol[-1] = start_pos
            #print('set starting value to something', protocol[-1], self.tweezer_grid[protocol[-1]])

        # protocol should be x_n,...,x_1 - so the algorithm below does that slightly weird...
        # Seems i have ordering wroong
        best_score = self.fidelity(protocol, unitaries)
        path.append(best_score)
        print('Start Fidelity: {0}'.format(best_score))
        lp = len(protocol)
        start_time = time.time()
        while 1:
            improv = False
            total_improv = 0.0
            I = np.random.permutation(lp)
            round_start_time = time.time()
            for W in I:               
                w_idx = lp - W - 1
                    
                #print('Handling ', W, ' w_idx: ', w_idx)     - this indexing crap is annoying.           
                end_size = lp - W -1
                u_nw = protocol[:end_size]
                Uw_list = [unitaries[k] for k in u_nw]
                assert len(Uw_list) == end_size, '{0} - {1}'.format(len(Uw_list), end_size)
                Uw = compute_mat_prod(Uw_list) if len(Uw_list) > 0 else identity
                # should be prod_{i=W+1}^n U_ki
                phiw1 = self.target_state.T.conj() @ Uw

                start_size = W
                if start_size == 0:
                    u_w1 = []
                else:
                    u_w1 = protocol[-start_size:]
                    
                assert len(u_w1) == start_size
                
                tmp = list(u_nw) + [protocol[w_idx]] + list(u_w1)
                assert len(tmp) == len(protocol), (len(tmp), len(protocol))
                assert np.all([x ==y for x, y in zip(tmp, protocol)]), list(zip(tmp, protocol))
                U1_list = [unitaries[k] for k in u_w1]
                U1 = compute_mat_prod(U1_list) if len(U1_list)  > 0 else identity
                psiw1 = U1 @ self.start_state
                         
                best_k = protocol[w_idx]
                # old_k = best_k
                options = enumerate(unitaries)
                if fix_start and (w_idx == (lp-1)):
                    #print('fixed start just continue', w_idx, protocol[w_idx])
                    options = []                                        

                for k, Uk in options:                        
                    #U = Uw @ Uk @ U1
                    #Z = self.target_state.T.conj() @ (U @ self.start_state)
                    fid_vec = phiw1 @ Uk @ psiw1
                    #tmp = fid_vec - Z
                    #assert(scipy.linalg.norm(tmp) < 1e-5), "not close"
                    fid = complex_norm_squared(fid_vec)
                    #fid2 = scipy.linalg.norm(fid_vec)**2
                    # assert np.abs(fid-fid2) < 1e-5
                    all_fidelities.append(max(fid, best_score))
                    if fid > best_score: 
                        # print('new best', fid, '     -- improvement: ', fid-best_score)
                        total_improv += (fid - best_score)
                        if (fid - best_score) > 1e-7:
                            # print('improve enough ')
                            improv = True
                        best_score = fid
                        best_k = k
                        
                protocol[w_idx] = best_k
                #tmp_fidel = self.fidelity(protocol)      
                #v1 = np.linalg.multi_dot([self.target_state] + [unitaries[k] for k in protocol] + [self.start_state])
                #cv = complex_norm_squared(v1)
                #assert np.allclose(cv, tmp_fidel)
                #assert np.allclose(tmp_fidel, best_score), 'check fideilty please'
                #protocol[-1] = len(unitaries)-1
            round_end_time = time.time()
            print(f'time per epoch {round_end_time - round_start_time}')
            fid = self.fidelity(protocol, unitaries)
            path.append(fid)
            print(f'New fidelity: {fid} - improvement {total_improv}')
            if (not improv) and (total_improv < 5e-10):
                print('Not enough improvement - Stop this thing', total_improv)
                break
            if fid >= goal:
                print(f'goal {goal} reached {fid}')
                break
        end_time = time.time()
        total_time = end_time - start_time    
        print(f'total time used (s) {total_time}')
        return best_score, protocol, path, all_fidelities


def superposition(_params, repeats=1, start_protocol=None):
    """ Run a specific setting of the superposition Quantum Experiment and show the result  """    
    params = _params.copy()
    print('Run {0}'.format(str(params)))
    params['superposition'] = True
    print('superposition', params)
    if start_protocol is not None:
        start_protocol = np.asarray(start_protocol)
  
    Q = Quantum(**params)
    sqr2 = np.sqrt(2)/2
    combs = [(sqr2 + 0j, 0 + sqr2 * 1j)] # (sqr2 + 0j, sqr2 + 0j),(sqr2 + 0j, -sqr2 + 0j),
    fidelities = []
    protocols = []
    for c in combs:
        Q.super_combination = c
        best = 0
        best_protocol = []
        for i in range(repeats):
            best_score, protocol, _, _ =  Q.stochastic_ascent()        
            if best_score > best:
                best = best_score
                best_protocol = protocol
            if best_score >= 0.999:
                break 
        fidelities.append(best_score)
        protocols.append([int(x) for x in best_protocol])

    print('superposition fidelities:', fidelities)
    return Q, fidelities, protocols

def plot_protocol(tweezer_spacing, protocol):
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    # fix this constant
    xr = np.linspace(-1, 1, len(protocol))
    #ax.plot(xr,  -1 + np.array(protocol[::-1])*tweezer_spacing, '-o', color='green', linewidth=2, markerfacecolor="None", markersize=6)
    ax.step(xr,  -1 + np.array(protocol[::-1])*tweezer_spacing, '-', color='green', linewidth=2, markerfacecolor="None", markersize=6)
    print('actual protocol', -1 + np.array(protocol[::-1])*tweezer_spacing)
    ax.set_ylim([-1, 1])
    ax.set_xlabel('Time')
    ax.set_ylabel('Tweezer Positions')
    ax.set_title('Protocol Plot')
    return fig, ax


def visualize_run(Q, protocol, path):
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.plot(range(len(path)), np.abs(path), 'g-x', label='fidelity')
    ax.set_xlabel('epoch')
    ax.set_ylabel('fidelity')
    fig, ax = plt.subplots(1, figsize=(10,6))
    #ax.plot(range(len(protocol)), -1 + protocol[::-1]*Q.tweezer_spacing, 'go')
    ax.plot(np.array(list(range(len(protocol)))),  -1 + protocol[::-1]*Q.tweezer_spacing, color='g', linewidth=0,  marker='o', fillstyle=None, markerfacecolor="None")
    ax.set_title('protocol')
    ax.set_yticks(np.linspace(-1,1,21))
    visualize_fidelity(Q, protocol)
    plt.show()


def run(params, visualize=False, start_protocol=None):
    """ Run a specific setting of the Quantum Experiment and show the result  """    
    print('Run {0}'.format(str(params)))
    Q = Quantum(**params)
    best_score, protocol, path, all_fidelities = Q.stochastic_ascent(protocol=start_protocol)
    return Q, best_score, protocol, path, all_fidelities

def run_multi_amplitude(params, start_protocol=None):
    print('Run multi amplitude {0}'.format(str(params)))
    Q = Quantum(**params)
    amps = np.arange(100, 165, 5)
    unitaries = []
    for a in amps:
        h, _, _ = Q.make_hamiltonians(a)
        u = Q.compute_unitaries(h)
        unitaries.extend(u)
    best_score, protocol, path, all_fidelities = Q.stochastic_ascent(unitaries=unitaries, protocol=start_protocol)
    print('Achieved fidelity {0} in {1} iterations'.format( best_score, len(path)))
    print('The final protocol: ')
    print(protocol)
    print('The cost after each iteration')
    print(path)
    return Q, best_score, protocol

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-run', dest='run', action='store_true', default=False)
    parser.add_argument('-run_ma', dest='run_ma', action='store_true', default=False)    
    parser.add_argument('-run_super', dest='run_super', action='store_true', default=False)
    
    parser.add_argument('-hgrid', dest='hgrid', type=int, default=64, help='Size of grid used for hamiltonians')
    parser.add_argument('-T', dest='T', type=float, default=0.1, help='Time/Duration T to run experiment')
    parser.add_argument('-tweezer', dest='tweeze', type=int, default=128, help='Size of Tweezer Grid')
    parser.add_argument('-N', dest='N', type=int, default=40, help='Number of steps in protocol')
    parser.add_argument('-A', dest='A', type=int, default=160, help='Intentisity of Moving Tweezer - 160 as defined in Dries paper')
    parser.add_argument('-B', dest='B', type=int, default=130, help='Intensity of trapping Tweezer - 130 is defined in nature paper')
    parser.add_argument('-superposition', dest='superposition', action='store_true', default=False, help='Set target state to superposition')
    parser.add_argument('-start', dest='start', action='store_true', default=False, help='Fix start to -L/2')
    parser.add_argument('-visualize', dest='visualize', action='store_true', default=False, help='Visualize run')
    
    args = parser.parse_args()
    params = {'T': args.T, 'N': args.N, 'tweezer_grid_size': args.tweeze, 'hamiltonian_grid_size': args.hgrid, 
              'Alpha': args.A, 'Beta': args.B, 'fix_start': args.start, 
              'superposition': args.superposition
              }
    assert params['T'] < 1, 'Bad time more than one'
    print('Args Used', args)
    start_protocol = None
    if args.run:
        Q, best_score, protocol, path, all_fid = run(params, False, start_protocol=start_protocol)

        print('Achieved fidelity {0} in {1} iterations'.format(best_score, len(path)))
        print('The final protocol: ')
        print(protocol)
        print('The cost after each iteration')
        print(path)
        if args.visualize:
            fig, _ = plot_protocol(Q.tweezer_spacing, protocol)
            visualize_fidelity(Q, protocol)
            plt.show()
    if args.run_ma:
        Q, best_score, protocol = run_multi_amplitude(params)

        #plt.show()
    if args.run_super:
        superposition(params, args.repeat, start_protocol=start_protocol)
