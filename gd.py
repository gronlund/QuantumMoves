import tensorflow as tf
import numpy as np
import time
import argparse
import functools
import matplotlib.pyplot as plt
import datetime
import os 
import pdb 
import json
from scipy.interpolate import interp1d


_ftype = tf.float64


def compute_mat_prod(mats):
    """ Compute Matrix product of matrices from left to right i.e mats[0] @ mats[1], ..., mats[n] 
    
        mats: list of matrices
        return matrix prod
    """
    if len(mats) == 1:
        return mats[0]
    tmp = functools.reduce(tf.matmul, mats)
    return tmp


def compute_smallest_eigenvector(h):
    """ Compute the smallest eigenvector of h, return as complex """
    val, vec = tf.linalg.eigh(h)
    idx = tf.argmin(val)        
    #print('Smallest eigenvalues ', val[idx])
    _zero = tf.constant(0, dtype=_ftype)
    return tf.reshape(tf.complex(real=vec[:, idx], imag=_zero), (-1, 1))


def complex_norm_squared(cvec):
    """ Compute the squared norm of a complex vector  """
    res = tf.real(tf.linalg.norm(cvec))**2
    #res2 =  tf.real(tf.tensordot(tf.transpose(tf.conj(cvec)), cvec, axes=1))
    return res


class GradQuantum():

    def __init__(self, m=1.0, T=0.1, N=40, L=1.1, Alpha=160, Beta=130, sigma=1.0/8.0, hamiltonian_grid_size = 128, 
                superposition=False, super_combination=None):
        """
        Class for Quantum Moves BringHomeWater Game algorithm

        args:
        m: float - mass (always 1 i think)
        T: float, time for protocol 
        N: int, number of steps in protocol
        L: float, tweezer position parameter
        Alpha: float - movable tweezer force
        Beta: float - Stuck tweezer force
        sigma: float - std of gausian attraction field
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
        self.superposition = superposition
        self.super_combination = super_combination
        
        self.t_mat = None # finite difference approximation of p^2 -1 2 -1: Tridiagonal matrix'
        self.b_mat = None # B Matrix exp(-(x - start_state)^2/(2std^2)))'
        #print('tweezer grid\n', self.tweezer_grid)
        #self.tweezer_spacing = 1.0 / ((self.tweezer_grid_size-1) / 2.0)
        self.hamiltonian_grid_size = hamiltonian_grid_size
        self.hamiltonian_grid = tf.lin_space(start=tf.constant(-1.0, dtype=_ftype), stop=1.0, num=self.hamiltonian_grid_size)
        self.hamiltonian_spacing = 1.0 /((self.hamiltonian_grid_size - 1.0) / 2.0)
        # in numerical gradient check we use something like h = 0.00001
        self.make_fixed_matrices()
        self.start_state = self.compute_initial_state()
        self.target_state = self.compute_target_state()
        assert self.target_state.shape == (self.hamiltonian_grid_size, 1), 'bad states'
    
    def make_fixed_matrices(self):
        grid_size = self.hamiltonian_grid_size
        ### T MATRIX ###
        #print('Make T matrix for finite difference approximation of p^2: -1 2 -1: Tridiagonal matrix')
        np_tmat = np.diag(2.0 * np.ones(grid_size), k=0) + np.diag(-np.ones(grid_size-1), k=-1) + np.diag(-np.ones(grid_size-1), k=+1)
        t_mat = tf.convert_to_tensor(np_tmat, dtype=_ftype)
        scale = tf.constant(0.5 / (self.hamiltonian_spacing ** 2), dtype=_ftype)
        #print('T Scale: 0.5 / h^2', scale)
        t_mat = scale * t_mat
        self.t_mat = t_mat
        ### B MATRIX ###
        #print('Make B Matrix exp(-(x - start_state)^2/(2std^2)))')
        std2 = tf.constant(2 * self.sigma ** 2, dtype=_ftype) # should be something
        diag_entries = -((self.hamiltonian_grid - self.x_start) ** 2) / std2
        exp_xi = tf.exp(diag_entries)
        b_mat = tf.diag(self.Beta * exp_xi)
        self.b_mat = b_mat


    def compute_initial_state(self):
        """ Compute the target state as the eigenvector of the smallest eigenvalue of H(something) """
        #print('Compute start state vector -  H=p^2/2m - 130 exp(-(x-x_start)^2/2\sigma).')
        h = self.t_mat - self.b_mat
        start_state = compute_smallest_eigenvector(h)
        #print('start_state norm', complex_norm_squared(start_state))
        return start_state

    def compute_target_state(self):
        """ Compute the target state as the eigenvector of the smallest eigenvalue of H(something) """
        #print('Compute target state vector - H=p^2/2m-130 exp(-(x-x_end)^2/2\sigma).')
        std2 = 2 * self.sigma ** 2
        exp_xi = tf.exp(-((self.hamiltonian_grid - self.x_end) ** 2) / std2)
        e_mat = tf.diag(self.Beta * exp_xi)
        self.e_mat = e_mat
        h = self.t_mat - e_mat
        end_state = compute_smallest_eigenvector(h)
        if self.superposition:
            sqr = tf.sqrt(2)/2
            if self.super_combination is None:
                c1 = sqr + 0j
                c2 = 0.0 + sqr * 1j
            else:
                c1 = self.super_combination[0]
                c2 = self.super_combination[1]
            target_state = c1 * self.start_state + c2 * end_state
            print('superposition target_state norm', complex_norm_squared(target_state))
            # assert tf.allclose(complex_norm_squared(target_state), 1.0)
            return target_state
        #print('target_state norm', complex_norm_squared(end_state))
        return end_state

    def make_hamiltonians(self, positions, amplitudes=None):
        """ Create Hamiltonian Matrices 
            returns: list tf.tensor(), dtype _ftype
        """
        #print('B matrix diagonal:\n', b_mat.diagonal())
        ### A MATRIX 
        a_mat_list = list()
        std2 = tf.constant(2.0 * self.sigma ** 2, dtype=_ftype) # should be something
        if amplitudes is None:
            amplitudes = [tf.constant(self.Alpha, dtype=_ftype) for _ in range(positions.shape[0])]
        for idx in range(positions.shape[0]):
            pos = positions[idx]   
            amp = amplitudes[idx]         
            grid_exponents  = (self.hamiltonian_grid - pos)**2
            grid_gaussians = tf.exp(- grid_exponents / std2)
            ak = tf.diag(amp * grid_gaussians)
            a_mat_list.append(ak)
        Hks = [self.t_mat - self.b_mat - ak for ak in a_mat_list] 
        #print('hamiltonians done')       
        return Hks

    def compute_unitaries_expm(self, hamiltonians):
        """ Made for time comparison - it is slower """
        #print('Compute unitaries - shape', hamiltonians[0].shape)
        start_time = time.time()
        const = tf.constant(-self.delta_t, dtype=_ftype)
        tf_zero = tf.constant(0, dtype=_ftype)
        unitaries = [tf.linalg.expm(tf.complex(real=tf_zero, imag=h*const)) for h in hamiltonians]
        end_time = time.time()
        print('Time to make unitaries expm', end_time-start_time)
        tmp = self.compute_unitaries_eig(hamiltonians)
        for k, v in zip(unitaries, tmp):
            norm = tf.linalg.norm(k-v)
            print(norm)
            if (norm.numpy()/(self.hamiltonian_grid_size**2)) > 1e-9:
                assert False
        assert False
        return unitaries

    def compute_unitaries(self, hamiltonians):
        start_time = time.time()
        out = []
        const = tf.constant(-self.delta_t, dtype=_ftype)
        tf_zero = tf.constant(0, dtype=_ftype)

        for h in hamiltonians:
            vals, vecs = tf.linalg.eigh(h)
            Q = vecs
            R = tf.transpose(Q)
            L = tf.diag(vals)
            B = (Q@L)@R 
            #print(B)
            #print(hamiltonians[0])
            #print('diff\n', B-hamiltonians[0])
            vals_exp = tf.diag(tf.exp(tf.complex(real=tf_zero, imag=const * vals)))
            h0 = (tf.complex(real=Q, imag=tf_zero) @ vals_exp) @ tf.complex(real=R, imag=tf_zero)
            #print(h0.dtype)
            #print(unitaries[0].dtype)
            #print(h0-unitaries[0])

            # print('change to eigendecomposition and compare')
            out.append(h0)
        end_time = time.time()
        # print('Time to make unitaries eig', end_time-start_time})

        return out


def uniform_init(size, low=-1.0, high=1.0):   
    """ Protocol init - uniform random in -1, 1 """
    print('random init')
    start = np.random.uniform(size=(size, 1), low=low, high=high)
    start[-1] = 0.55
    return start


def gauss_init(size, mean=0.0, std=0.25):
    """ Protocol init - gaussian """
    print('normal random init')
    start = np.random.normal(size=(size, 1), loc=mean, scale=std)
    start[-1] = 0.55
    return start


def tf_loss(quant, protocol, amplitudes=None):   
    """ Computes 1-fidelity for given protocol """ 
    hamiltonians = quant.make_hamiltonians(protocol, amplitudes=amplitudes)
    unitaries = quant.compute_unitaries(hamiltonians)
    unitary_prod = compute_mat_prod(unitaries)
    fidel_vec = tf.transpose(tf.conj(quant.target_state)) @(unitary_prod @ quant.start_state)    
    fidel = tf.real(tf.norm(fidel_vec))**2
    return tf.constant(1.0, dtype=_ftype) - fidel


def tf_step(Q, optimizer, params, global_step):
    """ Take a gradient step with the given optimizer """
    low_grad = 0
    with tf.GradientTape() as tape:
        loss = tf_loss(Q, params)
        grads = tape.gradient(loss, params)
        norm_grad = tf.linalg.norm(grads).numpy()
        if norm_grad <= 1e-2:
            low_grad = low_grad + 1
        else: 
            low_grad = 0
        optimizer.apply_gradients(zip([grads], [params]),
                                  global_step=global_step)
    return loss, low_grad > 10, norm_grad


def run(Q_params, lr, rounds=1, start_protocol=None): 
    """ The gd main algorithm """   
    tf.enable_eager_execution()
    Q = GradQuantum(**Q_params)
    learning_rate = tf.Variable(lr, trainable=False)

    global_step = tf.Variable(0, trainable=False)
    # some issue with updating learning rate i havent figured out so we just make new adam optimizer every time
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    if start_protocol is None:
        params = tf.Variable(uniform_init(Q.N), dtype=_ftype)        
    else:        
        params = tf.Variable(start_protocol, dtype=_ftype)        
    hist = []
    start = params.numpy()
    best_loss = tf_loss(Q, params).numpy()
    print('starting loss', best_loss, 'fid', 1.0 - best_loss)
    best_prot = params.numpy()
    drop = 0    
    dec = 0
    for i in range(rounds):
        cur_params = params.numpy()
        loss, bad_grad, grad_norm = tf_step(Q, optimizer, params, global_step)
        if loss.numpy() < best_loss:
            best_prot = cur_params.copy()
            best_loss = loss.numpy()
            #print('--- updating best --- ', 1.0 - best_loss)
            print('round {0}: fidel {1}, grad_norm {2}, New Best'.format(i, 1.0-loss, grad_norm))
            drop = max(drop - 1, 0)
        else:
            drop = drop + 1
            print('round {0}: fidel {1}, grad_norm {2}'.format(i, 1.0-loss, grad_norm))

        tmp = params.numpy()
        tmp = np.clip(tmp, a_min=-1, a_max=1)
        tmp[-1] = -0.55
        params = tf.Variable(tmp, dtype=_ftype)
        # change to just using watch in gradient tape perhaps later        

        hist.append(loss.numpy())
        if best_loss <= 0.001:
            print('goal achieved')
            break
        if bad_grad:
            print('bad gradients - breaking')
            break
        if drop > 20:        
            print('not improving in 20 steps - change learning rate - should really add a lr scheduler...')
            if dec >6:
                print('happened before - breaking', dec)
                break
            learning_rate = learning_rate / 2.0
            print('fix learnign rate by factor 2 - reset params to best', learning_rate.numpy())
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            dec = dec+1
            drop = 0
            params = tf.Variable(best_prot, dtype=_ftype)
        
    hist = [1.0 - x for x in hist]
    end = params.numpy()
    print('norm diff start, end', np.linalg.norm(start-end))
    print('best fidelity:', np.max(hist), 1.0-best_loss)
    print('max min of best', np.min(best_prot), np.max(best_prot))
    print('loss of best_prot returned', 1.0 - tf_loss(Q, tf.constant(best_prot, dtype=_ftype)).numpy())
    return 1.0 - best_loss, best_prot, hist


def plot_protocol(protocol, t):
    """ Plot protocol that uses time t """
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    delta_t = t/len(protocol)
    xr = delta_t + np.array(list(range(len(protocol)))) * delta_t
    ax.step(xr,  protocol[::-1], '-', color='green', linewidth=2, markerfacecolor="None", markersize=6)
    ax.set_ylim([-1, 1])
    ax.set_xlabel('Time')
    ax.set_ylabel('Tweezer Positions')
    ax.set_title('Protocol Plot')
    return fig, ax


def main(args):
    params = {'T': args.T, 'N': args.N, 'hamiltonian_grid_size': args.hgrid,
              'Alpha': args.A, 'Beta': args.B, 
              'superposition': args.superposition}
    print('Args Used', args)

    start_protocol = None

    best_fid, best_prot, scores = run(params, rounds=args.rounds,
                                      start_protocol=start_protocol,
                                      lr=args.lr)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-run', dest='run', action='store_true', default=False)
    parser.add_argument('-run_super', dest='run_super', action='store_true',
                        default=False)    
    parser.add_argument('-hgrid', dest='hgrid', type=int, default=32, help='Size of grid used for hamiltonians')
    parser.add_argument('-T', dest='T', type=float, default=0.1, help='Time T to run experiment')
    parser.add_argument('-N', dest='N', type=int, default=40, help='Number of steps in protocol')
    parser.add_argument('-A', dest='A', type=int, default=160, help='Intentisity of Moving Tweezer')
    parser.add_argument('-B', dest='B', type=int, default=130, help='Intensity of trapping Tweezer - 130 is defined in nature paper')
    parser.add_argument('-rounds', dest='rounds', type=int, default=3, help='Number of descent iterations')
    parser.add_argument('-lr', dest='lr', type=float, default=0.01, help='Adam learning rate')

    parser.add_argument('-superposition', dest='superposition',
                        action='store_true', default=False, help='Set target state to superposition')

    args = parser.parse_args()
    main(args)


    
