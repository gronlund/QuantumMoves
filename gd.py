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

    def __init__(self, m=1.0, T=0.1, N=40, L=1.1, Alpha=160, Beta=130, sigma=1.0/8.0, hamiltonian_grid_size = 32, 
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
        assert T > 0
        assert N > 0
        assert hamiltonian_grid_size > 0

        self.m = m 
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
        self.hamiltonian_grid_size = hamiltonian_grid_size
        self.hamiltonian_grid = tf.lin_space(start=tf.constant(-1.0, dtype=_ftype), stop=1.0, num=self.hamiltonian_grid_size)
        self.hamiltonian_spacing = 1.0 /((self.hamiltonian_grid_size - 1.0) / 2.0)
        self.make_fixed_matrices()
        self.start_state = self.compute_initial_state()
        self.target_state = self.compute_target_state()
        assert self.target_state.shape == (self.hamiltonian_grid_size, 1), 'bad states'
    
    def make_fixed_matrices(self):
        grid_size = self.hamiltonian_grid_size
        ### T MATRIX ###
        np_tmat = np.diag(2.0 * np.ones(grid_size), k=0) + np.diag(-np.ones(grid_size-1), k=-1) + np.diag(-np.ones(grid_size-1), k=+1)
        t_mat = tf.convert_to_tensor(np_tmat, dtype=_ftype)
        scale = tf.constant(0.5 / (self.hamiltonian_spacing ** 2), dtype=_ftype)
        t_mat = scale * t_mat
        self.t_mat = t_mat
        ### B MATRIX ###
        std2 = tf.constant(2 * self.sigma ** 2, dtype=_ftype) # should be something
        diag_entries = -((self.hamiltonian_grid - self.x_start) ** 2) / std2
        exp_xi = tf.exp(diag_entries)
        b_mat = tf.diag(self.Beta * exp_xi)
        self.b_mat = b_mat

    def compute_initial_state(self):
        """ Compute the target state as the eigenvector of the smallest eigenvalue of H(something)                
        """
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


    def make_hamilton(self, pos, amp):
        """ Construct hamilton matrix from position and amplitude"""
        std2 = tf.constant(2.0 * self.sigma ** 2, dtype=_ftype) 
        grid_exponents  = (self.hamiltonian_grid - pos)**2
        grid_gaussians = tf.exp(- grid_exponents / std2)
        ak = tf.diag(amp * grid_gaussians)
        return self.t_mat - self.b_mat - ak


    def make_hamiltonians(self, positions, amplitudes=None):
        """ Create Hamiltonian Matrices 
            returns: list tf.tensor(), dtype _ftype
        """
        #print('B matrix diagonal:\n', b_mat.diagonal())
        ### A MATRIX 
        a_mat_list = list()
        std2 = tf.constant(2.0 * self.sigma ** 2, dtype=_ftype) # should be something
        if type(positions) is list:
            n_pos = len(positions)
        else:
            n_pos = positions.shape[0]

        if amplitudes is None:
            amplitudes = [tf.constant(self.Alpha, dtype=_ftype) for _ in range(n_pos)]
        for idx in range(n_pos):
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
        """ Made for time comparison - it is slower - DEPRECATED"""
        #print('Compute unitaries - shape', hamiltonians[0].shape)
        assert False
        start_time = time.time()
        const = tf.constant(-self.delta_t, dtype=_ftype)
        tf_zero = tf.constant(0, dtype=_ftype)
        unitaries = [tf.linalg.expm(tf.complex(real=tf_zero, imag=h*const)) for h in hamiltonians]
        end_time = time.time()
        print('Time to make unitaries expm', end_time-start_time)
        #tmp = self.compute_unitaries_eig(hamiltonians)
        #for k, v in zip(unitaries, tmp):
        #    norm = tf.linalg.norm(k-v)
        #    print(norm)
        #    if (norm.numpy()/(self.hamiltonian_grid_size**2)) > 1e-9:
        #        assert False
        #assert False
        return unitaries

    def compute_unitary(self, h, tconst):
        tf_zero = tf.constant(0, dtype=_ftype)
        vals, vecs = tf.linalg.eigh(h)
        Q = vecs
        R = tf.transpose(Q)
        L = tf.diag(vals)
        # B = (Q @ L) @ R 
        #print(B)
        #print(hamiltonians[0])
        #print('diff\n', B-hamiltonians[0])
        vals_exp = tf.diag(tf.exp(tf.complex(real=tf_zero, imag=tconst * vals)))
        U = (tf.complex(real=Q, imag=tf_zero) @ vals_exp) @ tf.complex(real=R, imag=tf_zero)
        return U

    def compute_unitaries(self, hamiltonians):
        start_time = time.time()
        out = []
        const = tf.constant(-self.delta_t, dtype=_ftype)
        tf_zero = tf.constant(0, dtype=_ftype)

        for h in hamiltonians:
            h0 = self.compute_unitary(h, const)
            #vals, vecs = tf.linalg.eigh(h)
            #Q = vecs
            #R = tf.transpose(Q)
            #L = tf.diag(vals)
            #B = (Q @ L) @ R 
            #print(B)
            #print(hamiltonians[0])
            #print('diff\n', B-hamiltonians[0])
            #vals_exp = tf.diag(tf.exp(tf.complex(real=tf_zero, imag=const * vals)))
            #h0 = (tf.complex(real=Q, imag=tf_zero) @ vals_exp) @ tf.complex(real=R, imag=tf_zero)
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
    start[-1] = -0.55
    return start
    # start = np.clip(np.random.normal(0, 0.25, size=(size,1)), -1.0, 1.0)


def gauss_init(size, mean=0.0, std=0.25):
    """ Protocol init - gaussian """
    print('random init')
    start = np.random.normal(size=(size, 1), loc=mean, scale=std)
    start[-1] = 0.55
    return start

def fidelity(quant, sn):
    fidel_vec = tf.transpose(tf.conj(quant.target_state)) @ sn
    fidel = tf.real(tf.norm(fidel_vec))**2
    return fidel


def tf_loss(quant, positions, amplitudes=None):   
    """ Computes 1-fidelity for given protocol """ 
    hamiltonians = quant.make_hamiltonians(positions, amplitudes=amplitudes)
    unitaries = quant.compute_unitaries(hamiltonians)
    unitary_prod = compute_mat_prod(unitaries)
    fidel_vec = tf.transpose(tf.conj(quant.target_state)) @(unitary_prod @ quant.start_state)    
    fidel = tf.real(tf.norm(fidel_vec))**2
    return tf.constant(1.0, dtype=_ftype) - fidel


def tf_run(Q_params, lr, rounds, start_positions, start_amplitudes, verbose=1, optimize_amps=True):
    """ Run Gradient Descent with adam using both position and ampltitude as paramters 
        Scale the amplitudes to -1, 1 by divide by 80 and subttract one
    """       
    # print('running positiona adn amplitude gd - swapping between them')
    Q = GradQuantum(**Q_params)
    assert len(start_positions) == Q.N
    assert len(start_positions) == len(start_amplitudes)    
    print('starting tf_run T: {0}, N: {1}'.format(Q.T, Q.N))
    hist = []
    best_prot = start_positions
    best_amp = start_amplitudes
    best_loss = 1    
    dec = 0
    drop = 0
    drop_target = 40


    np_pos_range_min = np.full(shape=(Q.N, 1), fill_value=-1.0)
    np_pos_range_min[-1] = -0.55
    np_pos_range_max = np.full(shape=(Q.N, 1), fill_value=1.0)
    np_pos_range_max[-1] = -0.55
    tf_pos_min = tf.constant(np_pos_range_min, dtype=_ftype)
    tf_pos_max = tf.constant(np_pos_range_max, dtype=_ftype)


    tf_positions = tf.get_variable('positions', dtype=_ftype, initializer=tf.constant(start_positions.reshape(-1, 1)), constraint=lambda x: tf.clip_by_value(x, tf_pos_min, tf_pos_max))
    tf_amplitudes = tf.get_variable('amplitudes', dtype=_ftype, initializer=tf.constant(start_amplitudes.reshape(-1, 1)), constraint=lambda x: tf.clip_by_value(x, 0.0, Q.Alpha), trainable=optimize_amps)

    tf_err = tf_loss(Q, tf_positions, tf_amplitudes)
    tf_learning_rate = tf.placeholder(tf.float32, shape=[])
    tf_optimizer = tf.train.AdamOptimizer(learning_rate=tf_learning_rate)
    tf_gradients_and_vars = tf_optimizer.compute_gradients(loss=tf_err)
    tf_grad, _ = list(zip(*tf_gradients_and_vars))
    tf_norm = tf.global_norm(tf_grad)

    # scale the gradients here
    tf_train_op = tf_optimizer.apply_gradients(tf_gradients_and_vars)

    with tf.Session() as session:       
        session.run(tf.global_variables_initializer())
        for i in range(rounds):
            _, new_loss, new_grads, new_pos, new_amp, new_norm = session.run([tf_train_op, tf_err, tf_gradients_and_vars, tf_positions, tf_amplitudes, tf_norm], feed_dict={tf_learning_rate: lr})              
            if new_loss < best_loss:
                best_prot = new_pos
                best_amp = new_amp
                best_loss = new_loss
                drop = max(drop -1, 0)
                if verbose:
                    print('round {0}: fidel {1}, grad_norm {2}, New Best'.format(i, 1.0 - new_loss, new_norm))
            else:
                drop = min(drop+1, drop_target+1)
                if verbose:
                    print('round {0}: fidel {1}, grad_norm {2}'.format(i, 1.0 - new_loss, new_norm))


            hist.append(new_loss)       
            if best_loss <= 0.001:
                break
            if i < 100:
                continue            
            if new_norm <= 1e-5:
                print('No gradient left - lets quit')
                break
            if (drop >= drop_target):
                if verbose:
                    print('not improving in {0} steps - change learning rate - should really add a lr scheduler...'.format(drop_target))
                if dec > 8:
                    print('happened many times before - breaking', dec)
                    break
                lr = lr/2.0
                print('halve learning rate, reset parameters, reset ADAM', lr)
                #best_prot[-1] = -0.55
                session.run(tf_positions.assign(best_prot))
                if optimize_amps:
                    session.run(tf_amplitudes.assign(best_amp))
                dec = dec+1
                drop = 0
        

    print('norm diff start, end', np.linalg.norm(start_positions-best_prot))
    print('best fidelity:', 1.0-best_loss)
    print('min max of best', np.min(best_prot), np.max(best_prot), best_amp.min(), best_amp.max())
    tf.reset_default_graph()
    return 1.0 - best_loss, best_prot, best_amp, [1.0 - x for x in hist]



