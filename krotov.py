import tensorflow as tf 
import gd
import numpy as np


def compute_backward_step(Q, proj_state, hamilton_dict):
    """ Compute krotov backwards step """
    back_dict = {Q.N: proj_state}
    tconst = tf.constant(Q.delta_t, dtype=gd._ftype)
    for i in reversed(range(1, Q.N+1)):
        hi = hamilton_dict[i]
        mi = Q.compute_unitary(hi, tconst)
        state_i = mi @ back_dict[i]
        back_dict[i-1] = state_i
    return back_dict
    

def compute_gradient_hamiltonians(Q, positions, amplitudes):
    """ Compute Hamiltonian Derivative Matrices (diagonals)
        returns: (dict, dict) tf.tensor 
    """
    std2 = tf.constant(2.0 * Q.sigma ** 2, dtype=gd._ftype) # should be something    
    pos_grads = {}
    amp_grads = {}    
    for i in range(1, Q.N+1):
        pos = positions[i]
        amp = amplitudes[i]
        diff = Q.hamiltonian_grid - pos
        diff2 = diff **2
        gexp = -diff2/std2
        gs = tf.exp(gexp)
        out = amp * gs
        grad_pos = amp * gs * -(1.0/std2) * 2*diff * -1
        pos_grads[i] = grad_pos
        amp_grads[i] = gs
 
    return pos_grads, amp_grads


def krotov_step(Q, projected_endstate, tf_positions, tf_amplitudes, pos_optimizer, amp_optimizer):
    """ Construct a Krotov Step """
    tconst = tf.constant(-Q.delta_t, dtype=gd._ftype)
    for_dict = {0: Q.start_state} 
    pos_grads = {}
    amp_grads = {}
    new_hamilton_dict = {i+1: Q.make_hamilton(pos=tf_positions[i+1], amp=tf_amplitudes[i+1]) for i in range(Q.N)}
    unitaries = {k: Q.compute_unitary(h, tconst) for (k, h) in new_hamilton_dict.items()}
    J_positions, J_amplitudes = compute_gradient_hamiltonians(Q, tf_positions, tf_amplitudes)
    back_dict = compute_backward_step(Q, projected_endstate, new_hamilton_dict)
    tf_zero = tf.constant(0.0, dtype=gd._ftype)
    for i in range(0, Q.N):
        new_state = for_dict[i]
        back_state = back_dict[i]
        dpos = tf.reshape(tf.complex(real=J_positions[i+1], imag=tf_zero), (-1, 1))
        damp = tf.reshape(tf.complex(real=J_amplitudes[i+1], imag=tf_zero), (-1, 1))
        back_state_h = tf.transpose(tf.conj(back_state))                
        grad_pos = tf.reduce_sum(tf.imag(back_state_h @ (dpos * new_state)))
        grad_amp = tf.reduce_sum(tf.imag(back_state_h @ (damp * new_state)))
        pos_grads[i+1] = grad_pos 
        amp_grads[i+1] = grad_amp
        for_dict[i+1] = unitaries[i+1] @ new_state        
    new_fid = gd.fidelity(Q, for_dict[Q.N])
    # update controls
    assert len(pos_grads) == len(tf_positions)            
    pos_grads_and_vars = zip([pos_grads[i] for i in range(1, Q.N+1)], [tf_positions[i] for i in range(1, Q.N+1)])
    train_op_pos = pos_optimizer.apply_gradients(pos_grads_and_vars)
    amp_grads_and_vars = zip([amp_grads[i] for i in range(1, Q.N+1)], [tf_amplitudes[i] for i in range(1, Q.N+1)])
    train_op_amp = amp_optimizer.apply_gradients(amp_grads_and_vars)  
    len_proj = tf.transpose(tf.conj(Q.target_state)) @ for_dict[Q.N]
    projected_endstate = Q.target_state * len_proj
    return new_fid, train_op_pos, train_op_amp, projected_endstate


    

def input_to_dict(arr):
    d_arr = {i+1: val for (i, val) in enumerate(arr)}
    return d_arr

def krotov(params, start_positions, start_amplitudes, lr_pos, lr_amp, rounds, verbose=1):
    if verbose > 0:
        print('Krotov Session Called', params, 'rounds: ', rounds)
    Q = gd.GradQuantum(**params)
    #print('run krotov', lr_pos, lr_amp)
    assert len(start_positions) == Q.N
    assert len(start_positions) == len(start_amplitudes)    
    assert lr_pos > 0
    assert lr_amp > 0
    if type(start_positions) is np.ndarray:
        if verbose > 0:
            print('positions - array to dict'), 
        start_positions = input_to_dict(start_positions)
    if type(start_amplitudes) is np.ndarray:
        if verbose > 0:
            print('amplitudes - array to dict'), 
        start_amplitudes = input_to_dict(start_amplitudes)
    
    start_positions[1] = -0.55 
    tconst = tf.constant(-Q.delta_t, dtype=gd._ftype)

    tf_positions = {i: tf.get_variable('p'+str(i), dtype=gd._ftype, initializer=tf.constant(start_positions[i], dtype=gd._ftype), constraint=lambda x: tf.clip_by_value(x, -1, 1)) for i in range(2, Q.N+1)}
    tf_positions[1] = tf.get_variable('p1', dtype=gd._ftype, initializer=tf.constant(start_positions[1], dtype=gd._ftype), constraint=lambda x: tf.clip_by_value(x, -0.55, -0.55))
    tf_amplitudes = {i: tf.get_variable('a'+str(i), dtype=gd._ftype, initializer=tf.constant(start_amplitudes[i], dtype=gd._ftype), constraint=lambda x: tf.clip_by_value(x, 0.0, Q.Alpha)) for i in range(1, Q.N+1)}
    init_hamilton_dict = {i: Q.make_hamilton(pos=tf_positions[i], amp=tf_amplitudes[i]) for i in range(1, Q.N+1)}
    init_unitaries = {k: Q.compute_unitary(h, tconst) for (k, h) in init_hamilton_dict.items()}
    ### Initial computation of X_N
    forward_dict = {0: Q.start_state}
    # forward pass
    for i in range(1, Q.N+1):
        mi = init_unitaries[i]
        state_i = mi @ forward_dict[i-1]
        forward_dict[i] = state_i

    init_len_proj = tf.transpose(tf.conj(Q.target_state)) @ forward_dict[Q.N]
    init_fid = gd.fidelity(Q, forward_dict[Q.N])
    init_projected_endstate = Q.target_state * init_len_proj
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        projected_endstate_numpy, fid =  session.run([init_projected_endstate, init_fid])
    if verbose > 0:
        print('fid before start:', fid)
    # tf_parameters
    tf_lr_pos = tf.placeholder(dtype=tf.float32, shape=[]) 
    tf_lr_amp = tf.placeholder(dtype=tf.float32, shape=[]) 
    pos_optimizer = tf.train.AdamOptimizer(learning_rate=tf_lr_pos)
    amp_optimizer = tf.train.AdamOptimizer(learning_rate=tf_lr_amp)
    projected_endstate = tf.placeholder(dtype=tf.complex128, shape=(Q.hamiltonian_grid_size, 1))

    best_fid = 0
    best_positions = start_positions
    best_amplitudes = start_amplitudes
    tf_fid, tf_train_op_pos, tf_train_op_amp, tf_projected_endstate_new = krotov_step(Q, projected_endstate, tf_positions, tf_amplitudes, pos_optimizer, amp_optimizer)

    drop = 0
    drop_target = 100
    learning_rate_decreases = 0
    scores = []

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())    
        for r_idx in range(rounds):
            new_fid, new_pos, new_amp, _, _, projected_endstate_numpy  = session.run([tf_fid, tf_positions, tf_amplitudes, tf_train_op_pos, tf_train_op_amp, tf_projected_endstate_new], 
                                                                  feed_dict={projected_endstate: projected_endstate_numpy,
                                                                             tf_lr_pos: lr_pos, tf_lr_amp: lr_amp })
            scores.append(new_fid)
            if new_fid > best_fid:
                best_fid = new_fid
                best_positions = new_pos
                best_amplitudes = new_amp
                drop = max(drop -1, 0)
                if verbose > 0:
                    print('new best: fidelity of current controls,  round: {0}, fid: {1}'.format(r_idx, new_fid))
            else:
                drop = drop + 1
                if verbose > 0:
                    print('fidelity of current controls, round : {0}, fid: {1}'.format(r_idx, new_fid))
            
            if best_fid >= 0.999:
                break
            if (drop >= drop_target):
                if verbose:
                    print('not improving in {0} steps - change learning rate - should really add a lr scheduler...'.format(drop_target))
                if learning_rate_decreases > 4:
                    print('happened many times before - breaking', learning_rate_decreases  )
                    break
                lr_pos = lr_pos/2.0
                lr_amp = lr_amp/2.0
                if verbose > 0:
                    print('halve learning rate, reset parameters, reset ADAM', lr_pos, lr_amp)
                session.run({i: tf.assign(tf_positions[i], best_positions[i]) for i in range(1, Q.N+1)})
                session.run({i: tf.assign(tf_amplitudes[i], best_amplitudes[i]) for i in range(1, Q.N+1)})
                projected_endstate_numpy, fid =  session.run([init_projected_endstate, init_fid])
                learning_rate_decreases = learning_rate_decreases + 1
                drop = 0


    return best_fid, best_positions, best_amplitudes, scores

