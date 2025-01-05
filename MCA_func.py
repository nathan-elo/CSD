import numpy as np
import math

def build_node_order_and_t(o_dict, r_dict, s_dict, d_dict): 
    """
    Build the global cost matrix `t` and mapping `node_to_index` from node labels
    to indices. The function also returns the separate lists of d_nodes, o_nodes,
    r_nodes, and s_nodes in the order they appear in the cost matrix. (DORS)

    Parameters
    ----------
    o_dict : dict
        Mapping from each origin node (str) to a dict of r and d nodes, with cost.
        Example: {'0': {'1': 2.0, '2': 5.0}, ...}
    r_dict : dict
        Mapping from each r-node to a dict of s-nodes with cost.
    s_dict : dict
        Mapping from each s-node to a dict of r and d nodes, with cost.
    d_dict : dict
        Mapping from each d-node to nothing.

    Returns
    -------
    t : narray of shape (total_nodes, total_nodes)
        Global cost matrix. (d,o,r,s)
    node_to_index : dict
        Dictionary mapping each node label to its index in `t`.
    d_nodes : list
        List of destination node labels.
    o_nodes : list
        List of origin node labels.
    r_nodes : list
        List of r-layer node labels.
    s_nodes : list
        List of s-layer node labels.
    """
    d_nodes = list(d_dict.keys())
    o_nodes = list(o_dict.keys())
    r_nodes = list(r_dict.keys())
    s_nodes = list(s_dict.keys())

    node_order = d_nodes + o_nodes + r_nodes + s_nodes
    node_to_index = {node: i for i, node in enumerate(node_order)}

    t_size = len(node_order)
    t = np.zeros((t_size, t_size))

    for o_node, links in o_dict.items():
        o_i = node_to_index[o_node]
        for dest_node, value in links.items():
            t[o_i, node_to_index[dest_node]] = value

    for r_node, links in r_dict.items():
        r_i = node_to_index[r_node]
        for s_node_, value in links.items():
            t[r_i, node_to_index[s_node_]] = value

    for s_node_, links in s_dict.items():
        s_i = node_to_index[s_node_]
        for dest_node, value in links.items():
            t[s_i, node_to_index[dest_node]] = value

    return t, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes

def initalize_backwards_arrays(n_size, r_size, s_size, d_size):
    """
    Initialize arrays used in the backwards array, starting from d.

    Parameters
    ----------
    n_size : int
        Number of layers/iterations. (possible deliveries performed by a single driver in one go)
        Be careful as n_size = 2 represent a single column. n_size=3 represents 3 columns and so on...
    r_size : int
        Number of r-nodes.
    s_size : int
        Number of s-nodes.
    d_size : int
        Number of d-nodes.

    Returns
    -------
    Z_sd : np.array, shape (iterations, s_size, d_size)
        Z-values for s->d transitions, when there are n possible deliveries left to perform.
    mu_sd : np.array, shape (iterations, s_size, d_size)
        Minimum expected costs from s to d, when there are n possible deliveries left to perform.
    P_sd_rs : np.array, shape (iterations, r_size, s_size, s_size, d_size)
        Probability of using each r->s path (at n) when going from s->d.
    d_mu_sd : np.array, shape (iterations, r_size, s_size, s_size, d_size)
        Derivative of mu_sd with respect to v_rs, at n.

    Notes
    -----
    n_size = 1 isn't supported this would mean 0 columns...
    """
    iterations = n_size
    Z_sd = np.zeros((iterations, s_size, d_size))
    mu_sd = np.zeros((iterations, s_size, d_size))
    P_sd_rs = np.zeros((iterations, r_size, s_size, s_size, d_size)) 
    d_mu_sd = np.zeros((iterations, r_size, s_size, s_size, d_size))
    return Z_sd, mu_sd, P_sd_rs, d_mu_sd

def initialize_mu_sd_first_iteration(mu_sd, t, d_size, s_nodes, node_to_index):
    """
    Initialize mu_sd for the first iteration using t_sd

    Parameters
    ----------
    mu_sd : np.array, shape (iterations, s_size, d_size)
         Minimum expected costs from s to d, when there are n possible deliveries left to perform.
    t : np.array, shape (total_nodes, total_nodes)
        Global cost matrix. (d,o,r,s)
    d_size : int
        Number of d-nodes (destinations).
    s_nodes : list of str
        Labels of the s-nodes.
    node_to_index : dict
        Mapping from node label to global index in t.

    Returns
    -------
    None
        The function updates mu_sd in place for the first iteration.
    """
    for s_idx, s_node in enumerate(s_nodes):
        mu_sd[0, s_idx, :] = t[node_to_index[s_node], :d_size]

def compute_mu_sd_and_P_sd(Z_sd, mu_sd, P_sd_rs, t, node_to_index, d_nodes, r_nodes, s_nodes, theta_p, v):

    """
    Compute mu_sd  and P_sd_rs  for each iteration n>0,
    based on the previous iteration's mu_sd.

    Parameters
    ----------
    Z_sd : np.array, shape (iterations, s_size, d_size)
        Z-values for s->d transitions, when there are n possible deliveries left to perform.
    mu_sd : np.array, shape (iterations, s_size, d_size)
        Minimum expected costs from s to d, when there are n possible deliveries left to perform.
    P_sd_rs : np.array, shape (iterations, r_size, s_size, s_size, d_size)
        Probability of using each r->s path (at n) when going from s->d.
    t : np.array, shape (total_nodes, total_nodes)
        Global cost matrix. (d,o,r,s)
    node_to_index : dict
        Maps node labels to global indices in t.
    d_nodes : list
        Destination node labels.
    r_nodes : list
        R-node labels.
    s_nodes : list
        S-node labels.
    theta_p : float
        logit parameter
    v : np.array, shape (r_size, s_size)
        cost of delivery rs

    Returns
    -------
    """

    iterations = Z_sd.shape[0]
    d_size = len(d_nodes)
    exp = np.exp

    
    r_indices = [node_to_index[r_node] for r_node in r_nodes]  #list of indexes in t matrix for r nodes : [3,4,5,6]
    s_indices = [node_to_index[s_node2] for s_node2 in s_nodes] #list of indexes in t matrix for s nodes

    for n in range(1, iterations):
        mu_sd_prev = mu_sd[n - 1]
        mu_sd_curr = mu_sd[n]
        Z_sd_curr = Z_sd[n]
        P_sd_rs_curr = P_sd_rs[n] # allows for simpler code. "real" value gets updated when current value is updated.

        for s_idx, s_node in enumerate(s_nodes):
            s_index = node_to_index[s_node] #get the t_index for the current s node (s node in sd, not rs)

            
            t_s_r = t[s_index, r_indices]  # get the row for the current s, and all r. gives t_s to all r. size r_nodes
            t_r_s2 = t[np.ix_(r_indices, s_indices)]  # get the t from r to s submatrix. size r_nodes*s_nodes
            mask = (t_r_s2 > 0)  # boolean array, size  r*s. true only when there is a connection.

            
            exp_direct_s_d = exp(-theta_p * t[s_index, :d_size]) #direct cost from current s to all destinations d.

            
            for d_idx in range(d_size):
                #costs for partial path from current through rs delivery to d.
                # Broadcasting:
                # t_s_r: (r_size,) -> (r_size,1)
                # t_r_s2: (r_size,s_size)
                # v: (r_size,s_size)
                # mu_sd_prev: (s_size,d_size) -> (1,s_size)
                val_mat = exp(-theta_p * (t_s_r[:, None] #None added for broadcasting, so that the size aligns with t_r_s2.
                                          + t_r_s2
                                          - v 
                                          + mu_sd_prev[np.newaxis, :, d_idx])) #adds an axis for broadcasting 

                sum_tot = (val_mat[mask]).sum() #only sums when r is connected to s (when there is a delivery from r)
                Z_val = exp_direct_s_d[d_idx] + sum_tot
                Z_sd_curr[s_idx, d_idx] = Z_val
                mu_sd_curr[s_idx, d_idx] = (-1/theta_p)*math.log(Z_val)
                inv_Z = 1.0/Z_val

                # P_sd_rs_curr dimension: (r_size, s_size, s_size, d_size)
                # for the current s_d pair, compute the probabilities for every rs delivery. Only adds a value when rs is possible.
                P_sd_rs_curr[:, :, s_idx, d_idx] = (val_mat * mask) * inv_Z


def compute_d_mu_sd(d_mu_sd, P_sd_rs,s_nodes, d_nodes):
    """
    Computs d_mu_sd, the derivative of mu_sd, for every iterations, every s and d nodes, relative to rs deliveries.

    Parameters
    ----------
    d_mu_sd : ndarray, shape (iterations, r_size, s_size, s_size, d_size)
        The derivative of mu_sd with respect to rs deliveries.
    P_sd_rs : np.array, shape (iterations, r_size, s_size, s_size, d_size)
        Probability of using each r->s path (at n) when going from s->d.
    s_nodes : list
        Labels for s-nodes
    d_nodes : list
        Labels for d-nodes 

    Returns
    -------

    """

    iterations = d_mu_sd.shape[0]
    s_size = len(s_nodes)
    d_size = len(d_nodes)

    # starting from n=1 as n=0 is already initialized.
    for n in range(1, iterations):
        P_sd_rs_n = P_sd_rs[n]           # (r_size, s_size, s_size, d_size) #rs delivery, sd start and end node.
        d_mu_sd_prev = d_mu_sd[n - 1]    # (r_size, s_size, s_size, d_size)
        d_mu_sd_curr = d_mu_sd[n]        # (r_size, s_size, s_size, d_size)

        #looped over s and d. rs deliveries are vectorized for speed.
        for s__inside_idx in range(s_size): #inside index meaning index inside the s nodes only. First s node is 0. Different from t indexes.
            for d_inside_idx in range(d_size): 
                
                #P for every delivery, for the sd path.
                P_slice = P_sd_rs_n[:, :, s__inside_idx, d_inside_idx]  # shape: (r_size, s_size)

                # Sum of P_r to all s, for sd path.
                P_sum_r = P_slice.sum(axis=0)  # shape: (s_size,)

                # first term is the sum of P_sum_r over s, so the sum of Probablities for every rs delivery. The main sd path is fixed.
                #second term is the derivative, for every rs too, but d is fixed. The first s node is not fixed, as it's now the s from rs delivery.
                Multiplied_sum = np.einsum(
                'x, rsx -> rs', 
                P_sum_r, 
                d_mu_sd_prev[:, :, :, d_inside_idx]
                ) #size (r_size, s_size)

                # calcluations of d_mu for every rs delivery, sd being fixed.
                d_mu_sd_curr[:, :, s__inside_idx, d_inside_idx] = -P_slice + Multiplied_sum 


def compute_value_functions(rng, r_size, s_size, K, c_rs, v, theta_c):
    """
    Compute the perceived costs C_rs for every delivery. Then the total costs.

    Parameters
    ----------
    rng : np.random.Generator
        Random generator for reproducibility.
    r_size : int
        Number of r-nodes.
    s_size : int
        Number of s-nodes.
    K : int
        2 possibilities.
    c_rs : ndarray, shape (r_size, s_size, K)
        Base cost from r to s for K=0 (no outsourcing), and K=1 (outsourcing)
    v : np.array, shape (r_size, s_size)
        cost of rs delivery
    theta_c : float
        distribution parameter for Gumbel distribution.

    Returns
    -------
    C_rs : np.array, shape (r_size, s_size, K)
        Perceived costs
    sum_exp : np.array, shape (r_size, s_size)
        Sum of exponentials over the two choices
    bigV_rs : np.array, shape (r_size, s_size)
        Log-sum of sum_exp
    """

    epsilon = rng.gumbel(0, theta_c, size=(r_size,s_size,K))*0.01 # random parameter added to c_rs. different for every delivery rs. Follows theta_c.
    C_rs = c_rs + epsilon
    sum_exp = np.zeros((r_size,s_size))

    sum_0 = np.exp(-theta_c * C_rs[:, :, 0]) # total delivery cost when not outsourcing, for every rs delivery.
    sum_1 = np.exp(-theta_c * (C_rs[:, :, 1] + v)) # total delivery cost when outsourcing, for every rs delivery.
     
    sum_exp = sum_0 + sum_1 #total cost for shippers
    
    bigV_rs = (-1 / theta_c) * np.log(sum_exp) 
    
    return C_rs, sum_exp, bigV_rs

def compute_z_rs_1(z_rs_bar, C_rs, sum_exp, v, theta_c):
    """
    Compute the distribution z_rs of outsourcing shippers (choice=1).

    Parameters
    ----------
    z_rs_bar : np.array, shape (r_size, s_size)
        Base distribution or total number of shippers for every rs.
    C_rs : ndarray, shape (r_size, s_size, K)
        Perceived costs from r to s for K=0 (no outsourcing), and K=1 (outsourcing)
    sum_exp : np.array, shape (r_size, s_size)
        Sum of exponentials over the two choices
    v : np.array, shape (r_size, s_size)
        cost of rs delivery
    theta_c : float
        distribution parameter for Gumbel distribution.

    Returns
    -------
    np.array, shape (r_size, s_size)
        Number of outsourcing shippers per r-s.
    """
    return z_rs_bar * (np.exp(-theta_c*(C_rs[:,:,1]+v))) / sum_exp #distribution of outsourcing shippers

def compute_z_rs_0(z_rs_bar, C_rs, sum_exp,theta_c):
    """
    Compute the distribution z_rs of outsourcing shippers (choice=0).

    Parameters
    ----------
    z_rs_bar : np.array, shape (r_size, s_size)
        Base distribution or total number of shippers for every rs.
    C_rs : ndarray, shape (r_size, s_size, K)
        Perceived costs from r to s for K=0 (no outsourcing), and K=1 (outsourcing)
    sum_exp : np.array, shape (r_size, s_size)
        Sum of exponentials over the two choices
    v : np.array, shape (r_size, s_size)
        cost of rs delivery
    theta_c : float
        distribution parameter for Gumbel distribution.

    Returns
    -------
    np.array, shape (r_size, s_size)
        Number of non-outsourcing shippers per r-s.
    """
    return z_rs_bar * (np.exp(-theta_c*C_rs[:,:,0])) / sum_exp #distribution of non-outsourcing shippers

def initialize_final_arrays(o_size, d_size, r_size, s_size):
    """
    Initialize arrays for final iteration from o to d.

    Parameters
    ----------
    o_size : int
        Number of origin nodes.
    d_size : int
        Number of destination nodes.
    r_size : int
        Number of r-nodes.
    s_size : int
        Number of s-nodes.

    Returns
    -------
    Z_od : np.array, shape (o_size, d_size)
        exp value of all possible paths from o to d
    mu_od : np.array, shape (o_size, d_size)
        adjusted value of Z
    P_od_rs : np.array, shape (r_size, s_size, o_size, d_size)
        Probability of going from O->D through each (r,s).
    d_mu_od : np.array, shape (r_size, s_size, o_size, d_size)
        Derivative of mu_od with respect to vrs
    """
    Z_od = np.zeros((o_size, d_size))
    mu_od = np.zeros((o_size, d_size))
    P_od_rs = np.zeros((r_size, s_size, o_size, d_size))
    d_mu_od = np.zeros((r_size, s_size, o_size, d_size))
    return Z_od, mu_od, P_od_rs, d_mu_od

def compute_mu_od_P_od(Z_od, mu_od, P_od_rs, t, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes, mu_sd, n_size, theta_p, v):
    """
    Compute mu_od and P_od_rs 
    based on the final mu_sd of the backward portion.

    Parameters
    ----------
    Z_od : np.array, shape (o_size, d_size)
    mu_od : np.array shape (o_size, d_size)
    P_od_rs : np.array, shape (r_size, s_size, o_size, d_size)
    t : np.array, shape (total_nodes, total_nodes)
        Global cost matrix. (dors)
    node_to_index : dict
        Mapping from node labels to indices.
    d_nodes : list
        Destination node labels.
    o_nodes : list
        Origin node labels.
    r_nodes : list
        R-node labels.
    s_nodes : list
        S-node labels.
    mu_sd : np.array, shape (iterations, s_size, d_size)
    n_size : int
        number of possible deliveries carried in one go by a driver.
    theta_p : float
        logit parameter
    v : np.array, shape (r_size, s_size)
        costs of rs deliveries

    Returns
    -------
    """
    d_size = len(d_nodes)
    exp = np.exp

    r_indices = [node_to_index[r_node] for r_node in r_nodes] #for faster computing time
    s_indices = [node_to_index[s_node] for s_node in s_nodes]

    mu_sd_last = mu_sd[n_size - 2]  # shape: (s_size, d_size)      
    
    t_r_s = t[np.ix_(r_indices, s_indices)]  # # get the t from r to s submatrix. size r_nodes*s_nodes

   
    mask = (t_r_s > 0) # boolean array, size  r*s. true only when there is a connection.

    for o_idx, o_node in enumerate(o_nodes):
        o_index = node_to_index[o_node]

        # Precompute t_o_r for this o_node (cost from o to each r)
        t_o_r = t[o_index, r_indices]  # shape: (r_size,)

        for d_idx in range(d_size):
            #costs for partial path from current through rs delivery to d.
            # Broadcasting:
            # t_o_r: (r_size,) -> (r_size,1)
            # t_r_s: (r_size,s_size)
            # v: (r_size,s_size)
            # mu_sd_last: (s_size,d_size) -> (1,s_size)
            val_mat = exp(-theta_p*(t_o_r[:, None] + t_r_s - v + mu_sd_last[np.newaxis, :, d_idx]))

            #only sums when r is connected to s (when there is a delivery from r) i.e. mask==true
            sum_tot = val_mat[mask].sum()

            Z_val = exp(-theta_p*t[o_index, d_idx]) + sum_tot
            Z_od[o_idx, d_idx] = Z_val
            mu_od[o_idx, d_idx] = (-1/theta_p)*np.log(Z_val)
            inv_Z = 1.0/Z_val

            # P_od_rs_curr dimension: (r_size, s_size, o_size, d_size)
            # for the current s_d pair, compute the probabilities for every rs delivery. Only adds a value when rs is possible.
            P_od_rs[:, :, o_idx, d_idx] = (val_mat * mask) * inv_Z

def compute_d_mu_od(d_mu_od, P_od_rs, d_mu_sd, n_size,):
    """
    Compute the derivative d_mu_od based on P_od_rs and the last iteration of d_mu_sd.

    Parameters
    ----------
    d_mu_od : np.array, shape (r_size, s_size, o_size, d_size)
        Derivative of mu_od with respect to v_rs delivery prices.
    P_od_rs : np.array, shape (r_size, s_size, o_size, d_size)
        Probabilities of going from O->D via each (r, s).
    d_mu_sd : np.array, shape (iterations, r_size, s_size, s_size, d_size)
        The derivative for the s->d portion.
    n_size : int

    Returns
    -------
    """
    d_mu_sd_last = d_mu_sd[n_size - 2]  # shape: (r_size, s_size, s_size, d_size)

    P_sum_r_in = P_od_rs.sum(axis=0)  # (s_size, o_size, d_size)

    #using einsum for efficient calculation. r s a d is the size of d_mu. a o d is the size of P_sum.
    # because we want to sum over the second s (a here), we remove it from the last expression. This can be seen as a dot product index.
    #r s and d are carried along.
    # o is a new dimension.
    #final dimension : r s o d.

    sum = np.einsum('r s a d, a o d -> r s o d', d_mu_sd_last, P_sum_r_in)

    d_mu_od[:] = -P_od_rs + sum


def compute_MC(mu_od, y_od_bar, bigV_rs, z_rs_bar):
    """
    Compute the objective function MC

    Parameters
    ----------
    mu_od : np.array, shape (o_size, d_size)
        Minimum expected costs for O->D.
    y_od_bar : np.array, shape (o_size, d_size)
        Demand for each O->D pair.
    bigV_rs : np.array, shape (r_size, s_size)
        The log-sum for the r->s portion.
    z_rs_bar : np.array, shape (r_size, s_size)
        The total number of shippers assigned to each rs delivery spot.

    Returns
    -------
    MC : float
    """
    return np.sum(y_od_bar*mu_od)+np.sum(z_rs_bar*bigV_rs)

def compute_grad_MC(grad_MC, y_od_bar, d_mu_od, z_rs_1):
    """
    Compute the gradient of MC with respect to v_rs.

    Parameters
    ----------
    grad_MC : np.array, shape (r_size, s_size)
        Gradient
    y_od_bar : np.array, shape (o_size, d_size)
        Demand for each O->D pair.
    d_mu_od : np.array, shape (r_size, s_size, o_size, d_size)
        Derivative of mu_od
    z_rs_1 : np.array, shape (r_size, s_size)
        Distribution of outsourcing shippers.

    Returns
    -------
    """
    sum_dmu = np.sum(y_od_bar*d_mu_od, axis=(2,3)) # sum over o,d
    grad_MC[:,:] = sum_dmu + z_rs_1

def compute_MC_and_grad_MC(o_dict, r_dict, s_dict, d_dict, n_size, K, theta_p, theta_c, z_rs_bar, y_od_bar, c_rs, seed, v):
    """
    Computes MC and grad_MC from a network, drivers and shippers informations, parameters and costs.

    Parameters
    ----------
    o_dict, r_dict, s_dict, d_dict : dict
        Dictionaries describing the cost links among O, R, S, D layers.
    n_size : int
    K : int     
    theta_p : float
        logit parameter
    theta_c : float
        gumbel distribution parameter
    z_rs_bar : np.array, shape (r_size, s_size)
        Base distribution or total number of shippers in r-s.
    y_od_bar : np.array, shape (o_size, d_size)
        Demand for each O->D pair.
    c_rs : np.array, shape (r_size, s_size, K)
        Exact costs for each r-s
    seed : int
        Random seed for reproducibility.
    v : np.array, shape (r_size, s_size)
        delivery costs/reward

    Returns
    -------
    MC : float
    grad_MC : ndarray, shape (r_size, s_size)
    """
    rng = np.random.default_rng(seed)
    t,node_to_index, d_nodes, o_nodes, r_nodes, s_nodes = build_node_order_and_t(o_dict, r_dict, s_dict, d_dict)
    r_size = len(r_nodes)
    s_size = len(s_nodes)
    d_size = len(d_nodes)
    o_size = len(o_nodes)

    Z_sd, mu_sd, P_sd_rs, d_mu_sd = initalize_backwards_arrays(n_size, r_size, s_size, d_size)
    initialize_mu_sd_first_iteration(mu_sd, t, d_size, s_nodes, node_to_index)
    compute_mu_sd_and_P_sd(Z_sd, mu_sd, P_sd_rs, t, node_to_index, d_nodes, r_nodes, s_nodes, theta_p, v)
    compute_d_mu_sd(d_mu_sd, P_sd_rs,s_nodes, d_nodes)
    C_rs, sum_exp, bigV_rs = compute_value_functions(rng, r_size, s_size, K, c_rs, v, theta_c)
    z_rs_1_val = compute_z_rs_1(z_rs_bar, C_rs, sum_exp, v, theta_c)
    Z_od, mu_od, P_od_rs,d_mu_od = initialize_final_arrays(o_size, d_size, r_size, s_size)
    compute_mu_od_P_od(Z_od, mu_od, P_od_rs, t, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes, mu_sd, n_size, theta_p, v)
    compute_d_mu_od(d_mu_od, P_od_rs, d_mu_sd, n_size)

    MC = compute_MC(mu_od, y_od_bar, bigV_rs, z_rs_bar)
    grad_MC = np.zeros((r_size, s_size))
    compute_grad_MC(grad_MC, y_od_bar, d_mu_od, z_rs_1_val)
    return MC, grad_MC

def compute_MC_and_grad_MC_return_intermediate(o_dict, r_dict, s_dict, d_dict, n_size, K, theta_p, theta_c, z_rs_bar, y_od_bar, c_rs, seed, v):
    """
    Same as compute_MC_and_grad_MC but returns additional intermediate arrays for analysis.

    Parameters
    ----------
    o_dict, r_dict, s_dict, d_dict : dict
        Dictionaries describing the cost links among O, R, S, D layers.
    n_size : int
    K : int     
    theta_p : float
        logit parameter
    theta_c : float
        gumbel distribution parameter
    z_rs_bar : np.array, shape (r_size, s_size)
        Base distribution or total number of shippers in r-s.
    y_od_bar : np.array, shape (o_size, d_size)
        Demand for each O->D pair.
    c_rs : np.array, shape (r_size, s_size, K)
        Exact costs for each r-s
    seed : int
        Random seed for reproducibility.
    v : np.array, shape (r_size, s_size)
        delivery costs/reward

    Returns
    -------
    MC, grad_MC : 

    C_rs, sum_exp, theta_c, P_od_rs, P_sd_rs, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes :
    """
    rng = np.random.default_rng(seed) 
    t, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes = build_node_order_and_t(o_dict, r_dict, s_dict, d_dict)
    r_size = len(r_nodes)
    s_size = len(s_nodes)
    d_size = len(d_nodes)
    o_size = len(o_nodes)

    Z_sd, mu_sd, P_sd_rs, d_mu_sd = initalize_backwards_arrays(n_size, r_size, s_size, d_size)
    initialize_mu_sd_first_iteration(mu_sd, t, d_size, s_nodes, node_to_index)
    compute_mu_sd_and_P_sd(Z_sd, mu_sd, P_sd_rs, t, node_to_index, d_nodes, r_nodes, s_nodes, theta_p, v)
    compute_d_mu_sd(d_mu_sd, P_sd_rs,s_nodes, d_nodes)
    C_rs, sum_exp, bigV_rs = compute_value_functions(rng, r_size, s_size, K, c_rs, v, theta_c)
    z_rs_1_val = compute_z_rs_1(z_rs_bar, C_rs, sum_exp, v, theta_c)
    Z_od, mu_od, P_od_rs,d_mu_od = initialize_final_arrays(o_size, d_size, r_size, s_size)
    compute_mu_od_P_od(Z_od, mu_od, P_od_rs, t, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes, mu_sd, n_size, theta_p, v)
    compute_d_mu_od(d_mu_od, P_od_rs, d_mu_sd, n_size)

    MC = compute_MC(mu_od, y_od_bar, bigV_rs, z_rs_bar)
    grad_MC = np.zeros((r_size, s_size))
    compute_grad_MC(grad_MC, y_od_bar, d_mu_od, z_rs_1_val)
    return MC, grad_MC, C_rs, sum_exp, theta_c, P_od_rs, P_sd_rs, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes

def compute_y_distrib(r_size,n_size,s_size, o_size, d_size,y_od_bar,P_od_rs,P_sd_rs):
    """
    Propagate flow from O->D through r->s (P_od_rs) and then within s->d (P_sd_rs) 
    backwards through n (so forward in the network), returning the distribution of flows.

    Parameters
    ----------
    r_size : int
        Number of r-nodes.
    n_size : int
    s_size : int
        Number of s-nodes.
    o_size : int
        Number of origin nodes.
    d_size : int
        Number of destination nodes.
    y_od_bar : np.array, shape (o_size, d_size)
        Demand from each O to each D.
    P_od_rs : np.array, shape (r_size, s_size, o_size, d_size)
        Probabilities of selecting each (r, s) for O->D.
    P_sd_rs : np.array, shape (n_size, r_size, s_size, s_size, d_size)
        Probabilities of selecting each (r, s) for S->D.

    Returns
    -------
    y_od_rs : np.array, shape (r_size, s_size, o_size, d_size)
        Total flow in each (r, s), for each O->D.
    y_rs : np.array, shape (r_size, s_size)
        Total flow in each (r, s) across all O->D.
    """

    y_flow_od = np.zeros((n_size, o_size, d_size)) #flow of drivers between od pairs
    y_flow_sd = np.zeros((n_size, o_size, s_size, d_size))  #flow of drivers from o, through s d.
    y_flow_od[n_size - 1, :, :] = y_od_bar #initial flow

    delivery_count_od = np.zeros((n_size, r_size, s_size, o_size, d_size)) # for drivers of od pair OD, number of drivers doing delivery rs at stage n.

   
    # Loop backward in n:
    for n in range(n_size - 1, -1, -1):
        #for simplicity, only take the actual step flow
        flow_od = y_flow_od[n]  # shape (o,d)
        # We create flow_delivery_all from P_od_rs (r,s,o,d) * flow_od(o,d)
        flow_delivery_all = P_od_rs * flow_od[np.newaxis, np.newaxis, :, :]  #broadcasting over rs. gives the flow at each delivery
        # Apply threshold:
        mask = flow_delivery_all > 1e-12
        # Add to delivery_count_od:
        delivery_count_od[n][mask] += flow_delivery_all[mask]

        if n > 0:
            # sum over r to distribute to y_flow_sd:
            sum_over_r = (flow_delivery_all * mask).sum(axis=0)  # shape (s,o,d)
            # transpose to (o,s,d)
            sum_over_r = sum_over_r.transpose(1,0,2)
            # Add to y_flow_sd[n-1]:
            y_flow_sd[n - 1] += sum_over_r

            flow_sd_n = y_flow_sd[n]  # (o,s,d)
            # For each (o,s,d),  multiply by P_sd_rs[n, :, :, s, d] and distribute
             
            for s_idx in range(s_size):
                flow_sd_slice = flow_sd_n[:, s_idx, :]  # (o,d)
                # Check threshold:
                mask_sd = flow_sd_slice > 1e-12
                if not np.any(mask_sd):
                    continue

                # P_sd_rs slice: P_sd_rs[n, :, :, s_idx, d_idx] gives (r,s)
                # shape of P_sd_rs_s: (r,s_size,d_size) after fixing n and s_idx.
                P_sd_rs_s = P_sd_rs[n, :, :, s_idx, :]  # (r,s,d)
                # Broadcast to (r,s,o,d):
                flow_delivery_all_sd = P_sd_rs_s[:, :, np.newaxis, :] * flow_sd_slice[np.newaxis, np.newaxis, :, :]

                # Apply threshold:
                mask_sd_full = (flow_delivery_all_sd > 1e-12)
                # Add to delivery_count_od:
                delivery_count_od[n][mask_sd_full.transpose(0,1,2,3)] += flow_delivery_all_sd[mask_sd_full]

                # Propagate to y_flow_sd[n-1]:
                # sum over r to get shape (s,o,d)
                sum_over_r_sd = (flow_delivery_all_sd * mask_sd_full).sum(axis=0)  # (s,o,d)
                # transpose (s,o,d) to (o,s,d):
                sum_over_r_sd = sum_over_r_sd.transpose(1,0,2)
                y_flow_sd[n-1, :, :, :] += sum_over_r_sd

    y_od_rs = np.sum(delivery_count_od, axis=0)
    y_rs = np.sum(y_od_rs, axis=(2,3))
    return y_od_rs, y_rs





#############################""
#not useful rn
def compute_total_link_flow(n_size, 
                            o_nodes, d_nodes, r_nodes, s_nodes, 
                            node_to_index, 
                            y_od_bar, 
                            P_od_rs,  # shape: (r_size, s_size, o_size, d_size)
                            P_sd_rs,  # shape: (n_size, r_size, s_size, s_size, d_size)
                            threshold=1e-12):

    # Sizes
    o_size = len(o_nodes)
    d_size = len(d_nodes)
    r_size = len(r_nodes)
    s_size = len(s_nodes)

    # We'll maintain the same 'y_flow_od' and 'y_flow_sd' that appear in compute_y_distrib:
    y_flow_od = np.zeros((n_size, o_size, d_size))
    y_flow_sd = np.zeros((n_size, o_size, s_size, d_size))
    # Initially, all demand is at the last layer (n_size-1)
    y_flow_od[n_size - 1, :, :] = y_od_bar

    # Initialize a matrix of link flows in global indexing
    N = len(node_to_index)
    link_flow = np.zeros((N, N))  # will accumulate all flow i->j

    # Helper to add flow to link_flow
    def add_flow(i_label, j_label, flow_val):
        if flow_val > threshold:
            i_idx = node_to_index[i_label]
            j_idx = node_to_index[j_label]
            link_flow[i_idx, j_idx] += flow_val

    
    for n in range(n_size - 1, -1, -1):
        
        flow_od = y_flow_od[n]  # shape (o_size, d_size)
        # broadcast: shape => (r_size, s_size, o_size, d_size)
        flow_delivery_all = P_od_rs * flow_od[np.newaxis, np.newaxis, :, :]

        for r_idx, r_node in enumerate(r_nodes):
            for s_idx, _ in enumerate(s_nodes):
                for o_idx, o_node in enumerate(o_nodes):
                    for d_idx, d_node in enumerate(d_nodes):
                        fval = flow_delivery_all[r_idx, s_idx, o_idx, d_idx]
                        if fval > threshold:
                            
                            add_flow(o_node, r_node, fval)


        if n > 0:
            sum_over_r = flow_delivery_all.sum(axis=0)  # shape (s_size, o_size, d_size)
            sum_over_r = sum_over_r.transpose(1, 0, 2)  # => (o_size, s_size, d_size)
            y_flow_sd[n - 1] += sum_over_r

        if n > 0:
            # y_flow_sd at step n => shape (o_size, s_size, d_size)
            flow_sd_n = y_flow_sd[n]
            for s_idx, s_node_label in enumerate(s_nodes):
                flow_sd_slice = flow_sd_n[:, s_idx, :]  # shape (o_size, d_size)
                if not np.any(flow_sd_slice > threshold):
                    continue
                P_sd_rs_s = P_sd_rs[n, :, :, s_idx, :]  # shape: (r_size, s_size, d_size)

                # flow_delivery_all_sd => shape (r_size, s_size, o_size, d_size)
                flow_delivery_all_sd = P_sd_rs_s[:, :, np.newaxis, :] * flow_sd_slice[np.newaxis, np.newaxis, :, :]

                for r_idx, r_node_label in enumerate(r_nodes):
                    for s2_idx, s2_node_label in enumerate(s_nodes):
                        for o_idx, o_node_label in enumerate(o_nodes):
                            for d_idx, d_node_label in enumerate(d_nodes):
                                fval = flow_delivery_all_sd[r_idx, s2_idx, o_idx, d_idx]
                                if fval > threshold:
                                    add_flow(r_node_label, s2_node_label, fval)

                sum_over_r_sd = flow_delivery_all_sd.sum(axis=0)  # => (s_size, o_size, d_size)
                sum_over_r_sd = sum_over_r_sd.transpose(1, 0, 2)
                y_flow_sd[n - 1] += sum_over_r_sd

    return link_flow

