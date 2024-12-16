import numpy as np
import math

# choses à changer (putain ça va être long)
#changer rs
# tout ce qui est en rs, passer en 2 dimensions, r puis s. Donc changer les boucles sur rs (ou souvent r) par deux boucles (r puis s)

def build_node_order_and_t(o_dict, r_dict, s_dict, d_dict): #ici rien à changer tout va bien
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

def initialize_downstream_arrays(n_size, r_size, s_size, d_size):
    # P_sd_rs(n,r,s,s,d) et d_mu_sd(n,r,s,s,d)
    iterations = max(2, n_size - 1)
    Z_sd = np.zeros((iterations, s_size, d_size))
    mu_sd = np.zeros((iterations, s_size, d_size))
    P_sd_rs = np.zeros((iterations, r_size, s_size, s_size, d_size)) 
    d_mu_sd = np.zeros((iterations, r_size, s_size, s_size, d_size))
    return Z_sd, mu_sd, P_sd_rs, d_mu_sd

def initialize_mu_sd_first_iteration(mu_sd, t, d_size, s_nodes, node_to_index):
    for s_idx, s_node in enumerate(s_nodes):
        mu_sd[0, s_idx, :] = t[node_to_index[s_node], :d_size]

def compute_mu_sd_and_P_sd(Z_sd, mu_sd, P_sd_rs, t, node_to_index, d_nodes, r_nodes, s_nodes, theta_p, v):
    iterations = Z_sd.shape[0]
    d_size = len(d_nodes)
    r_size = len(r_nodes)
    s_size = len(s_nodes)
    exp = np.exp

    for n in range(1, iterations):
        mu_sd_prev = mu_sd[n - 1]
        mu_sd_curr = mu_sd[n]
        Z_sd_curr = Z_sd[n]
        P_sd_rs_curr = P_sd_rs[n]

        for s_idx, s_node in enumerate(s_nodes): #s_idx is the index of s starting from 0, used for Z matrix for example
            s_index = node_to_index[s_node] #index of s in the DORS matrix
            for d_idx in range(d_size):
                sum_tot = 0.0
                # somme sur tous (r,s')
                for r_inside_idx, r_node in enumerate(r_nodes):
                    r_outside_idx = node_to_index[r_node]
                    for s_inside_idx2, s_node2 in enumerate(s_nodes):
                        s__outside_index2 = node_to_index[s_node2]
                        if t[r_outside_idx, s__outside_index2] > 0:
                            val = exp(-theta_p*(t[s_index,r_outside_idx]  + t[r_outside_idx, s__outside_index2] - v[r_inside_idx, s_inside_idx2] + mu_sd_prev[s_inside_idx2, d_idx]))
                            sum_tot += val
                            #validé jusque là

                Z_val = exp(-theta_p*t[s_idx,d_idx]) + sum_tot
                Z_sd_curr[s_idx, d_idx] = Z_val
                mu_sd_curr[s_idx, d_idx] = (-1/theta_p)*math.log(Z_val)
                inv_Z = 1.0/Z_val

                for r_inside_idx, r_node in enumerate(r_nodes):
                    r_outside_idx = node_to_index[r_node]
                    for s_inside_idx2, s_node2 in enumerate(s_nodes):
                        s__outside_index2 = node_to_index[s_node2]
                        if t[r_outside_idx, s__outside_index2]>0:
                            P_sd_rs_curr[r_inside_idx, s_inside_idx2, s_idx, d_idx] = exp(-theta_p*(t[s_index,r_outside_idx] + t[r_outside_idx,s__outside_index2] - v[r_inside_idx, s_inside_idx2] + mu_sd_prev[s_inside_idx2,d_idx]))*inv_Z
                        else:
                            P_sd_rs_curr[r_inside_idx, s_inside_idx2, s_idx, d_idx] = 0.0
                # ça m'a l'air ok aussi.

def compute_d_mu_sd(d_mu_sd, P_sd_rs, r_nodes, s_nodes, d_nodes):
    # d_mu_sd(n,r,s,s,d), P_sd_rs(n,r,s,s,d)
    iterations = d_mu_sd.shape[0]
    r_size = len(r_nodes)
    s_size = len(s_nodes)
    d_size = len(d_nodes)

    for n in range(1, iterations):
        P_sd_rs_n = P_sd_rs[n]
        d_mu_sd_prev = d_mu_sd[n-1]
        d_mu_sd_curr = d_mu_sd[n]

        for s__inside_idx in range(s_size):
            for d_inside_idx in range(d_size):
                for r_inside_idx in range(r_size):
                    for s_inside_index2 in range(s_size):
                        # sum over (r_in,s_in)
                        sum_tot2 = 0.0
                        for r_in in range(r_size):
                            for s_in in range(s_size):
                                sum_tot2 += P_sd_rs_n[r_in, s_in, s__inside_idx, d_inside_idx]*d_mu_sd_prev[r_inside_idx, s_inside_index2, s_in, d_inside_idx]

                        d_mu_sd_curr[r_inside_idx, s_inside_index2, s__inside_idx, d_inside_idx] = -P_sd_rs_n[r_inside_idx, s_inside_index2, s__inside_idx, d_inside_idx] + sum_tot2

def compute_value_functions(rng, r_size, s_size, K, c_rs, v, theta_c):
    # c_rs(r,s,K), v(r,s)
    epsilon = rng.gumbel(0, theta_c, size=(r_size,s_size,K))*0.1
    C_rs = c_rs + epsilon
    sum_exp = np.zeros((r_size,s_size))
    for rr_idx in range(r_size):
        for ss_idx in range(s_size):
            val0 = np.exp(-theta_c*(C_rs[rr_idx,ss_idx,0]))
            val1 = np.exp(-theta_c*(C_rs[rr_idx,ss_idx,1]+v[rr_idx,ss_idx]))
            sum_exp[rr_idx, ss_idx] = val0+val1
    bigV_rs = (-1/theta_c)*np.log(sum_exp)
    return C_rs, sum_exp, bigV_rs

def compute_z_rs_1(z_rs_bar, C_rs, sum_exp, v, theta_c):
    return z_rs_bar * (np.exp(-theta_c*(C_rs[:,:,1]+v))) / sum_exp

def compute_z_rs_0(z_rs_bar, C_rs, sum_exp, v, theta_c):
    return z_rs_bar * (np.exp(-theta_c*C_rs[:,:,0])) / sum_exp

def initialize_upstream_arrays(o_size, d_size, r_size, s_size):
    Z_od = np.zeros((o_size, d_size))
    mu_od = np.zeros((o_size, d_size))
    P_od_rs = np.zeros((r_size, s_size, o_size, d_size))
    P_od_no_rs = np.ones((o_size, d_size))
    d_mu_od = np.zeros((r_size, s_size, o_size, d_size))
    return Z_od, mu_od, P_od_rs, P_od_no_rs, d_mu_od

def compute_mu_od_P_od(Z_od, mu_od, P_od_rs, t, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes, mu_sd, n_size, theta_p, v):
    d_size = len(d_nodes)
    exp = np.exp
    mu_sd_last = mu_sd[n_size - 2]

    for o_idx, o_node in enumerate(o_nodes):
        o_index = node_to_index[o_node]
        for d_idx in range(d_size):
            sum_tot = 0.0
            for r_inside_idx, r_node in enumerate(r_nodes):
                r_outside_index = node_to_index[r_node]
                for s_inside_idx, s_node in enumerate(s_nodes):
                    s_index2 = node_to_index[s_node]
                    if t[r_outside_index,s_index2]>0:
                        sum_tot += exp(-theta_p*(t[o_index,r_outside_index]+t[r_outside_index,s_index2]-v[r_inside_idx,s_inside_idx]+mu_sd_last[s_inside_idx,d_idx]))
            Z_val = exp(-theta_p*t[o_index,d_idx]) + sum_tot
            Z_od[o_idx, d_idx] = Z_val
            mu_od[o_idx, d_idx] = (-1/theta_p)*math.log(Z_val)
            inv_Z = 1.0/Z_val

            for r_inside_idx, r_node in enumerate(r_nodes):
                r_outside_index = node_to_index[r_node]
                for s_inside_idx, s_node in enumerate(s_nodes):
                    s_outside_index2 = node_to_index[s_node]
                    if t[r_outside_index, s_outside_index2]>0:
                        P_od_rs[r_inside_idx, s_inside_idx, o_idx, d_idx] = exp(-theta_p*(t[o_index,r_outside_index]+t[r_outside_index,s_outside_index2]-v[r_inside_idx,s_inside_idx]+mu_sd_last[s_inside_idx,d_idx]))*inv_Z
                    else:
                        P_od_rs[r_inside_idx, s_inside_idx, o_idx, d_idx] = 0.0
            #malheureusement ça à l'air ok aussi...

def compute_d_mu_od(d_mu_od, P_od_rs, d_mu_sd, n_size, r_nodes, o_nodes, d_nodes, s_nodes, P_od_no_rs):
    # d_mu_od(r,s,o,d), P_od_rs(r,s,o,d), d_mu_sd(r,s,s,s,d)
    r_size = len(r_nodes)
    s_size = len(s_nodes)
    o_size = len(o_nodes)
    d_size = len(d_nodes)
    d_mu_sd_last = d_mu_sd[n_size - 2]

    for rr_idx in range(r_size):
        for ss_idx in range(s_size):
            for o_idx in range(o_size):
                for d_idx in range(d_size):
                    sum_tot2 = 0.0
                    # sum over r_in,s_in,s_state_idx
                    for r_in in range(r_size):
                        for s_in in range(s_size):  
                            sum_tot2 += P_od_rs[r_in, s_in, o_idx, d_idx]*d_mu_sd_last[rr_idx, ss_idx, s_in, d_idx]
                    d_mu_od[rr_idx, ss_idx, o_idx, d_idx] = -P_od_rs[rr_idx, ss_idx, o_idx, d_idx] + sum_tot2

    #for o_idx in range(o_size):
    #    for d_idx in range(d_size):
    #        P_od_no_rs[o_idx, d_idx] = 1 - np.sum(P_od_rs[:,:,o_idx,d_idx])

def compute_MC(mu_od, y_od_bar, bigV_rs, z_rs_bar):
    return np.sum(y_od_bar*mu_od)+np.sum(z_rs_bar*bigV_rs)

def compute_grad_MC(grad_MC, y_od_bar, d_mu_od, z_rs_1):
    # grad_MC(r,s)
    sum_dmu = np.sum(y_od_bar*d_mu_od, axis=(2,3)) # sum over o,d
    grad_MC[:,:] = sum_dmu + z_rs_1

def compute_MC_and_grad_MC(o_dict, r_dict, s_dict, d_dict, n_size, K, theta_p, theta_c, z_rs_bar, y_od_bar, c_rs, seed, v):
    rng = np.random.default_rng(seed)
    t,node_to_index, d_nodes, o_nodes, r_nodes, s_nodes = build_node_order_and_t(o_dict, r_dict, s_dict, d_dict)
    r_size = len(r_nodes)
    s_size = len(s_nodes)
    d_size = len(d_nodes)
    o_size = len(o_nodes)

    Z_sd, mu_sd, P_sd_rs, d_mu_sd = initialize_downstream_arrays(n_size, r_size, s_size, d_size)
    
    initialize_mu_sd_first_iteration(mu_sd, t, d_size, s_nodes, node_to_index)
    
    compute_mu_sd_and_P_sd(Z_sd, mu_sd, P_sd_rs, t, node_to_index, d_nodes, r_nodes, s_nodes, theta_p, v)
    
    compute_d_mu_sd(d_mu_sd, P_sd_rs, r_nodes, s_nodes, d_nodes)
    
    C_rs, sum_exp, bigV_rs = compute_value_functions(rng, r_size, s_size, K, c_rs, v, theta_c)
    
    z_rs_1_val = compute_z_rs_1(z_rs_bar, C_rs, sum_exp, v, theta_c)
    
    Z_od, mu_od, P_od_rs, P_od_no_rs, d_mu_od = initialize_upstream_arrays(o_size, d_size, r_size, s_size)
    
    compute_mu_od_P_od(Z_od, mu_od, P_od_rs, t, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes, mu_sd, n_size, theta_p, v)
    
    compute_d_mu_od(d_mu_od, P_od_rs, d_mu_sd, n_size, r_nodes, o_nodes, d_nodes, s_nodes, P_od_no_rs)
    
    MC = compute_MC(mu_od, y_od_bar, bigV_rs, z_rs_bar)
    
    grad_MC = np.zeros((r_size, s_size))
    
    compute_grad_MC(grad_MC, y_od_bar, d_mu_od, z_rs_1_val)
    return MC, grad_MC

def compute_MC_and_grad_MC_return_intermediate(o_dict, r_dict, s_dict, d_dict, n_size, K, theta_p, theta_c, z_rs_bar, y_od_bar, c_rs, seed, v):
    rng = np.random.default_rng(seed)
    
    t, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes = build_node_order_and_t(o_dict, r_dict, s_dict, d_dict)
    r_size = len(r_nodes)
    s_size = len(s_nodes)
    d_size = len(d_nodes)
    o_size = len(o_nodes)

    Z_sd, mu_sd, P_sd_rs, d_mu_sd = initialize_downstream_arrays(n_size, r_size, s_size, d_size)
    
    initialize_mu_sd_first_iteration(mu_sd, t, d_size, s_nodes, node_to_index)
    
    compute_mu_sd_and_P_sd(Z_sd, mu_sd, P_sd_rs, t, node_to_index, d_nodes, r_nodes, s_nodes, theta_p, v)
    
    compute_d_mu_sd(d_mu_sd, P_sd_rs, r_nodes, s_nodes, d_nodes)
    
    C_rs, sum_exp, bigV_rs = compute_value_functions(rng, r_size, s_size, K, c_rs, v, theta_c)
    
    z_rs_1_val = compute_z_rs_1(z_rs_bar, C_rs, sum_exp, v, theta_c)
    
    Z_od, mu_od, P_od_rs, P_od_no_rs, d_mu_od = initialize_upstream_arrays(o_size, d_size, r_size, s_size)
    
    compute_mu_od_P_od(Z_od, mu_od, P_od_rs, t, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes, mu_sd, n_size, theta_p, v)
    
    compute_d_mu_od(d_mu_od, P_od_rs, d_mu_sd, n_size, r_nodes, o_nodes, d_nodes, s_nodes, P_od_no_rs)
    
    MC = compute_MC(mu_od, y_od_bar, bigV_rs, z_rs_bar)
    
    grad_MC = np.zeros((r_size, s_size))
    
    compute_grad_MC(grad_MC, y_od_bar, d_mu_od, z_rs_1_val)
    return MC, grad_MC, C_rs, sum_exp, theta_c, P_od_rs, P_od_no_rs, P_sd_rs, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes

def compute_y_distrib(r_size,n_size,s_size, o_size, d_size,y_od_bar,P_od_rs,P_od_no_rs,P_sd_rs,node_to_index,d_nodes,o_nodes,r_nodes,s_nodes):
    # double boucle sur (r,s)
    y_flow_od = np.zeros((n_size, o_size, d_size))
    y_flow_sd = np.zeros((n_size, o_size, s_size, d_size))
    y_flow_od[n_size - 1, :, :] = y_od_bar

    delivery_count_od = np.zeros((n_size, r_size, s_size, o_size, d_size))
    for n in range(n_size - 1, -1, -1):
        y_flow_od_n = y_flow_od[n]
        for o_idx in range(o_size):
            for d_idx in range(d_size):
                flow_od = y_flow_od_n[o_idx, d_idx]
                if flow_od > 1e-12:
                    flow_delivery_all = flow_od * P_od_rs[:,:,o_idx,d_idx] # (r,s)
                    for rr_idx in range(r_size):
                        for ss_idx in range(s_size):
                            flow_delivery = flow_delivery_all[rr_idx, ss_idx]
                            if flow_delivery > 1e-12:
                                delivery_count_od[n, rr_idx, ss_idx, o_idx, d_idx] += flow_delivery
                                if n - 1 >= 0:
                                    r_node = r_nodes[rr_idx]
                                    s_node2 = s_nodes[ss_idx]
                                    r_index = node_to_index[r_node]
                                    s_index2 = node_to_index[s_node2]
                                    s_new_idx = s_index2 - (len(d_nodes) + len(o_nodes) + len(r_nodes))
                                    y_flow_sd[n - 1, o_idx, s_new_idx, d_idx] += flow_delivery

        if n > 0:
            y_flow_sd_n = y_flow_sd[n]
            y_flow_sd_nm1 = y_flow_sd[n-1]
            for o_idx in range(o_size):
                for s_idx in range(s_size):
                    for d_idx in range(d_size):
                        flow_sd = y_flow_sd_n[o_idx, s_idx, d_idx]
                        if flow_sd > 1e-12:
                            P_sd_n = P_sd_rs[n, :, :, s_idx, d_idx] # (r,s)
                            flow_delivery_all = flow_sd * P_sd_n
                            for rr_idx in range(r_size):
                                for ss_idx in range(s_size):
                                    flow_delivery = flow_delivery_all[rr_idx, ss_idx]
                                    if flow_delivery > 1e-12:
                                        delivery_count_od[n, rr_idx, ss_idx, o_idx, d_idx] += flow_delivery
                                        r_node = r_nodes[rr_idx]
                                        s_node2 = s_nodes[ss_idx]
                                        r_index = node_to_index[r_node]
                                        s_index2 = node_to_index[s_node2]
                                        s_new_idx = s_index2 - (len(d_nodes) + len(o_nodes) + len(r_nodes))
                                        y_flow_sd_nm1[o_idx, s_new_idx, d_idx] += flow_delivery

    y_od_rs = np.sum(delivery_count_od, axis=0)
    y_rs = np.sum(y_od_rs, axis=(2,3))
    return y_od_rs, y_rs
