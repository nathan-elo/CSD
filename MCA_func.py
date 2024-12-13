import numpy as np
import math

def build_node_order_and_t(o_dict, r_dict, s_dict, d_dict):
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

    return t, node_order, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes

def compute_r_to_s_mapping(r_dict, node_to_index): # à changer
    r_to_s_mapping = {}
    for r_node, s_links in r_dict.items():
        s_node = list(s_links.keys())[0]
        r_to_s_mapping[node_to_index[r_node]] = node_to_index[s_node]
    return r_to_s_mapping

def initialize_downstream_arrays(n_size, s_size, d_size, r_size):
    iterations = max(2, n_size - 1)
    Z_sd = np.zeros((iterations, s_size, d_size))
    mu_sd = np.zeros((iterations, s_size, d_size))
    P_sd_rs = np.zeros((iterations, r_size, s_size, d_size))
    d_mu_sd = np.zeros((iterations, r_size, s_size, d_size))
    return Z_sd, mu_sd, P_sd_rs, d_mu_sd

def initialize_mu_sd_first_iteration(mu_sd, t, d_size, s_nodes, node_to_index):
    for s_idx, s_node in enumerate(s_nodes):
        mu_sd[0, s_idx, :] = t[node_to_index[s_node], :d_size]

def compute_mu_sd_and_P_sd(Z_sd, mu_sd, P_sd_rs, t, node_to_index, d_nodes, r_nodes, s_nodes, r_to_s_mapping, n_size, theta_p, v):
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
        for s_idx, s_node in enumerate(s_nodes):
            s_index = node_to_index[s_node]
            t_s = t[s_index]
            for d_idx in range(d_size):
                d_index = d_idx
                sum_tot = 0.0
                for rs_idx, r_node in enumerate(r_nodes):
                    r_index = node_to_index[r_node]
                    s_corr = r_to_s_mapping[r_index]
                    val = exp(-theta_p*(t_s[r_index] + t[r_index, s_corr] - v[rs_idx] + mu_sd_prev[rs_idx, d_idx])) # à changer 
                    sum_tot += val
                Z_val = exp(-theta_p*t_s[d_index]) + sum_tot
                Z_sd_curr[s_idx, d_idx] = Z_val
                mu_sd_curr[s_idx, d_idx] = (-1/theta_p)*math.log(Z_val)
                inv_Z = 1.0/Z_val
                for rs_idx, r_node in enumerate(r_nodes):
                    r_index = node_to_index[r_node]
                    s_corr = r_to_s_mapping[r_index]
                    P_sd_rs_curr[rs_idx, s_idx, d_idx] = exp(-theta_p*(t_s[r_index] + t[r_index, s_corr] - v[rs_idx] + mu_sd_prev[rs_idx, d_idx]))*inv_Z 
                    # à changer aussi

def compute_d_mu_sd(d_mu_sd, P_sd_rs, mu_sd, n_size, r_nodes, s_nodes, d_nodes):
    iterations = d_mu_sd.shape[0]
    r_size = len(r_nodes)
    s_size = len(s_nodes)
    d_size = len(d_nodes)
    for n in range(1, iterations):
        P_sd_rs_n = P_sd_rs[n]
        d_mu_sd_prev = d_mu_sd[n-1]
        d_mu_sd_curr = d_mu_sd[n]
        for s_idx in range(s_size):
            for d_idx in range(d_size):
                for rs_part in range(r_size):
                    P_col = P_sd_rs_n[:, s_idx, d_idx]
                    d_prev_col = d_mu_sd_prev[rs_part, :, d_idx]
                    sum_tot2 = np.dot(P_col, d_prev_col)
                    d_mu_sd_curr[rs_part, s_idx, d_idx] = -P_sd_rs_n[rs_part, s_idx, d_idx] + sum_tot2

def compute_value_functions(rng, rs_size, K, c_rs, v, theta_c):
    epsilon = rng.gumbel(0, theta_c, size=(rs_size, K))*0.1
    C_rs = c_rs + epsilon
    sum_exp = np.zeros(rs_size)
    for rs_idx in range(rs_size):
        val0 = np.exp(-theta_c*(C_rs[rs_idx,0]))
        val1 = np.exp(-theta_c*(C_rs[rs_idx,1]+v[rs_idx]))
        sum_exp[rs_idx] = val0+val1
    bigV_rs = (-1/theta_c)*np.log(sum_exp)
    return C_rs, sum_exp, bigV_rs

def compute_z_rs_1(z_rs_bar, C_rs, sum_exp, v, theta_c):
    return z_rs_bar * (np.exp(-theta_c * (C_rs[:, 1] + v))) / sum_exp

def compute_z_rs_0(z_rs_bar, C_rs, sum_exp, v, theta_c):
    return z_rs_bar * (np.exp(-theta_c * C_rs[:, 0])) / sum_exp

def initialize_upstream_arrays(o_size, d_size, r_size):
    Z_od = np.zeros((o_size, d_size))
    mu_od = np.zeros((o_size, d_size))
    P_od_rs = np.zeros((r_size, o_size, d_size))
    P_od_no_rs = np.ones((o_size, d_size))
    d_mu_od = np.zeros((r_size, o_size, d_size))
    return Z_od, mu_od, P_od_rs, P_od_no_rs, d_mu_od

def compute_mu_od_P_od(Z_od, mu_od, P_od_rs, t, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes, r_to_s_mapping, mu_sd, n_size, theta_p, v):
    d_size = len(d_nodes)
    o_size = len(o_nodes)
    r_size = len(r_nodes)
    exp = np.exp
    mu_sd_last = mu_sd[n_size - 2]

    for o_idx, o_node in enumerate(o_nodes):
        o_index = node_to_index[o_node]
        t_o = t[o_index]
        for d_idx in range(d_size):
            sum_tot = 0.0
            for rs_idx, r_node in enumerate(r_nodes):
                r_index = node_to_index[r_node]
                s_corr = r_to_s_mapping[r_index]
                sum_tot += exp(-theta_p*(t_o[r_index] + t[r_index, s_corr] - v[rs_idx] + mu_sd_last[rs_idx, d_idx]))
            Z_val = exp(-theta_p*t_o[d_idx]) + sum_tot
            Z_od[o_idx, d_idx] = Z_val
            mu_od[o_idx, d_idx] = (-1/theta_p)*math.log(Z_val)
            inv_Z = 1.0/Z_val
            for rs_idx, r_node in enumerate(r_nodes):
                r_index = node_to_index[r_node]
                s_corr = r_to_s_mapping[r_index]
                P_od_rs[rs_idx, o_idx, d_idx] = exp(-theta_p*(t_o[r_index]+t[r_index, s_corr]-v[rs_idx]+mu_sd_last[rs_idx, d_idx]))*inv_Z

def compute_d_mu_od(d_mu_od, P_od_rs, d_mu_sd, y_od_bar, n_size, r_nodes, o_nodes, d_nodes, P_od_no_rs):
    r_size = len(r_nodes)
    o_size = len(o_nodes)
    d_size = len(d_nodes)
    d_mu_sd_last = d_mu_sd[n_size - 2]
    for rs_part in range(r_size):
        for o_idx in range(o_size):
            for d_idx in range(d_size):
                P_col = P_od_rs[:, o_idx, d_idx]
                d_prev_col = d_mu_sd_last[rs_part, :, d_idx]
                sum_tot2 = np.dot(P_col, d_prev_col)
                d_mu_od[rs_part, o_idx, d_idx] = -P_od_rs[rs_part, o_idx, d_idx] + sum_tot2
    for o_idx in range(o_size):
        for d_idx in range(d_size):
            P_od_no_rs[o_idx, d_idx] = 1 - np.sum(P_od_rs[:, o_idx, d_idx])

def compute_MC(mu_od, y_od_bar, bigV_rs, z_rs_bar):
    return np.sum(y_od_bar*mu_od)+np.sum(z_rs_bar*bigV_rs)

def compute_grad_MC(grad_MC, y_od_bar, d_mu_od, z_rs_1, r_nodes):
    grad_MC[:] = np.sum(y_od_bar*d_mu_od, axis=(1,2)) + z_rs_1

def compute_MC_and_grad_MC(o_dict, r_dict, s_dict, d_dict, n_size, K, theta_p, theta_c, z_rs_bar, y_od_bar, c_rs, seed, v):
    rng = np.random.default_rng(seed)
    t, node_order, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes = build_node_order_and_t(o_dict, r_dict, s_dict, d_dict)
    r_size = len(r_nodes)
    s_size = len(s_nodes)
    d_size = len(d_nodes)
    o_size = len(o_nodes)

    r_to_s_mapping = compute_r_to_s_mapping(r_dict, node_to_index)

    Z_sd, mu_sd, P_sd_rs, d_mu_sd = initialize_downstream_arrays(n_size, s_size, d_size, r_size)
    
    initialize_mu_sd_first_iteration(mu_sd, t, d_size, s_nodes, node_to_index)
    
    compute_mu_sd_and_P_sd(Z_sd, mu_sd, P_sd_rs, t, node_to_index, d_nodes, r_nodes, s_nodes, r_to_s_mapping, n_size, theta_p, v)
    
    compute_d_mu_sd(d_mu_sd, P_sd_rs, mu_sd, n_size, r_nodes, s_nodes, d_nodes)
    
    C_rs, sum_exp, bigV_rs = compute_value_functions(rng, r_size, K, c_rs, v, theta_c)
    
    z_rs_1_val = compute_z_rs_1(z_rs_bar, C_rs, sum_exp, v, theta_c)
    
    Z_od, mu_od, P_od_rs, P_od_no_rs, d_mu_od = initialize_upstream_arrays(o_size, d_size, r_size)
    
    compute_mu_od_P_od(Z_od, mu_od, P_od_rs, t, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes, r_to_s_mapping, mu_sd, n_size, theta_p, v)
    
    compute_d_mu_od(d_mu_od, P_od_rs, d_mu_sd, y_od_bar, n_size, r_nodes, o_nodes, d_nodes, P_od_no_rs)
    
    MC = compute_MC(mu_od, y_od_bar, bigV_rs, z_rs_bar)
    
    grad_MC = np.zeros((r_size))
    
    compute_grad_MC(grad_MC, y_od_bar, d_mu_od, z_rs_1_val, r_nodes)
    return MC, grad_MC

def compute_MC_and_grad_MC_return_intermediate(o_dict, r_dict, s_dict, d_dict, n_size, K, theta_p, theta_c, z_rs_bar, y_od_bar, c_rs, seed, v):
    rng = np.random.default_rng(seed)
    
    t, node_order, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes = build_node_order_and_t(o_dict, r_dict, s_dict, d_dict)
    r_size = len(r_nodes)
    s_size = len(s_nodes)
    d_size = len(d_nodes)
    o_size = len(o_nodes)

    
    r_to_s_mapping = compute_r_to_s_mapping(r_dict, node_to_index)
    
    Z_sd, mu_sd, P_sd_rs, d_mu_sd = initialize_downstream_arrays(n_size, s_size, d_size, r_size)
    
    initialize_mu_sd_first_iteration(mu_sd, t, d_size, s_nodes, node_to_index)
    
    compute_mu_sd_and_P_sd(Z_sd, mu_sd, P_sd_rs, t, node_to_index, d_nodes, r_nodes, s_nodes, r_to_s_mapping, n_size, theta_p, v)
    
    compute_d_mu_sd(d_mu_sd, P_sd_rs, mu_sd, n_size, r_nodes, s_nodes, d_nodes)
    
    C_rs, sum_exp, bigV_rs = compute_value_functions(rng, r_size, K, c_rs, v, theta_c)
    
    z_rs_1_val = compute_z_rs_1(z_rs_bar, C_rs, sum_exp, v, theta_c)
    
    Z_od, mu_od, P_od_rs, P_od_no_rs, d_mu_od = initialize_upstream_arrays(o_size, d_size, r_size)
    
    compute_mu_od_P_od(Z_od, mu_od, P_od_rs, t, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes, r_to_s_mapping, mu_sd, n_size, theta_p, v)
    
    compute_d_mu_od(d_mu_od, P_od_rs, d_mu_sd, y_od_bar, n_size, r_nodes, o_nodes, d_nodes, P_od_no_rs)
    
    MC = compute_MC(mu_od, y_od_bar, bigV_rs, z_rs_bar)
    
    grad_MC = np.zeros((r_size))
    
    compute_grad_MC(grad_MC, y_od_bar, d_mu_od, z_rs_1_val, r_nodes)
    return MC, grad_MC, C_rs, sum_exp, theta_c, P_od_rs, P_od_no_rs, P_sd_rs, node_to_index, r_to_s_mapping, d_nodes, o_nodes, r_nodes

def compute_y_distrib(r_size,n_size,s_size, o_size, d_size,y_od_bar,P_od_rs,P_od_no_rs,P_sd_rs,node_to_index,r_to_s_mapping,d_nodes,o_nodes,r_nodes):
    y_flow_od = np.zeros((n_size, o_size, d_size))
    y_flow_sd = np.zeros((n_size, o_size, s_size, d_size))
    y_flow_od[n_size - 1, :, :] = y_od_bar

    delivery_count_od = np.zeros((n_size, r_size, o_size, d_size))
    for n in range(n_size - 1, -1, -1):
        y_flow_od_n = y_flow_od[n]
        for o_idx in range(o_size):
            for d_idx in range(d_size):
                flow_od = y_flow_od_n[o_idx, d_idx]
                if flow_od > 1e-12:
                    P_col = P_od_rs[:, o_idx, d_idx]
                    flow_delivery_all = flow_od * P_col
                    for rs_idx, flow_delivery in enumerate(flow_delivery_all):
                        if flow_delivery > 1e-12:
                            delivery_count_od[n, rs_idx, o_idx, d_idx] += flow_delivery
                            if n - 1 >= 0:
                                r_index = node_to_index[r_nodes[rs_idx]]
                                s_corresponding = r_to_s_mapping[r_index]
                                s_new_idx = s_corresponding - (len(d_nodes) + len(o_nodes) + len(r_nodes))
                                y_flow_sd[n - 1, o_idx, s_new_idx, d_idx] += flow_delivery
                    # No delivery:
                    # flow_no_delivery = flow_od * P_od_no_rs[o_idx, d_idx] (not currently used)

        if n > 0:
            y_flow_sd_n = y_flow_sd[n]
            y_flow_sd_nm1 = y_flow_sd[n-1]
            for o_idx in range(o_size):
                for s_idx in range(s_size):
                    for d_idx in range(d_size):
                        flow_sd = y_flow_sd_n[o_idx, s_idx, d_idx]
                        if flow_sd > 1e-12:
                            P_sd_n = P_sd_rs[n, :, s_idx, d_idx]
                            flow_delivery_all = flow_sd * P_sd_n
                            for rs_idx, flow_delivery in enumerate(flow_delivery_all):
                                if flow_delivery > 1e-12:
                                    delivery_count_od[n, rs_idx, o_idx, d_idx] += flow_delivery
                                    r_index = node_to_index[r_nodes[rs_idx]]
                                    s_corresponding = r_to_s_mapping[r_index]
                                    s_new_idx = s_corresponding - (len(d_nodes) + len(o_nodes) + len(r_nodes))
                                    y_flow_sd_nm1[o_idx, s_new_idx, d_idx] += flow_delivery

    y_od_rs = np.sum(delivery_count_od, axis=0)
    y_rs = np.sum(y_od_rs, axis=(1,2))
    return y_od_rs, y_rs
