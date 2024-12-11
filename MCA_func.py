import numpy as np

def build_node_order_and_t(o_dict, r_dict, s_dict, d_dict):
    """
    Build the node order and the cost matrix t.
    Node order: D O R S
    Return : cost matrix, node order, node dictionnaries
    """
    d_nodes = list(d_dict.keys())
    o_nodes = list(o_dict.keys())
    r_nodes = list(r_dict.keys())
    s_nodes = list(s_dict.keys())

    node_order = d_nodes + o_nodes + r_nodes + s_nodes
    d_size = len(d_nodes)
    o_size = len(o_nodes)
    r_size = len(r_nodes)
    s_size = len(s_nodes)
    t_size = len(node_order)

    node_to_index = {node: i for i, node in enumerate(node_order)}

    # Initialize t matrix
    t = np.zeros((t_size, t_size))

    # Fill t for O->(D/R) links
    for o_node, links in o_dict.items():
        for dest_node, value in links.items():
            t[node_to_index[o_node], node_to_index[dest_node]] = value

    # Fill t for R->S links
    for r_node, links in r_dict.items():
        for s_node, value in links.items():
            t[node_to_index[r_node], node_to_index[s_node]] = value

    # Fill t for S->R and D links
    for s_node, links in s_dict.items():
        for dest_node, value in links.items():
            t[node_to_index[s_node], node_to_index[dest_node]] = value

    return t, node_order, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes


def compute_r_to_s_mapping(r_dict, node_to_index):
    """
    Compute mapping from R nodes to their corresponding S node.
    Assumes each R node is connected to a single S node.
    """
    r_to_s_mapping = {}
    for r_node, s_links in r_dict.items():
        s_node = list(s_links.keys())[0]
        r_to_s_mapping[node_to_index[r_node]] = node_to_index[s_node]
    return r_to_s_mapping


def initialize_downstream_arrays(n_size, s_size, d_size, r_size):
    """
    Initialize arrays for backward calculations.
    """
    iterations = max(2, n_size - 1)
    Z_sd = np.zeros((iterations, s_size, d_size))
    mu_sd = np.zeros((iterations, r_size, d_size))
    P_sd_rs = np.zeros((iterations, r_size, s_size, d_size))
    d_mu_sd = np.zeros((iterations, r_size, s_size, d_size))
    return Z_sd, mu_sd, P_sd_rs, d_mu_sd


def initialize_mu_sd_first_iteration(mu_sd, t, d_size, s_nodes, node_to_index):
    """
    Initialize mu_sd for the first iteration (n=0) where mu_sd=t_sd.
    Other values like P(0), Z(0) and d_mu(0) stay null.
    """
    i = 0
    # Starting index for S nodes: after D and O and R
    # node order: D O R S
    # So S nodes start at index d_size + o_size + r_size
    # We can just iterate over s_nodes directly.
    for s_idx, s_node in enumerate(s_nodes):
        s_index = node_to_index[s_node]
        mu_sd[0, s_idx, :] = t[s_index, :d_size]
        i += 1


def compute_mu_sd_and_P_sd(Z_sd, mu_sd, P_sd_rs, t, node_to_index, d_nodes, r_nodes, s_nodes, r_to_s_mapping, n_size, theta_p, v):
    """
    Compute mu_sd and P_sd_rs for n >= 1.
    """
    iterations = Z_sd.shape[0]
    d_size = len(d_nodes)
    r_size = len(r_nodes)
    s_size = len(s_nodes)

    for n in range(1, iterations):
        for s_idx, s_node in enumerate(s_nodes):
            s_index = node_to_index[s_node]
            for d_idx, d_node in enumerate(d_nodes):
                d_index = node_to_index[d_node]
                sum_tot = 0.0
                for rs_idx, r_node in enumerate(r_nodes):
                    r_index = node_to_index[r_node]
                    s_corresponding = r_to_s_mapping[r_index]
                    sum_elem = np.exp(-theta_p * (
                        t[s_index, r_index] +
                        t[r_index, s_corresponding] - v[rs_idx] +
                        mu_sd[n - 1, rs_idx, d_idx]
                    ))
                    sum_tot += sum_elem

                Z_sd[n, s_idx, d_idx] = np.exp(-theta_p * t[s_index, d_index]) + sum_tot
                mu_sd[n, s_idx, d_idx] = (-1 / theta_p) * np.log(Z_sd[n, s_idx, d_idx])

                for rs_idx, r_node in enumerate(r_nodes):
                    r_index = node_to_index[r_node]
                    s_corresponding = r_to_s_mapping[r_index]
                    P_sd_rs[n, rs_idx, s_idx, d_idx] = (
                        np.exp(-theta_p * (t[s_index, r_index] + t[r_index, s_corresponding] - v[rs_idx] + mu_sd[n - 1, rs_idx, d_idx])) 
                        / Z_sd[n, s_idx, d_idx]
                    )


def compute_d_mu_sd(d_mu_sd, P_sd_rs, mu_sd, n_size, r_nodes, s_nodes, d_nodes):
    """
    Compute d_mu_sd for n >= 1.
    """
    iterations = d_mu_sd.shape[0]
    r_size = len(r_nodes)
    s_size = len(s_nodes)
    d_size = len(d_nodes)

    for n in range(1, iterations):
        for s_idx in range(s_size):
            for d_idx in range(d_size):
                for rs_part in range(r_size):
                    sum_tot2 = 0.0
                    for rs_in in range(r_size):
                        sum_elem2 = P_sd_rs[n, rs_in, s_idx, d_idx] * d_mu_sd[n - 1, rs_part, rs_in, d_idx]
                        sum_tot2 += sum_elem2
                    d_mu_sd[n, rs_part, s_idx, d_idx] = -P_sd_rs[n, rs_part, s_idx, d_idx] + sum_tot2


def compute_value_functions(rng, rs_size, K, c_rs, v, theta_c):
    """
    Compute the C_rs matrix (with Gumbel disturbances), sum_exp, bigV_rs, and return them.
    """
    loc = 0  # Gumbel location parameter
    scale = theta_c  # Gumbel scale parameter

    # Draw from Gumbel
    epsilon = rng.gumbel(loc, scale, size=(rs_size, K)) * 0.1
    C_rs = c_rs + epsilon

    sum_exp = np.zeros((rs_size))
    for rs_idx in range(rs_size):
        for k in range(K):
            sum_exp[rs_idx] += np.exp(-theta_c * (C_rs[rs_idx, k] + k * v[rs_idx]))

    bigV_rs = np.zeros((rs_size))
    for rs_idx in range(rs_size):
        bigV_rs[rs_idx] = -1 / theta_c * np.log(sum_exp[rs_idx])

    return C_rs, sum_exp, bigV_rs


def compute_z_rs_1(z_rs_bar, C_rs, sum_exp, v, theta_c):
    """
    Compute z_rs_1 
    """
    z_rs_1 = z_rs_bar * (np.exp(-theta_c * (C_rs[:, 1] + v))) / sum_exp
    return z_rs_1


def initialize_upstream_arrays(o_size, d_size, r_size):
    Z_od = np.zeros((o_size, d_size))
    mu_od = np.zeros((o_size, d_size))
    P_od_rs = np.zeros((r_size, o_size, d_size))
    d_mu_od = np.zeros((r_size, o_size, d_size))
    return Z_od, mu_od, P_od_rs, d_mu_od


def compute_mu_od_P_od(Z_od, mu_od, P_od_rs, t, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes, r_to_s_mapping, mu_sd, n_size, theta_p, v):
    """
    Compute mu_od and P_od_rs.
    """
    d_size = len(d_nodes)
    o_size = len(o_nodes)
    r_size = len(r_nodes)

    for o_idx, o_node in enumerate(o_nodes):
        o_index = node_to_index[o_node]
        for d_idx, d_node in enumerate(d_nodes):
            d_index = node_to_index[d_node]
            sum_tot = 0.0
            for rs_idx, r_node in enumerate(r_nodes):
                r_index = node_to_index[r_node]
                s_corresponding = r_to_s_mapping[r_index]
                sum_elem = np.exp(-theta_p * (
                    t[o_index, r_index] +
                    t[r_index, s_corresponding] - v[rs_idx] +
                    mu_sd[n_size - 2, rs_idx, d_idx]
                ))
                sum_tot += sum_elem

            Z_od[o_idx, d_idx] = np.exp(-theta_p * t[o_index, d_index]) + sum_tot
            mu_od[o_idx, d_idx] = (-1 / theta_p)*np.log(Z_od[o_idx, d_idx])

            for rs_idx, r_node in enumerate(r_nodes):
                r_index = node_to_index[r_node]
                s_corresponding = r_to_s_mapping[r_index]
                P_od_rs[rs_idx, o_idx, d_idx] = (
                    np.exp(-theta_p * (t[o_index, r_index] + t[r_index, s_corresponding] - v[rs_idx] + mu_sd[n_size - 2, rs_idx, d_idx]))
                    / Z_od[o_idx, d_idx]
                )


def compute_d_mu_od(d_mu_od, P_od_rs, d_mu_sd, y_od_bar, n_size, r_nodes, o_nodes, d_nodes):
    """
    Compute d_mu_od.
    """
    r_size = len(r_nodes)
    o_size = len(o_nodes)
    d_size = len(d_nodes)

    for rs_part in range(r_size):
        for o_idx in range(o_size):
            for d_idx in range(d_size):
                sum_tot2 = 0.0
                for rs_in in range(r_size):
                    sum_elem2 = P_od_rs[rs_in, o_idx, d_idx] * d_mu_sd[n_size - 2, rs_part, rs_in, d_idx]
                    sum_tot2 += sum_elem2
                d_mu_od[rs_part, o_idx, d_idx] = -P_od_rs[rs_part, o_idx, d_idx] + sum_tot2


def compute_MC(mu_od, y_od_bar, bigV_rs, z_rs_bar):
    sum_rs = np.sum(z_rs_bar * bigV_rs)
    sum_od = np.sum(y_od_bar * mu_od)
    MC = sum_od + sum_rs
    return MC


def compute_grad_MC(grad_MC, y_od_bar, d_mu_od, z_rs_1, r_nodes):
    r_size = len(r_nodes)
    for rs_idx in range(r_size):
        sum_part_od = np.sum(y_od_bar * d_mu_od[rs_idx, :, :])
        grad_MC[rs_idx] = sum_part_od + z_rs_1[rs_idx]


def compute_MC_and_grad_MC(o_dict, r_dict, s_dict, d_dict, n_size, K, theta_p, theta_c, z_rs_bar, y_od_bar, c_rs, seed, v):
    # Initialize random generator
    rng = np.random.default_rng(seed)

    # Build node order and t matrix
    t, node_order, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes = build_node_order_and_t(o_dict, r_dict, s_dict, d_dict)
    d_size = len(d_nodes)
    o_size = len(o_nodes)
    r_size = len(r_nodes)
    s_size = len(s_nodes)

    # Compute R->S mapping
    r_to_s_mapping = compute_r_to_s_mapping(r_dict, node_to_index)

    # Initialize arrays for downstream
    Z_sd, mu_sd, P_sd_rs, d_mu_sd = initialize_downstream_arrays(n_size, s_size, d_size, r_size)
    initialize_mu_sd_first_iteration(mu_sd, t, d_size, s_nodes, node_to_index)

    # Compute downstream flows
    compute_mu_sd_and_P_sd(Z_sd, mu_sd, P_sd_rs, t, node_to_index, d_nodes, r_nodes, s_nodes, r_to_s_mapping, n_size, theta_p, v)
    compute_d_mu_sd(d_mu_sd, P_sd_rs, mu_sd, n_size, r_nodes, s_nodes, d_nodes)

    # Compute value functions (C_rs, bigV_rs)
    C_rs, sum_exp, bigV_rs = compute_value_functions(rng, r_size, K, c_rs, v, theta_c)

    # Compute z_rs_1
    z_rs_1 = compute_z_rs_1(z_rs_bar, C_rs, sum_exp, v, theta_c)

    # Initialize arrays for upstream
    Z_od, mu_od, P_od_rs, d_mu_od = initialize_upstream_arrays(o_size, d_size, r_size)

    # Compute upstream flows
    compute_mu_od_P_od(Z_od, mu_od, P_od_rs, t, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes, r_to_s_mapping, mu_sd, n_size, theta_p, v)
    compute_d_mu_od(d_mu_od, P_od_rs, d_mu_sd, y_od_bar, n_size, r_nodes, o_nodes, d_nodes)

    # Compute MC
    MC = compute_MC(mu_od, y_od_bar, bigV_rs, z_rs_bar)

    # Compute gradient of MC
    grad_MC = np.zeros((r_size))
    compute_grad_MC(grad_MC, y_od_bar, d_mu_od, z_rs_1, r_nodes)

    return MC, grad_MC
