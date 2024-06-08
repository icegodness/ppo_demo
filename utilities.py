import numpy as np
import random
import torch



def calculate_distance(node, base_station):
    return np.linalg.norm(node - base_station)


def calculate_angle(node, base_station):
    return np.degrees(np.arctan2(node[1] - base_station[1], node[0] - base_station[0])) % 360


def calculate_received_power(Ptx_BS_dBm, distance, pathloss, alpha, Gt_vec, Gr_vec):
    return Ptx_BS_dBm * Gt_vec * Gr_vec * pathloss * ( distance ** (- alpha) )


def calculate_snr(N0, received_power):
    # return 10 ** ((received_power - N0) / 10)
    return received_power - N0


def calculate_data_rate(B, snr):
    return B * np.log10(1 + snr)


def find_min_snr_node(nodes, base_station, direction, beamwidth, distance, alpha, Gr_vec, N0, B, z, pathloss, Ptx_BS_dBm
                      ):
    start_angle = direction - beamwidth / 2
    end_angle = direction + beamwidth / 2

    Gt_vec = (2 * np.pi - (2 * np.pi - beamwidth * np.pi / 180) * z) / (
                beamwidth * np.pi / 180)  # Transmitting antenna gain

    min_snr = float('inf')
    min_snr_node = None
    covered = []
    for i, node in zip(range(len(nodes)), nodes):
        node_angle = calculate_angle(node, base_station)
        node_distance = calculate_distance(node, base_station)
        if start_angle <= node_angle <= end_angle and node_distance <= distance:
            received_power = calculate_received_power(Ptx_BS_dBm, node_distance, pathloss, alpha, Gt_vec, Gr_vec)
            covered.append(i)
            received_power = 10 * np.log10(received_power)
            snr = calculate_snr(N0, received_power)
            if snr < min_snr:
                min_snr = snr
                min_snr_node = i

    if min_snr_node is not None:
        data_rate = calculate_data_rate(B, min_snr)
        # print("Node with minimum SNR:", min_snr_node)
        # print("Minimum SNR:", min_snr)
        # print("Data rate:", data_rate)
        print("Covered nodes:", covered)

        return min_snr_node, min_snr, covered, data_rate
    else:
        print("No node found within the beam boundaries.")
        return None, None, None, None
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False