import numpy as np


def load_data(file_path):
    cites = np.loadtxt(file_path, delimiter='\t')
    return cites


def find_root_node(node_list, parent_csr):
    root_nodes = []
    for node in node_list:
        if node not in parent_csr:
            root_nodes.append(node)

    return root_nodes


# def gen_csr(node_list, child_csr):
#     indptr = []
#     indices = []

#     for node in node_list:
#         indptr.append(len(indices))
#         if node in child_csr:
#             indices.extend(child_csr[node])

#     for parent in child_csr:
#         indptr.append(len(indices))


def bfs_traverse(root_nodes, child_csr):
    queue = []
    seen = set()
    sequence = list()
    root_node_q = root_nodes.copy()
    while len(root_node_q) > 0:
        root = root_node_q.pop(0)
        queue.append(root)
        seen.add(root)
        sequence.append(root)
        while len(queue) > 0:
            vetex = queue.pop(0)
            if vetex in child_csr:
                nodes = child_csr[vetex]
            else:
                nodes = []
            # print(nodes)
            for w in nodes:
                # print(w)
                if w not in seen:
                    queue.append(w)
                    seen.add(w)
                    sequence.append(w)
                # print(len(seen), len(queue))

        # Pop nodes in root_node_q which have been visited
        while len(root_node_q) > 0 and root_node_q[0] in seen:
            root_node_q.pop(0)
        # print('>> seen size: ', len(seen), 'seq size: ', len(sequence))

    return sequence


def get_ngh(nodes, parent_csr, child_csr, node_evo_q):
    nghs = []
    for node_id in nodes:
        ngh_list = []
        if node_id in parent_csr:
            ngh_list.extend(parent_csr[node_id])
        if node_id in child_csr:
            ngh_list.extend(child_csr[node_id])

        for ngh in ngh_list:
            if ngh in node_evo_q:
                nghs.append(ngh)
    return nghs


def count_ngh(parent_csr, child_csr, node_q, snapshot_num, new_node_num_per_snapshot, ngh_hop_num):
    ngh_num_list = []
    node_evo_q = set()  # Record the nodes which have existed in evolved snapshot
    for snapshot_id in range(snapshot_num):
        new_nodes = node_q[snapshot_id * new_node_num_per_snapshot:min((snapshot_id + 1) *
                                                                       new_node_num_per_snapshot,
                                                                       len(node_q) - 1)]
        node_evo_q.update(new_nodes)
        # nghs_per_snapshot = []
        nghs_per_snapshot = set()
        ngh_q = new_nodes
        for hop in range(ngh_hop_num):
            ngh_q = get_ngh(ngh_q, parent_csr, child_csr, node_evo_q)
            # nghs_per_snapshot.extend(ngh_q)
            nghs_per_snapshot.update(ngh_q)
            print("hop: ", hop, " Ngh_num: ", len(ngh_q))
        ngh_num_list.append(len(nghs_per_snapshot))
    return ngh_num_list


def count_affected_node_ratio():
    """
    Count the affected nodes ratio with graph evolving
    """
    cites = load_data('../../../dataset/cora_src/cora_cites.txt')

    node_set = set()
    child_csr = dict()  # Record the children of each node
    parent_csr = dict()  # Record the parents of each node
    [row, col] = cites.shape
    for row_id in range(row):
        parent = int(cites[row_id][0])  # Cited paper
        child = int(cites[row_id][1])  # Citing paper

        if child in parent_csr:
            parent_csr[child].append(parent)
        else:
            parent_csr[child] = [parent]

        if parent in child_csr:
            child_csr[parent].append(child)
        else:
            child_csr[parent] = [child]

        node_set.add(child)
        node_set.add(parent)

    root_nodes = find_root_node(node_set, parent_csr)
    init_node_q = root_nodes + list(node_set)
    node_q = bfs_traverse(init_node_q, child_csr)

    snapshot_num = 8
    ngh_hop_num = 4  # Represent k-hop nghs
    new_node_num_per_snapshot = round(len(node_q) / snapshot_num)
    ngh_num_list = count_ngh(parent_csr, child_csr, node_q, snapshot_num, new_node_num_per_snapshot,
                             ngh_hop_num)
    print(ngh_num_list)

    node_num_snapshot = [
        min((i + 1) * new_node_num_per_snapshot, len(node_q)) for i in range(snapshot_num)
    ]

    percentage = [round(ngh_num_list[i] / node_num_snapshot[i], 2) for i in range(snapshot_num)]
    print(percentage)


count_affected_node_ratio()