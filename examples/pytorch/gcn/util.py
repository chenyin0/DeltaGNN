from collections import Counter
# from msilib import sequence
import torch as th
import dgl


def gen_root_node_queue(g):
    """
    Gen root node according to in-degree
    """
    src_edges = g.edges()[0].numpy().tolist()
    nodes = Counter(src_edges).most_common(len(src_edges))
    nodes = [x[0] for x in nodes]
    return nodes


def bfs_traverse(g_csr, root_node_q):
    """
    Gen node evolve sequence
    """

    # g_csr = g.adj_sparse('csr')
    indptr = g_csr[0]
    indices = g_csr[1]

    queue = []
    seen = set()
    sequence = list()
    while len(root_node_q) > 0:
        root = root_node_q.pop(0)
        queue.append(root)
        seen.add(root)
        sequence.append(root)
        while len(queue) > 0:
            vetex = queue.pop(0)
            begin = indptr[root]
            end = indptr[root + 1]
            nodes = indices[begin:end].tolist()
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
        print('>> seen size: ', len(seen), 'seq size: ', len(sequence))

    return sequence


def graph_evolve(new_nodes, g_orig_csr, g_orig, g_evo=None):
    """
    Construct evolve graph from an orginal static graph
    """

    indptr = g_orig_csr[0]
    indices = g_orig_csr[1]

    edge_src_nodes = list()
    edge_dst_nodes = list()
    for node in new_nodes:
        # e_dest_nodes = indices[indptr[node]:indptr[node + 1]]
        # e_src_nodes = th.linspace(node, node, edge_dest_nodes.size()[0])
        e_src_nodes = indices[indptr[node]:indptr[node + 1]]
        e_dst_nodes = th.linspace(node, node, len(e_src_nodes))
        edge_dst_nodes.extend(e_src_nodes.numpy().tolist())
        edge_src_nodes.extend(e_dst_nodes.numpy().tolist())

    edge_dst_nodes = th.tensor(edge_dst_nodes)
    edge_src_nodes = th.tensor(edge_src_nodes)

    edge_src_nodes = th.ones(edge_dst_nodes.size()[0])

    if g_evo is None:
        # Construct a new graph
        g_evo = dgl.graph((edge_src_nodes, edge_dst_nodes))
    else:
        g_evo.add_edges(th.tensor(edge_src_nodes), th.tensor(edge_dst_nodes))

    features_evo, labels_evo, train_mask_evo, val_mask_evo, test_mask_evo = update_g_evo(g_evo, g_orig, new_nodes)

    return features_evo, labels_evo, train_mask_evo, val_mask_evo, test_mask_evo


def update_g_evo(g_evo, g_orig, new_nodes):
    """
    Update feature and train/eval/test mask
    """
