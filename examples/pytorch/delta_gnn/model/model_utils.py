import numpy as np
import torch as th
import time


def feature_scatter(dst_t, src_t, index):
    """
    Usage: Replace src_t -> dst_t (https://zhuanlan.zhihu.com/p/339043454)

    Args:
        src_t (Tensor): the source tensor
        dst_t (Tensor): the target tensor
        index (Numpy): the index of dst_t for scatter
    """

    src_index = np.array([[index[row]] * dst_t.shape[1] for row in range(src_t.shape[0])])
    src_index = th.from_numpy(src_index)
    src_index = src_index.to(src_t.device)
    dst_t_tmp = dst_t.scatter(0, src_index, src_t)

    return dst_t_tmp


def feature_scatter_add(dst_t, src_t, index):
    """
    Usage: Replace src_t -> dst_t + src_t (https://zhuanlan.zhihu.com/p/339043454)

    Args:
        src_t (Tensor): the source tensor
        dst_t (Tensor): the target tensor
        index (Numpy): the index of dst_t for scatter
    """

    src_index = np.array([[index[row]] * dst_t.shape[1] for row in range(src_t.shape[0])])
    src_index = th.from_numpy(src_index)
    src_index = src_index.to(src_t.device)
    dst_t_tmp = dst_t.scatter_add(0, src_index, src_t)

    return dst_t_tmp


def feature_merge(embedding, feat, feat_n_id, v_sen_feat_loc, v_insen_feat_loc, v_sen_id, v_insen_id):
    """ Update feat and model embedding
    Usage:
        embedding (2d Tensor): all the vertices feature of GNN model
        feat (2d Tensor): updated features in this round forward
        feat_n_id (Tensor): original vertex ID of each vertex feature in "feat"
        v_sen_feat_loc (Tensor): remapped ID (location) of "feat" from v_sen_id
        v_insen_feat_loc (Tensor): remapped ID (location) of "feat" from v_insen_id
        v_sen (Tensor): vertex ID of sensititive vertices
        v_insen (Tensor): vertex ID of insensitive vertices
    """

    # Compulsorily execute in CPU (GPU not suits for scalar execution)
    device = feat.device
    time_start = time.perf_counter()

    # feat = feat.to('cpu')
    # embedding = embedding.to('cpu')

    # load_time = time.perf_counter() - time_start
    # print('>> Load feat: {}'.format(util.time_format(load_time)))

    ##
    """ Workflow:
        1. Update the embedding with updated features
        2. Read out features from the updated embedding
    """

    feat_n_id = feat_n_id.cpu().squeeze().numpy()
    # # v_sen_id = np.array(list(v_sen_id))
    # v_insen_id = np.array(list(v_insen_id))
    # v_sen_id = np.setdiff1d(feat_n_id, v_insen_id)
    # # v_sen_id = np.intersect1d(feat_n_id, v_sen_id)
    # v_insen_id = np.intersect1d(feat_n_id, v_insen_id)

    # v_sen_feat_loc = np.zeros_like(v_sen_id)
    # v_insen_feat_loc = np.zeros_like(v_insen_id)
    # for i, v_sen in enumerate(v_sen_id):
    #     v_sen_feat_loc[i] = np.where(feat_n_id == v_sen)[0][0]
    # for i, v_insen in enumerate(v_insen_id):
    #     v_insen_feat_loc[i] = np.where(feat_n_id == v_insen)[0][0]

    # # v_sen_feat_loc = th.Tensor(v_sen_feat_loc).long()
    # # v_insen_feat_loc = th.Tensor(v_insen_feat_loc).long()
    # v_sen_feat_loc = th.Tensor(v_sen_feat_loc).long().to(device)
    # v_insen_feat_loc = th.Tensor(v_insen_feat_loc).long().to(device)

    feat_sen = feat.index_select(0, v_sen_feat_loc)
    if len(v_insen_feat_loc) != 0:
        feat_insen = feat.index_select(0, v_insen_feat_loc)

    embedding = embedding.to(device)
    embedding_update_tmp = feature_scatter(embedding, feat_sen, v_sen_id)
    if len(v_insen_feat_loc) != 0:
        embedding_update = feature_scatter_add(embedding_update_tmp, feat_insen, v_insen_id)
    else:
        embedding_update = embedding_update_tmp

    feat_n_id = th.from_numpy(feat_n_id).to(device)
    if len(v_insen_feat_loc) != 0:
        feat_update = embedding_update.index_select(0, feat_n_id)
    else:
        feat_update = feat

    # feat_update = feat_update.to(device)
    # embedding_update = embedding_update.to(device)

    return feat_update, embedding_update


# def feature_merge(embedding, feat, feat_n_id, v_sen_id, v_insen_id):
#     """ Update feat and model embedding
#     Usage:
#         embedding (2d Tensor): all the vertices feature of GNN model
#         feat (2d Tensor): updated features in this round forward
#         feat_n_id (Tensor): original vertex ID of each vertex feature in "feat"
#         v_sen (Tensor): vertex ID of sensititive vertices
#         v_insen (Tensor): vertex ID of insensitive vertices
#     """

#     # Compulsorily execute in CPU (GPU not suits for scalar execution)
#     device = feat.device
#     time_start = time.perf_counter()

#     # feat = feat.to('cpu')
#     # embedding = embedding.to('cpu')

#     # load_time = time.perf_counter() - time_start
#     # print('>> Load feat: {}'.format(util.time_format(load_time)))

#     ##
#     """ Workflow:
#         1. Update the embedding with updated features
#         2. Read out features from the updated embedding
#     """

#     feat_n_id = feat_n_id.cpu().squeeze().numpy()
#     # v_sen_id = np.array(list(v_sen_id))
#     v_insen_id = np.array(list(v_insen_id))
#     v_sen_id = np.setdiff1d(feat_n_id, v_insen_id)
#     # v_sen_id = np.intersect1d(feat_n_id, v_sen_id)
#     v_insen_id = np.intersect1d(feat_n_id, v_insen_id)

#     v_sen_feat_loc = np.zeros_like(v_sen_id)
#     v_insen_feat_loc = np.zeros_like(v_insen_id)
#     for i, v_sen in enumerate(v_sen_id):
#         v_sen_feat_loc[i] = np.where(feat_n_id == v_sen)[0][0]
#     for i, v_insen in enumerate(v_insen_id):
#         v_insen_feat_loc[i] = np.where(feat_n_id == v_insen)[0][0]

#     # v_sen_feat_loc = th.Tensor(v_sen_feat_loc).long()
#     # v_insen_feat_loc = th.Tensor(v_insen_feat_loc).long()
#     v_sen_feat_loc = th.Tensor(v_sen_feat_loc).long().to(device)
#     v_insen_feat_loc = th.Tensor(v_insen_feat_loc).long().to(device)
#     feat_sen = feat.index_select(0, v_sen_feat_loc)
#     if len(v_insen_feat_loc) != 0:
#         feat_insen = feat.index_select(0, v_insen_feat_loc)

#     embedding = embedding.to(device)
#     embedding_update_tmp = feature_scatter(embedding, feat_sen, v_sen_id)
#     if len(v_insen_feat_loc) != 0:
#         embedding_update = feature_scatter_add(embedding_update_tmp, feat_insen, v_insen_id)
#     else:
#         embedding_update = embedding_update_tmp

#     feat_n_id = th.from_numpy(feat_n_id).to(device)
#     if len(v_insen_feat_loc) != 0:
#         feat_update = embedding_update.index_select(0, feat_n_id)
#     else:
#         feat_update = feat

#     # feat_update = feat_update.to(device)
#     # embedding_update = embedding_update.to(device)

#     return feat_update, embedding_update


# def feature_merge(embedding, feat, feat_n_id, v_sen_id, v_insen_id):
#     """ Update feat and model embedding
#     Usage:
#         embedding (2d Tensor): all the vertices feature of GNN model
#         feat (2d Tensor): updated features in this round forward
#         feat_n_id (Tensor): original vertex ID of each vertex feature in "feat"
#         v_sen (Tensor): vertex ID of sensititive vertices
#         v_insen (Tensor): vertex ID of insensitive vertices
#     """

#     # Compulsorily execute in CPU (GPU not suits for scalar execution)
#     device = feat.device
#     time_start = time.perf_counter()
#     feat = feat.to('cpu')
#     embedding = embedding.to('cpu')
#     # load_time = time.perf_counter() - time_start
#     # print('>> Load feat: {}'.format(util.time_format(load_time)))

#     ##
#     """ Workflow:
#         1. Update the embedding with updated features
#         2. Read out features from the updated embedding
#     """

#     feat_n_id = feat_n_id.cpu().squeeze().numpy()
#     # v_sen_id = np.array(list(v_sen_id))
#     v_insen_id = np.array(list(v_insen_id))
#     v_sen_id = np.setdiff1d(feat_n_id, v_insen_id)
#     # v_sen_id = np.intersect1d(feat_n_id, v_sen_id)
#     v_insen_id = np.intersect1d(feat_n_id, v_insen_id)

#     v_sen_feat_loc = np.zeros_like(v_sen_id)
#     v_insen_feat_loc = np.zeros_like(v_insen_id)
#     for i, v_sen in enumerate(v_sen_id):
#         v_sen_feat_loc[i] = np.where(feat_n_id == v_sen)[0][0]
#     for i, v_insen in enumerate(v_insen_id):
#         v_insen_feat_loc[i] = np.where(feat_n_id == v_insen)[0][0]

#     v_sen_feat_loc = th.Tensor(v_sen_feat_loc).long()
#     v_insen_feat_loc = th.Tensor(v_insen_feat_loc).long()
#     feat_sen = feat.index_select(0, v_sen_feat_loc)
#     if len(v_insen_feat_loc) != 0:
#         feat_insen = feat.index_select(0, v_insen_feat_loc)

#     embedding_update_tmp = feature_scatter(embedding, feat_sen, v_sen_id)
#     if len(v_insen_feat_loc) != 0:
#         embedding_update = feature_scatter_add(embedding_update_tmp, feat_insen, v_insen_id)
#     else:
#         embedding_update = embedding_update_tmp

#     feat_n_id = th.from_numpy(feat_n_id)
#     if len(v_insen_feat_loc) != 0:
#         feat_update = embedding_update.index_select(0, feat_n_id)
#     else:
#         feat_update = feat

#     feat_update = feat_update.to(device)
#     embedding_update = embedding_update.to(device)

#     return feat_update, embedding_update


def store_embedding(embedding, feat, ind):
    device = feat.device
    ind = ind.squeeze().cpu().numpy().tolist()
    feat_t = feat.cpu()
    embedding = embedding.cpu()

    # feat_index = [ind[row] for col in range(embedding.shape[1])] for row in range(feat.shape[0])]
    feat_index = np.array([[ind[row]] * embedding.shape[1] for row in range(feat_t.shape[0])])
    feat_index = th.from_numpy(feat_index)
    embedding = embedding.scatter(0, feat_index, feat_t)

    return embedding.to(device)