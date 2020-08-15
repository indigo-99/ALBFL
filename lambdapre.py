import numpy as np
import math

def CalcDCG(labels):
    sumdcg = 0.0
    for i in range(len(labels)):
        rel = labels[i]
        if rel != 0:
            sumdcg += ((2 ** rel) - 1) / math.log2(i + 2)
    return sumdcg

def fetch_qid_data(y, qid, eval_at=None):
    """Fetch indices, relevances, idcg and dcg for each query id.
    Parameters
    ----------
    y : array, shape (n_samples,)
        Target labels.
    qid: array, shape (n_samples,)
        Query id that represents the grouping of samples.
    eval_at: integer
        The rank postion to evaluate dcg and idcg. ndcg@n<- 1/3/5/10
    Returns
    -------
    qid2indices : array, shape (n_unique_qid,)
        Start index for each qid.
    qid2rel : array, shape (n_unique_qid,)
        A list of target labels (relevances) for each qid.
    qid2idcg: array, shape (n_unique_qid,)
        Calculated idcg@eval_at for each qid.
    """
    qid_unique, qid2indices, qid_inverse_indices = np.unique(qid, return_index=True, return_inverse=True)
    # get item belong to each query id
    qid2rel = [[] for _ in range(len(qid_unique))]
    for i, qid_unique_index in enumerate(qid_inverse_indices):
        qid2rel[qid_unique_index].append(y[i])
    # get dcg, idcg for each query id @eval_at
    if eval_at:
        qid2idcg = [CalcDCG(sorted(qid2rel[i], reverse=True)[:eval_at]) for i in range(len(qid_unique))]
    else:
        qid2idcg = [CalcDCG(sorted(qid2rel[i], reverse=True)) for i in range(len(qid_unique))]
    return qid2indices, qid2rel, qid2idcg

def transform_pairwise(X, y, qid):
    """Transform data into lambdarank pairs with balanced labels for binary classification.
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Features.
    y : array, shape (n_samples,)
        Target labels.
    qid: array, shape (n_samples,)
        Query id that represents the grouping of samples.
    Returns
    -------
    X1_trans : array, shape (k, n_feaures)
        Features of pair 1
    X2_trans : array, shape (k, n_feaures)
        Features of pair 2
    weight: array, shape (k, n_faetures)
        Sample weight lambda.
    y_trans : array, shape (k,)
        Output class labels, where classes have values {0, 1}
    """
    qid2indices, qid2rel, qid2idcg= fetch_qid_data(y, qid)
    X1 = []
    X2 = []
    weight = []
    Y = []
    for qid_unique_idx in range(len(qid2indices)):
        if qid2idcg[qid_unique_idx] == 0:
            continue
        IDCG = 1.0 / qid2idcg[qid_unique_idx]
        rel_list = qid2rel[qid_unique_idx]
        qid_start_idx = qid2indices[qid_unique_idx]
        for pos_idx in range(len(rel_list)):  # fault
            for neg_idx in range(len(rel_list)):  # right
                if rel_list[pos_idx] <= rel_list[neg_idx]:
                    continue
                pos_loginv = 1.0 / math.log2(pos_idx + 2)
                neg_loginv = 1.0 / math.log2(neg_idx + 2)
                pos_label = rel_list[pos_idx]
                neg_label = rel_list[neg_idx]
                # before change sequence of the pos_neg pair, the pair's ndcg score
                original = ((1 << pos_label) - 1) * pos_loginv + ((1 << neg_label) - 1) * neg_loginv
                # after change sequence, the pair's ndcg score
                # delta indeed equals: fault sentence in raw pos'ndcg - in right sentence's pos'ndcg
                changed = ((1 << neg_label) - 1) * pos_loginv + ((1 << pos_label) - 1) * neg_loginv
                # the impact of change sequence (wrong prediction)
                delta = (original - changed) * IDCG
                if delta < 0:
                    delta = -delta # weight should > 0, do abs()
                # balanced class
                if 1 != (-1) ** (qid_unique_idx + pos_idx + neg_idx):
                    X1.append(X[qid_start_idx + pos_idx])
                    X2.append(X[qid_start_idx + neg_idx])
                    weight.append(delta) # weight no more like ranknet to be 1
                    # if this pair to be predicted wrong(1->0 or in turn),
                    # the impact will be more serious, so the sample weight is relatively bigger
                    Y.append(1)
                else:
                    X1.append(X[qid_start_idx + neg_idx])
                    X2.append(X[qid_start_idx + pos_idx])
                    weight.append(delta)
                    Y.append(0)
    return np.asarray(X1), np.asarray(X2), np.asarray(Y), np.asarray(weight)
