#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:liruihui
@file: tf_utils2.py 
@time: 2019/10/07
@contact: ruihuili.lee@gmail.com
@github: https://liruihui.github.io/
@description: 
"""
import tensorflow as tf
import numpy as np
# A shape is (N, P_A, C), B shape is (N, P_B, C)
# D shape is (N, P_A, P_B)
def batch_distance_matrix_general(A, B):
    r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
    r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
    m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
    D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
    return D


# A shape is (N, P_A, C), B shape is (N, P_B, C)
# D shape is (N, P_A, P_B)
def batch_cross_matrix_general(A, B):
    r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
    r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
    m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
    D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
    return D

# A shape is (N, P, C)
def find_duplicate_columns(A):
    N = A.shape[0]
    P = A.shape[1]
    indices_duplicated = np.ones((N, 1, P), dtype=np.int32)
    for idx in range(N):
        _, indices = np.unique(A[idx], return_index=True, axis=0)
        indices_duplicated[idx, :, indices] = 0
    return indices_duplicated

# add a big value to duplicate columns
def prepare_for_unique_top_k(D, A):
    indices_duplicated = tf.py_func(find_duplicate_columns, [A], tf.int32)
    D += tf.reduce_max(D)*tf.cast(indices_duplicated, tf.float32)

def knn_point(k, points, queries, sort=True, unique=True,use_cos=False,neg_dis=True):
    """
    points: dataset points (N, P0, K)
    queries: query points (N, P, K)
    return indices is (N, P, K, 2) used for tf.gather_nd(points, indices)
    distances (N, P, K)
    """
    with tf.name_scope("knn_point"):
        batch_size = tf.shape(queries)[0]
        point_num = tf.shape(queries)[1]
        if use_cos:
            D = batch_cross_matrix_general(queries, points)
        else:
            D = batch_distance_matrix_general(queries, points)
        if unique:
            prepare_for_unique_top_k(D, points)
        distances, point_indices = tf.nn.top_k(-D, k=k, sorted=sort)  # (N, P, K)
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, point_num, k, 1))
        indices = tf.concat([batch_indices, tf.expand_dims(point_indices, axis=3)], axis=3)
        if neg_dis:
            return -distances, indices
        else:
            return distances, indices


def get_spatial_edge_feature(xyz, features, k=20, idx=None):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k)
        k: int
    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """
    if idx is None:
        _, idx = knn_point(k+1, xyz, xyz, unique=True, sort=True)
        idx = idx[:, :, 1:, :]

    # [N, P, K, Dim]
    point_cloud_neighbors = tf.gather_nd(features, idx)
    point_cloud_central = tf.expand_dims(features, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    edge_feature = tf.concat(
        [point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
    return edge_feature, idx

def get_edge_feature(point_cloud, k=20, idx=None):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k, 2)
        k: int
    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """
    if idx is None:
        _, idx = knn_point(k+1, point_cloud, point_cloud, unique=True, sort=True)
        idx = idx[:, :, 1:, :]

    # [N, P, K, Dim]
    point_cloud_neighbors = tf.gather_nd(point_cloud, idx)
    point_cloud_central = tf.expand_dims(point_cloud, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    edge_feature = tf.concat(
        [point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
    return edge_feature, idx

def get_edge_feature_general(feat, new_feat, k=20, xyz=None,new_xyz=None, use_xyz=False):
    """Construct edge feature for each point
    Args:
        feat: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k, 2)
        k: int
    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """

    _, idx = knn_point(k + 1, feat, new_feat, unique=True, sort=True)
    idx = idx[:, :, 1:, :]

    # [N, P, K, Dim]
    feat_neighbors = tf.gather_nd(feat, idx)
    feat_central = tf.expand_dims(new_feat, axis=-2)

    feat_central = tf.tile(feat_central, [1, 1, k, 1])

    edge_feature = tf.concat(
        [feat_central, feat_neighbors - feat_central], axis=-1)

    if use_xyz:
        point_cloud_neighbors = tf.gather_nd(xyz, idx)
        point_cloud_central = tf.expand_dims(new_xyz, axis=-2)

        point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

        pc_feature = tf.concat([point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)

        edge_feature = tf.concat([pc_feature, edge_feature], axis=-1)

    return edge_feature