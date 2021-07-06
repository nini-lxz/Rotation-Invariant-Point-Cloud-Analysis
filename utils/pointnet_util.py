""" PointNet++ Layers

Author: Charles R. Qi
Date: November 2017
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'grouping'))
sys.path.append(os.path.join(ROOT_DIR, '3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util_pointnet2
import math

def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=False):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=False):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util_pointnet2.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format)
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util_pointnet2.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format)
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx


def prepare_rotinv_input_radius_tetrahedron_angle_new(new_xyz, radius, nsample, grouped_xyz):
    # new_xyz: [B, npoint, 3]
    # nsample: neighbor size
    # grouped_xyz: [B, npoint, nsample, 3]

    batch_size = new_xyz.get_shape()[0].value
    point_num = new_xyz.get_shape()[1].value

    # calculate neighbor centroid
    centroid_xyz = tf.reduce_mean(grouped_xyz, axis=2)  # [B, N, 3]

    # calculate intersection point
    reference_vector_norm = tf.norm(new_xyz, axis=-1, keepdims=True)  # [B, N, 1]
    reference_vector_unit = new_xyz / (reference_vector_norm + 0.0001)  # [B, N, 3]
    inter_xyz = radius * reference_vector_unit + new_xyz

    # prepare features of center point
    centroid_reference_vector = new_xyz - centroid_xyz
    centroid_reference_dist = tf.norm(centroid_reference_vector, axis=-1, keepdims=True)  # [B, N, 1]

    centroid_inter_vector = inter_xyz - centroid_xyz
    centroid_inter_dist = tf.norm(centroid_inter_vector, axis=-1, keepdims=True)  # [B, N, 1]

    dot_product = tf.reduce_sum(tf.multiply(centroid_reference_vector, centroid_inter_vector), axis=-1,
                                keepdims=True)
    reference_centroid_inter_angle = dot_product / (centroid_reference_dist * centroid_inter_dist + 0.000001)

    inter_reference_vector = new_xyz - inter_xyz
    inter_centroid_vector = centroid_xyz - inter_xyz
    dot_product = tf.reduce_sum(tf.multiply(inter_reference_vector, inter_centroid_vector), axis=-1,
                                keepdims=True)
    reference_inter_centroid_angle = dot_product / (radius * centroid_inter_dist + 0.000001)

    center_point_features = tf.concat([reference_vector_norm, centroid_reference_dist, centroid_inter_dist,
                                       reference_centroid_inter_angle, reference_inter_centroid_angle],
                                      axis=-1)  # [B, N, 5]
    center_point_features_tile = tf.tile(tf.expand_dims(center_point_features, axis=2),
                                         [1, 1, nsample, 1])  # [B, N, K, 5]

    # prepare features of neighbor points
    centroid_xyz_tile = tf.tile(tf.expand_dims(centroid_xyz, axis=2), [1, 1, nsample, 1])
    neighbor_centroid_vector = centroid_xyz_tile - grouped_xyz
    reference_vector_tile = tf.tile(tf.expand_dims(new_xyz, axis=2), [1, 1, nsample, 1])
    neighbor_reference_vector = reference_vector_tile - grouped_xyz
    inter_pts = tf.tile(tf.expand_dims(inter_xyz, axis=2), [1, 1, nsample, 1])
    neighbor_inter_vector = inter_pts - grouped_xyz

    neighbor_centroid_dist = tf.norm(neighbor_centroid_vector, axis=-1, keepdims=True)
    neighbor_reference_dist = tf.norm(neighbor_reference_vector, axis=-1, keepdims=True)
    neighbor_inter_dist = tf.norm(neighbor_inter_vector, axis=-1, keepdims=True)

    dot_product = tf.reduce_sum(tf.multiply(neighbor_centroid_vector, neighbor_reference_vector), axis=-1,
                                keepdims=True)
    centroid_neighbor_reference_angle = dot_product / (neighbor_centroid_dist *
                                                       neighbor_reference_dist + 0.000001)

    dot_product = tf.reduce_sum(tf.multiply(neighbor_reference_vector, neighbor_inter_vector), axis=-1,
                                keepdims=True)
    reference_neighbor_inter_angle = dot_product / (neighbor_reference_dist *
                                                    neighbor_inter_dist + 0.000001)

    dot_product = tf.reduce_sum(tf.multiply(neighbor_inter_vector, neighbor_centroid_vector), axis=-1,
                                keepdims=True)
    inter_neighbor_centroid_angle = dot_product / (neighbor_inter_dist *
                                                   neighbor_centroid_dist + 0.000001)

    #################### calculate angle ####################
    reference_plane_params = get_plane_equation(inter_pts, reference_vector_tile, centroid_xyz_tile)  # [B, N, K, 4]
    reference_normal_vector = reference_plane_params[:, :, :, 0:3]
    # reference_normal_length = tf.norm(reference_normal_vector, axis=-1, keepdims=True)  # [B, N, K, 1]

    neighbor_plane_params = get_plane_equation(inter_pts, reference_vector_tile, grouped_xyz)
    neighbor_normal_vector = neighbor_plane_params[:, :, :, 0:3]
    # neighbor_normal_length = tf.norm(neighbor_normal_vector, axis=-1, keepdims=True)  # [B, N, K, 1]

    dot_product = tf.reduce_sum(tf.multiply(reference_normal_vector, neighbor_normal_vector), axis=-1,
                                keepdims=True)
    # cos_plane_angle = dot_product / (reference_normal_length * neighbor_normal_length + 0.000001)
    cos_plane_angle = dot_product
    plane_angle = tf.acos(cos_plane_angle)  # [B, N, K, 1]  in [0, pi]

    pos_state = tf.reduce_sum(tf.multiply(reference_normal_vector, -neighbor_reference_vector), axis=-1,
                                keepdims=True)
    pos_state = tf.sign(pos_state)   # [B, N, K, 1]
    plane_angle_direction = plane_angle * pos_state  # [0, pi)
    # # sin_plane_angle = tf.sin(plane_angle_direction)
    #
    angle = tf.cos(0.25*plane_angle_direction) - tf.sin(0.25*plane_angle_direction) - 0.75
    ###############################################################

    neighbor_point_features = tf.concat([neighbor_centroid_dist, neighbor_reference_dist, neighbor_inter_dist,
                                         centroid_neighbor_reference_angle, reference_neighbor_inter_angle,
                                         inter_neighbor_centroid_angle, angle], axis=-1)  # [B, N, K, 6]

    rotinv_features = tf.concat([center_point_features_tile, neighbor_point_features], axis=-1)

    return center_point_features_tile, neighbor_point_features, rotinv_features

def get_plane_equation(p1, p2, p3):
    # p1 (B, N, K, 3)
    # return: (B, N, K, 4)
    p1p2 = p2-p1
    p2p3 = p3-p2
    normal = tf.cross(p1p2, p2p3)
    normal_length = tf.norm(normal, axis=-1, keepdims=True)  # [B, N, K, 1]
    normal = normal / (normal_length + 0.00001)
    plane_a = normal[:, :, :, 0]
    plane_b = normal[:, :, :, 1]
    plane_c = normal[:, :, :, 2]

    x1 = p1[:, :, :, 0]
    y1 = p1[:, :, :, 1]
    z1 = p1[:, :, :, 2]
    plane_d = -1.0*(plane_a*x1+plane_b*y1+plane_c*z1)

    plane_params = tf.concat([tf.expand_dims(plane_a, axis=-1),
                              tf.expand_dims(plane_b, axis=-1),
                              tf.expand_dims(plane_c, axis=-1),
                              tf.expand_dims(plane_d, axis=-1)], axis=-1)
    return plane_params

def get_robust_centroid(input):
    # input: [B, N, K, 3]
    batch_size = input.get_shape()[0].value
    point_num = input.get_shape()[1].value
    neighbor_size = input.get_shape()[2].value
    subset_size = tf.cast(tf.round(neighbor_size * 0.9), tf.int32)  # select 90% of K

    centroid_list = []
    sample_num = 10  #10  #5
    sample_num_selected = tf.cast(tf.round(sample_num * 0.7), tf.int32)
    for i in range(sample_num):
        # subset = tf.py_func(generate_subset, [input, neighbor_size, subset_size], tf.float32)  # [B, N, K*0.9, 3]
        input_transpose = tf.transpose(input, perm=[2, 1, 0, 3])  # [K, N, B, 3]
        input_transpose_shuffle = tf.random_shuffle(input_transpose)
        subset_transpose = input_transpose_shuffle[0:subset_size, ...]
        subset = tf.transpose(subset_transpose, perm=[2, 1, 0, 3])

        centroid = tf.reduce_mean(subset, axis=2, keepdims=True)  # [B, N, 1, 3]
        centroid_list.append(centroid)
    centroid_list = tf.concat(centroid_list, axis=2)  # [B, N, 5, 3]

    centroid_list_transpose = tf.transpose(centroid_list, perm=[0, 1, 3, 2])  # [B, N, 3, 5]
    inner = tf.matmul(centroid_list, centroid_list_transpose)  # [B, N, 5, 5]

    centroid_list_square = tf.reduce_sum(tf.square(centroid_list), axis=-1, keepdims=True)  # [B, N, 5, 1]
    centroid_list_square_transpose = tf.transpose(centroid_list_square, perm=[0, 1, 3, 2])  # [B, N, 1, 5]

    pairwise_dist = centroid_list_square + centroid_list_square_transpose - 2 * inner  # [B, N, 5, 5]
    pairwise_dist_sum = tf.reduce_sum(pairwise_dist, axis=-1)  # [B, N, 5]

    sorted_indices = tf.nn.top_k(pairwise_dist_sum, sample_num).indices  # sorted from largest to smallest
    sorted_indices_selected = sorted_indices[:, :, sample_num_selected:]  # [B, N, P]

    # group
    flattened_centroid = tf.reshape(centroid_list, [-1, 3])
    indices_reshape = tf.reshape(sorted_indices_selected, [batch_size*point_num, -1])
    offset = tf.expand_dims(tf.range(0, batch_size*point_num) * sample_num, 1)
    flattened_indices = tf.reshape(indices_reshape + offset, [-1])
    selected_rows = tf.gather(flattened_centroid, flattened_indices)
    centroid_selected = tf.reshape(selected_rows, [batch_size, point_num, -1, 3])  # [B, N, P, 3]

    centroid_avg = tf.reduce_mean(centroid_selected, axis=2)  # [B, N, 3]

    return centroid_avg

def prepare_rotinv_input_radius_tetrahedron_angle_new_robust(new_xyz, radius, nsample, grouped_xyz):
    # new_xyz: [B, npoint, 3]
    # nsample: neighbor size
    # grouped_xyz: [B, npoint, nsample, 3]

    batch_size = new_xyz.get_shape()[0].value
    point_num = new_xyz.get_shape()[1].value

    # calculate neighbor centroid
    #centroid_xyz = tf.reduce_mean(grouped_xyz, axis=2)  # [B, N, 3]
    centroid_xyz = get_robust_centroid(grouped_xyz)  # [B, N, 3]

    # calculate intersection point
    reference_vector_norm = tf.norm(new_xyz, axis=-1, keepdims=True)  # [B, N, 1]
    reference_vector_unit = new_xyz / (reference_vector_norm + 0.0001)  # [B, N, 3]
    inter_xyz = radius * reference_vector_unit + new_xyz

    # prepare features of center point
    centroid_reference_vector = new_xyz - centroid_xyz
    centroid_reference_dist = tf.norm(centroid_reference_vector, axis=-1, keepdims=True)  # [B, N, 1]

    centroid_inter_vector = inter_xyz - centroid_xyz
    centroid_inter_dist = tf.norm(centroid_inter_vector, axis=-1, keepdims=True)  # [B, N, 1]

    dot_product = tf.reduce_sum(tf.multiply(centroid_reference_vector, centroid_inter_vector), axis=-1,
                                keepdims=True)
    reference_centroid_inter_angle = dot_product / (centroid_reference_dist * centroid_inter_dist + 0.000001)

    inter_reference_vector = new_xyz - inter_xyz
    inter_centroid_vector = centroid_xyz - inter_xyz
    dot_product = tf.reduce_sum(tf.multiply(inter_reference_vector, inter_centroid_vector), axis=-1,
                                keepdims=True)
    reference_inter_centroid_angle = dot_product / (radius * centroid_inter_dist + 0.000001)

    center_point_features = tf.concat([reference_vector_norm, centroid_reference_dist, centroid_inter_dist,
                                       reference_centroid_inter_angle, reference_inter_centroid_angle],
                                      axis=-1)  # [B, N, 5]
    #center_point_features = tf.concat([reference_vector_norm, centroid_reference_dist, centroid_inter_dist],
    #                                  axis=-1)  # [B, N, 5]
    center_point_features_tile = tf.tile(tf.expand_dims(center_point_features, axis=2),
                                         [1, 1, nsample, 1])  # [B, N, K, 5]

    # prepare features of neighbor points
    centroid_xyz_tile = tf.tile(tf.expand_dims(centroid_xyz, axis=2), [1, 1, nsample, 1])
    neighbor_centroid_vector = centroid_xyz_tile - grouped_xyz
    reference_vector_tile = tf.tile(tf.expand_dims(new_xyz, axis=2), [1, 1, nsample, 1])
    neighbor_reference_vector = reference_vector_tile - grouped_xyz
    inter_pts = tf.tile(tf.expand_dims(inter_xyz, axis=2), [1, 1, nsample, 1])
    neighbor_inter_vector = inter_pts - grouped_xyz

    neighbor_centroid_dist = tf.norm(neighbor_centroid_vector, axis=-1, keepdims=True)
    neighbor_reference_dist = tf.norm(neighbor_reference_vector, axis=-1, keepdims=True)
    neighbor_inter_dist = tf.norm(neighbor_inter_vector, axis=-1, keepdims=True)

    dot_product = tf.reduce_sum(tf.multiply(neighbor_centroid_vector, neighbor_reference_vector), axis=-1,
                                keepdims=True)
    centroid_neighbor_reference_angle = dot_product / (neighbor_centroid_dist *
                                                       neighbor_reference_dist + 0.000001)

    dot_product = tf.reduce_sum(tf.multiply(neighbor_reference_vector, neighbor_inter_vector), axis=-1,
                                keepdims=True)
    reference_neighbor_inter_angle = dot_product / (neighbor_reference_dist *
                                                    neighbor_inter_dist + 0.000001)

    dot_product = tf.reduce_sum(tf.multiply(neighbor_inter_vector, neighbor_centroid_vector), axis=-1,
                                keepdims=True)
    inter_neighbor_centroid_angle = dot_product / (neighbor_inter_dist *
                                                   neighbor_centroid_dist + 0.000001)

    #################### calculate angle ####################
    reference_plane_params = get_plane_equation(inter_pts, reference_vector_tile, centroid_xyz_tile)  # [B, N, K, 4]
    reference_normal_vector = reference_plane_params[:, :, :, 0:3]
    # reference_normal_length = tf.norm(reference_normal_vector, axis=-1, keepdims=True)  # [B, N, K, 1]

    neighbor_plane_params = get_plane_equation(inter_pts, reference_vector_tile, grouped_xyz)
    neighbor_normal_vector = neighbor_plane_params[:, :, :, 0:3]
    # neighbor_normal_length = tf.norm(neighbor_normal_vector, axis=-1, keepdims=True)  # [B, N, K, 1]

    dot_product = tf.reduce_sum(tf.multiply(reference_normal_vector, neighbor_normal_vector), axis=-1,
                                keepdims=True)
    # cos_plane_angle = dot_product / (reference_normal_length * neighbor_normal_length + 0.000001)
    cos_plane_angle = dot_product
    plane_angle = tf.acos(cos_plane_angle)  # [B, N, K, 1]  in [0, pi]

    pos_state = tf.reduce_sum(tf.multiply(reference_normal_vector, -neighbor_reference_vector), axis=-1,
                                keepdims=True)
    pos_state = tf.sign(pos_state)   # [B, N, K, 1]
    plane_angle_direction = plane_angle * pos_state  # [0, pi)
    # # sin_plane_angle = tf.sin(plane_angle_direction)
    #
    angle = tf.cos(0.25*plane_angle_direction) - tf.sin(0.25*plane_angle_direction) - 0.75
    # angle = plane_angle_direction/math.pi
    # angle = tf.sin(plane_angle_direction/2.0)
    ###############################################################

    neighbor_point_features = tf.concat([neighbor_centroid_dist, neighbor_reference_dist, neighbor_inter_dist,
                                         centroid_neighbor_reference_angle, reference_neighbor_inter_angle,
                                         inter_neighbor_centroid_angle, angle], axis=-1)  # [B, N, K, 6]
    #neighbor_point_features = tf.concat([neighbor_centroid_dist, neighbor_reference_dist, neighbor_inter_dist, angle], axis=-1)  # [B, N, K, 6]
    #neighbor_point_features = tf.concat([neighbor_centroid_dist, neighbor_reference_dist, neighbor_inter_dist,
    #                                     centroid_neighbor_reference_angle, reference_neighbor_inter_angle,
    #                                     inter_neighbor_centroid_angle], axis=-1)  # [B, N, K, 6]

    rotinv_features = tf.concat([center_point_features_tile, neighbor_point_features], axis=-1)

    # rotinv_features = neighbor_point_features

    return center_point_features_tile, neighbor_point_features, rotinv_features

def prepare_rotinv_input_global_3dv(new_xyz, radius, nsample, grouped_xyz):
    # new_xyz: [B, npoint, 3]
    # nsample: neighbor size
    # grouped_xyz: [B, npoint, nsample, 3]

    batch_size = new_xyz.get_shape()[0].value
    point_num = new_xyz.get_shape()[1].value

    # calculate neighbor centroid
    #centroid_xyz = tf.reduce_mean(grouped_xyz, axis=2)  # [B, N, 3]
    centroid_xyz = get_robust_centroid(grouped_xyz)  # [B, N, 3]

    # calculate intersection point
    reference_vector_norm = tf.norm(new_xyz, axis=-1, keepdims=True)  # [B, N, 1]
    reference_vector_unit = new_xyz / (reference_vector_norm + 0.0001)  # [B, N, 3]
    inter_xyz = radius * reference_vector_unit + new_xyz

    # prepare features of center point
    centroid_reference_vector = new_xyz - centroid_xyz
    centroid_reference_dist = tf.norm(centroid_reference_vector, axis=-1, keepdims=True)  # [B, N, 1]

    centroid_inter_vector = inter_xyz - centroid_xyz
    centroid_inter_dist = tf.norm(centroid_inter_vector, axis=-1, keepdims=True)  # [B, N, 1]

    dot_product = tf.reduce_sum(tf.multiply(centroid_reference_vector, centroid_inter_vector), axis=-1,
                                keepdims=True)
    reference_centroid_inter_angle = dot_product / (centroid_reference_dist * centroid_inter_dist + 0.000001)

    inter_reference_vector = new_xyz - inter_xyz
    inter_centroid_vector = centroid_xyz - inter_xyz
    dot_product = tf.reduce_sum(tf.multiply(inter_reference_vector, inter_centroid_vector), axis=-1,
                                keepdims=True)
    reference_inter_centroid_angle = dot_product / (radius * centroid_inter_dist + 0.000001)

    center_point_features = tf.concat([reference_vector_norm, centroid_reference_dist, centroid_inter_dist,
                                       reference_centroid_inter_angle, reference_inter_centroid_angle],
                                      axis=-1)  # [B, N, 5]
    center_point_features_tile = tf.tile(tf.expand_dims(center_point_features, axis=2),
                                         [1, 1, nsample, 1])  # [B, N, K, 5]

    # calculate neighbor centroid

    # prepare features of neighbor points
    centroid_xyz_tile = tf.tile(tf.expand_dims(centroid_xyz, axis=2), [1, 1, nsample, 1])
    centroid_neighbor_vector = grouped_xyz - centroid_xyz_tile

    reference_vector_tile = tf.tile(tf.expand_dims(new_xyz, axis=2), [1, 1, nsample, 1])
    neighbor_reference_vector = reference_vector_tile - grouped_xyz
    centroid_reference_vector = reference_vector_tile - centroid_xyz_tile

    neighbor_centroid_dist = tf.norm(centroid_neighbor_vector, axis=-1, keepdims=True)
    neighbor_reference_dist = tf.norm(neighbor_reference_vector, axis=-1, keepdims=True)
    centroid_reference_dist = tf.norm(centroid_reference_vector, axis=-1, keepdims=True)

    dot_product = tf.reduce_sum(tf.multiply(centroid_neighbor_vector, centroid_reference_vector), axis=-1,
                                keepdims=True)
    neighbor_centroid_reference_angle = dot_product / (neighbor_centroid_dist *
                                                       centroid_reference_dist + 0.000001)

    dot_product = tf.reduce_sum(tf.multiply(-centroid_reference_vector, -neighbor_reference_vector), axis=-1,
                                keepdims=True)
    neighbor_reference_centroid_angle = dot_product / (centroid_reference_dist *
                                                       neighbor_reference_dist + 0.000001)

    neighbor_point_features = tf.concat([neighbor_reference_dist, neighbor_centroid_dist,
                                         neighbor_reference_centroid_angle,
                                         neighbor_centroid_reference_angle], axis=-1)  # [B, N, K, 6]

    rotinv_features = tf.concat([center_point_features_tile, neighbor_point_features], axis=-1)

    # rotinv_features = neighbor_point_features

    return center_point_features_tile, neighbor_point_features, rotinv_features

def prepare_rotinv_input_3dv(new_xyz, radius, nsample, grouped_xyz):
    # new_xyz: [B, npoint, 3]
    # nsample: neighbor size
    # grouped_xyz: [B, npoint, nsample, 3]

    # calculate neighbor centroid
    centroid_xyz = tf.reduce_mean(grouped_xyz, axis=2)  # [B, N, 3]

    # prepare features of neighbor points
    centroid_xyz_tile = tf.tile(tf.expand_dims(centroid_xyz, axis=2), [1, 1, nsample, 1])
    centroid_neighbor_vector = grouped_xyz - centroid_xyz_tile

    reference_vector_tile = tf.tile(tf.expand_dims(new_xyz, axis=2), [1, 1, nsample, 1])
    neighbor_reference_vector = reference_vector_tile - grouped_xyz
    centroid_reference_vector = reference_vector_tile - centroid_xyz_tile

    neighbor_centroid_dist = tf.norm(centroid_neighbor_vector, axis=-1, keepdims=True)
    neighbor_reference_dist = tf.norm(neighbor_reference_vector, axis=-1, keepdims=True)
    centroid_reference_dist = tf.norm(centroid_reference_vector, axis=-1, keepdims=True)

    dot_product = tf.reduce_sum(tf.multiply(centroid_neighbor_vector, centroid_reference_vector), axis=-1,
                                keepdims=True)
    neighbor_centroid_reference_angle = dot_product / (neighbor_centroid_dist *
                                                       centroid_reference_dist + 0.000001)

    dot_product = tf.reduce_sum(tf.multiply(-centroid_reference_vector, -neighbor_reference_vector), axis=-1,
                                keepdims=True)
    neighbor_reference_centroid_angle = dot_product / (centroid_reference_dist *
                                                    neighbor_reference_dist + 0.000001)


    neighbor_point_features = tf.concat([neighbor_reference_dist, neighbor_centroid_dist,
                                         neighbor_reference_centroid_angle,
                                         neighbor_centroid_reference_angle], axis=-1)  # [B, N, K, 6]

    rotinv_features = neighbor_point_features

    return neighbor_point_features, neighbor_point_features, rotinv_features

def pointnet_sa_module_rotinv_v66(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=False, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

        # _, _, rotinv_features = prepare_rotinv_input_radius_tetrahedron(new_xyz, 1.0, nsample, grouped_xyz)
        _, _, rotinv_features = prepare_rotinv_input_radius_tetrahedron_angle_new_robust(new_xyz, 1.0, nsample, grouped_xyz)

        # _, _, rotinv_features = prepare_rotinv_input_3dv(new_xyz, 1.0, nsample, grouped_xyz)
        rotinv_features = tf_util_pointnet2.conv2d(rotinv_features, mlp[0], [1, 1],
                                                   padding='VALID', stride=[1, 1], bn=bn,
                                                   is_training=is_training,
                                                   scope='rotinv_conv', bn_decay=bn_decay)  # [B, N, K, 64]

        # Point Feature Embedding
        # new_points = new_points - rotinv_features
        grouped_features = tf.concat([rotinv_features, new_points], axis=-1)
        if use_nchw: grouped_features = tf.transpose(grouped_features, [0,3,1,2])

        for i, num_out_channel in enumerate(mlp):
            grouped_features = tf_util_pointnet2.conv2d(grouped_features, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format)
        if use_nchw: grouped_features = tf.transpose(grouped_features, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            grouped_features = tf.reduce_max(grouped_features, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            grouped_features = tf.reduce_mean(grouped_features, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                grouped_features *= weights # (batch_size, npoint, nsample, mlp[-1])
                grouped_features = tf.reduce_sum(grouped_features, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(grouped_features, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(grouped_features, axis=[2], keep_dims=True, name='avgpool')
            grouped_features = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing
        if mlp2 is not None:
            if use_nchw: grouped_features = tf.transpose(grouped_features, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                grouped_features = tf_util_pointnet2.conv2d(grouped_features, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format)
            if use_nchw: grouped_features = tf.transpose(grouped_features, [0,2,3,1])

        grouped_features = tf.squeeze(grouped_features, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, grouped_features, idx

def pointnet_sa_module_rotinv_v66_robust(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=False, use_nchw=False, robust=True):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

        if robust == True:
            _, _, rotinv_features = prepare_rotinv_input_radius_tetrahedron_angle_new_robust(new_xyz, 1.0, nsample, grouped_xyz)
        else:
            _, _, rotinv_features = prepare_rotinv_input_radius_tetrahedron_angle_new(new_xyz, 1.0, nsample,
                                                                                             grouped_xyz)
        rotinv_features = tf_util_pointnet2.conv2d(rotinv_features, mlp[0], [1, 1],
                                                   padding='VALID', stride=[1, 1], bn=bn,
                                                   is_training=is_training,
                                                   scope='rotinv_conv', bn_decay=bn_decay)  # [B, N, K, 64]

        # Point Feature Embedding
        # new_points = new_points - rotinv_features
        grouped_features = tf.concat([rotinv_features, new_points], axis=-1)
        if use_nchw: grouped_features = tf.transpose(grouped_features, [0,3,1,2])

        for i, num_out_channel in enumerate(mlp):
            grouped_features = tf_util_pointnet2.conv2d(grouped_features, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format)
        if use_nchw: grouped_features = tf.transpose(grouped_features, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            grouped_features = tf.reduce_max(grouped_features, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            grouped_features = tf.reduce_mean(grouped_features, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                grouped_features *= weights # (batch_size, npoint, nsample, mlp[-1])
                grouped_features = tf.reduce_sum(grouped_features, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(grouped_features, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(grouped_features, axis=[2], keep_dims=True, name='avgpool')
            grouped_features = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing
        if mlp2 is not None:
            if use_nchw: grouped_features = tf.transpose(grouped_features, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                grouped_features = tf_util_pointnet2.conv2d(grouped_features, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format)
            if use_nchw: grouped_features = tf.transpose(grouped_features, [0,2,3,1])

        grouped_features = tf.squeeze(grouped_features, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, grouped_features, idx

def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, is_training, bn_decay, scope, bn=True, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])
            if points is not None:
                grouped_points = group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0,3,1,2])
            for j,num_out_channel in enumerate(mlp_list[i]):
                grouped_points = tf_util_pointnet2.conv2d(grouped_points, num_out_channel, [1,1],
                                                padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                                scope='conv%d_%d'%(i,j), bn_decay=bn_decay)
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0,2,3,1])
            new_points = tf.reduce_max(grouped_points, axis=[2])
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return new_xyz, new_points_concat

def pointnet_sa_module_msg_rotinv_v66(xyz, points, npoint, radius_list, nsample_list, mlp_list, is_training, bn_decay, scope, bn=True, use_xyz=False, use_nchw=False):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))   # [B, N, 3]
        # batch_size = new_xyz.get_shape()[0].value
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)   # [B, N, K, 3]

            # _, _, rotinv_features_raw = prepare_rotinv_input_radius_tetrahedron(new_xyz, radius, nsample, grouped_xyz)
            _, _, rotinv_features_raw = prepare_rotinv_input_radius_tetrahedron_angle_new_robust(new_xyz, radius, nsample, grouped_xyz)
            #_, _, rotinv_features_raw = prepare_rotinv_input_global_3dv(new_xyz, radius, nsample, grouped_xyz)

            #_, _, rotinv_features_raw = prepare_rotinv_input_3dv(new_xyz, radius, nsample, grouped_xyz)

            if points is not None:
                rotinv_features = tf_util_pointnet2.conv2d(rotinv_features_raw, mlp_list[i][-1], [1, 1],
                                                            padding='VALID', stride=[1, 1], bn=bn,
                                                            is_training=is_training,
                                                            scope='rotinv_conv%d'%(i), bn_decay=bn_decay)  # [B, N, K, 64]

                grouped_points = group_point(points, idx)  # [B, N, K, C]

                # grouped_points = grouped_points - rotinv_features

                grouped_features = tf.concat([rotinv_features, grouped_points], axis=-1)   # [B, N, K, C+64]
            else:
                grouped_features = rotinv_features_raw
            if use_nchw: grouped_features = tf.transpose(grouped_features, [0,3,1,2])
            for j,num_out_channel in enumerate(mlp_list[i]):
                grouped_features = tf_util_pointnet2.conv2d(grouped_features, num_out_channel, [1,1],
                                                padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                                scope='conv%d_%d'%(i,j), bn_decay=bn_decay)
            if use_nchw: grouped_features = tf.transpose(grouped_features, [0,2,3,1])
            new_points = tf.reduce_max(grouped_features, axis=[2], keepdims=True)
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)

        new_points_concat = tf_util_pointnet2.conv2d(new_points_concat, new_points_concat.get_shape()[-1].value, [1, 1],
                                                    padding='VALID', stride=[1, 1], bn=bn, is_training=is_training,
                                                    scope='conv_concat', bn_decay=bn_decay)  # [B, N, 1, C]

        return new_xyz, tf.squeeze(new_points_concat, axis=2), rotinv_features_raw

def pointnet_sa_module_msg_rotinv_v66_robust(xyz, points, npoint, radius_list, nsample_list, mlp_list, is_training, bn_decay, scope, bn=True, use_xyz=False, use_nchw=False, robust=True):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))   # [B, N, 3]
        # batch_size = new_xyz.get_shape()[0].value
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)   # [B, N, K, 3]

            if robust == True:
                _, _, rotinv_features_raw = prepare_rotinv_input_radius_tetrahedron_angle_new_robust(new_xyz, radius, nsample, grouped_xyz)
            else:
                _, _, rotinv_features_raw = prepare_rotinv_input_radius_tetrahedron_angle_new(new_xyz, radius,
                                                                                                     nsample,
                                                                                                     grouped_xyz)

            if points is not None:
                rotinv_features = tf_util_pointnet2.conv2d(rotinv_features_raw, mlp_list[i][-1], [1, 1],
                                                            padding='VALID', stride=[1, 1], bn=bn,
                                                            is_training=is_training,
                                                            scope='rotinv_conv%d'%(i), bn_decay=bn_decay)  # [B, N, K, 64]

                grouped_points = group_point(points, idx)  # [B, N, K, C]

                # grouped_points = grouped_points - rotinv_features

                grouped_features = tf.concat([rotinv_features, grouped_points], axis=-1)   # [B, N, K, C+64]
            else:
                grouped_features = rotinv_features_raw
            if use_nchw: grouped_features = tf.transpose(grouped_features, [0,3,1,2])
            for j,num_out_channel in enumerate(mlp_list[i]):
                grouped_features = tf_util_pointnet2.conv2d(grouped_features, num_out_channel, [1,1],
                                                padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                                scope='conv%d_%d'%(i,j), bn_decay=bn_decay)
            if use_nchw: grouped_features = tf.transpose(grouped_features, [0,2,3,1])
            new_points = tf.reduce_max(grouped_features, axis=[2], keepdims=True)
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)

        new_points_concat = tf_util_pointnet2.conv2d(new_points_concat, new_points_concat.get_shape()[-1].value, [1, 1],
                                                    padding='VALID', stride=[1, 1], bn=bn, is_training=is_training,
                                                    scope='conv_concat', bn_decay=bn_decay)  # [B, N, 1, C]

        return new_xyz, tf.squeeze(new_points_concat), rotinv_features_raw

def relation_conv(features, nsample, out_channel, bn, is_training, bn_decay, scope):
    # features: [B, N, K, C]

    # features_sum = tf.reduce_sum(features, axis=2, keepdims=True)  # [B, N, 1, C]
    # features_sum = tf.tile(features_sum, [1, 1, nsample, 1])  # [B, N, K, C]
    # features_mean = (features_sum - features) / (nsample - 1)

    # features = tf.concat([features, features_mean], axis=-1)

    features_max = tf.reduce_max(features, axis=2, keepdims=True)  # [B, N, 1, C]
    features_max_tile = tf.tile(features_max, [1, 1, nsample, 1])

    features = tf.concat([features, features_max_tile], axis=-1)

    features = tf_util_pointnet2.conv2d(features, out_channel, [1, 1],
                                               padding='VALID', stride=[1, 1], bn=bn,
                                               is_training=is_training,
                                               scope=scope, bn_decay=bn_decay)  # [B, N, K, 64]

    return features

def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util_pointnet2.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1

def pointnet_sa_module_msg_rotinv_v66_pointnet2(xyz, points, npoint, radius_list, nsample_list, mlp_list, is_training, bn_decay, scope, bn=True, use_xyz=False, use_nchw=False):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))   # [B, N, 3]
        # batch_size = new_xyz.get_shape()[0].value
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)   # [B, N, K, 3]

            # _, _, rotinv_features_raw = prepare_rotinv_input_radius_tetrahedron(new_xyz, radius, nsample, grouped_xyz)
            _, _, rotinv_features_raw = prepare_rotinv_input_radius_tetrahedron_angle_new_robust(new_xyz, radius, nsample, grouped_xyz)

            # _, _, rotinv_features_raw = prepare_rotinv_input_3dv(new_xyz, radius, nsample, grouped_xyz)

            if points is not None:

                grouped_features = group_point(points, idx)  # [B, N, K, C]

            else:
                grouped_features = rotinv_features_raw

            if use_nchw: grouped_features = tf.transpose(grouped_features, [0,3,1,2])
            for j,num_out_channel in enumerate(mlp_list[i]):
                grouped_features = tf_util_pointnet2.conv2d(grouped_features, num_out_channel, [1,1],
                                                padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                                scope='conv%d_%d'%(i,j), bn_decay=bn_decay)
            if use_nchw: grouped_features = tf.transpose(grouped_features, [0,2,3,1])
            new_points = tf.reduce_max(grouped_features, axis=[2], keepdims=True)
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)

        return new_xyz, tf.squeeze(new_points_concat, axis=2), rotinv_features_raw

def pointnet_sa_module_rotinv_v66_pointnet2(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=False, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)


        # Point Feature Embedding
        # new_points = new_points - rotinv_features
        grouped_features = new_points
        if use_nchw: grouped_features = tf.transpose(grouped_features, [0,3,1,2])

        for i, num_out_channel in enumerate(mlp):
            grouped_features = tf_util_pointnet2.conv2d(grouped_features, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format)
        if use_nchw: grouped_features = tf.transpose(grouped_features, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            grouped_features = tf.reduce_max(grouped_features, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            grouped_features = tf.reduce_mean(grouped_features, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                grouped_features *= weights # (batch_size, npoint, nsample, mlp[-1])
                grouped_features = tf.reduce_sum(grouped_features, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(grouped_features, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(grouped_features, axis=[2], keep_dims=True, name='avgpool')
            grouped_features = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing
        if mlp2 is not None:
            if use_nchw: grouped_features = tf.transpose(grouped_features, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                grouped_features = tf_util_pointnet2.conv2d(grouped_features, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format)
            if use_nchw: grouped_features = tf.transpose(grouped_features, [0,2,3,1])

        grouped_features = tf.squeeze(grouped_features, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, grouped_features, idx

def get_rotinv_features(xyz, bn, is_training, bn_decay):
    radius = 0.1
    nsample = 16
    idx, pts_cnt = query_ball_point(radius, nsample, xyz, xyz)
    grouped_xyz = group_point(xyz, idx)  # [B, N, K, 3]

    _, _, rotinv_features_raw = prepare_rotinv_input_radius_tetrahedron_angle_new_robust(xyz, radius, nsample,
                                                                                         grouped_xyz) #[B, N, K, 12]

    rotinv_features = tf_util_pointnet2.conv2d(rotinv_features_raw, 16, [1, 1],
                                               padding='VALID', stride=[1, 1], bn=bn,
                                               is_training=is_training,
                                               scope='rotinv_input', bn_decay=bn_decay)  # [B, N, K, 16]
    rotinv_features = tf.reduce_max(rotinv_features, axis=2)  # [B, N, 16]

    return rotinv_features