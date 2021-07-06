import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'grouping'))
sys.path.append(os.path.join(ROOT_DIR, '3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point

def knn_search(downsampled_xyz, original_xyz):
    # downsampled_xyz: (B, M, 3)
    # original_xyz: (B, N, 3) M<=N

    original_xyz_transpose = tf.transpose(original_xyz, perm=[0, 2, 1])  # [B, 3, N]
    inner = tf.matmul(downsampled_xyz, original_xyz_transpose)  # [B, M, N]
    inner = -2*inner
    downsampled_xyz_square = tf.reduce_sum(tf.square(downsampled_xyz), axis=-1, keepdims=True)  # [B, M, 1]
    original_xyz_square = tf.reduce_sum(tf.square(original_xyz), axis=-1, keepdims=True)  # [B, N, 1]
    original_xyz_square_transpose = tf.transpose(original_xyz_square, perm=[0, 2, 1])  # [B, 1, N]
    adj_matrix = downsampled_xyz_square + inner + original_xyz_square_transpose  # [B, M, N]

    return adj_matrix

def xyz2rotinv(downsampled_xyz, grouped_xyz):
    # downsampled_xyz: [B, M, 3]
    # grouped_xyz: [B, M, nsample, 3]

    batch_size = downsampled_xyz.get_shape()[0].value
    point_num = downsampled_xyz.get_shape()[1].value

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

def rotinv_feature_extraction(input_xyz, input_features, sample_pts_num, dilation_rate, nsample_list,
                              mlp_list, is_training, bn_decay, scope, bn=True):

    '''
        Input:
            input_xyz: (batch_size, point_number, 3) TF tensor
            input_features: (batch_size, point_number, channel) TF tensor
            sample_pts_num: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor

        Step 1: farthest sampling to select a subset of points for feature extraction
        Step 2: transform point coordinates to rotation-invariant representations
        Step 3: feature convolution
    '''

    batch_size = input_xyz.get_shape()[0].value
    original_num_points = input_xyz.get_shape()[1].value

    # Step 1 ##
    with tf.variable_scope(scope) as sc:
        downsampled_xyz = gather_point(input_xyz, farthest_point_sample(sample_pts_num, input_xyz))   # [B, N, 3]

        # for each selected point, find its k-nearest neighbor points in the original points
        adj_matrix = knn_search(downsampled_xyz, input_xyz)  # [B, M, N]
        neg_adj = -adj_matrix

        idx_ = tf.range(batch_size) * original_num_points
        idx_ = tf.reshape(idx_, [batch_size, 1, 1])
        input_xyz_flat = tf.reshape(input_xyz, [-1, 3])  # [B*N, 3]
        input_features_flat = tf.reshape(input_features, [-1, 3])  # [B*N, C]

        for i in range(len(nsample_list)):
            nsample = nsample_list[i]
            _, nn_idx_dilated = tf.nn.top_k(neg_adj, k=nsample*dilation_rate)  # [B, M, k*D]
            nn_idx = nn_idx_dilated[:, :, ::dilation_rate]  # [B, M, k]

            grouped_xyz = tf.gather(input_xyz_flat, nn_idx + idx_)  # [B, M, k, 3]
            grouped_features = tf.gather(input_features_flat, nn_idx + idx_)  # [B, M, k, C]

            ## Step 2 ##
            _, _, rotinv_features_raw = xyz2rotinv(new_xyz, radius,
                                                                                                 nsample, grouped_xyz)

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

            ## Step 3 ##
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
