import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util_pointnet2
import tf_util
from transform_nets import input_transform_net
from pointnet_util import pointnet_sa_module_msg_rotinv_v66, pointnet_sa_module_rotinv_v66, \
							pointnet_sa_module, pointnet_sa_module_msg, \
							pointnet_fp_module
#from models.ops import dense_attn_unit

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def placeholder_inputs_seg(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    cls_labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl, cls_labels_pl

def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def region_atten_new(inputs_xyz, inputs_features, scope, is_training, bn_decay):
    # inputs_xyz: [B, N, 3]
    # inputs_features: [B, N, C]
    with tf.variable_scope(scope):
        feature_dim = inputs_features.get_shape()[-1].value
        pt_num = inputs_features.get_shape()[1].value

        # compute pairwise distance and angle
        inputs_xyz_transpose = tf.transpose(inputs_xyz, [0, 2, 1])  # [B, 3, N]
        inputs_xyz_inner = tf.matmul(inputs_xyz, inputs_xyz_transpose)  # [B, N, N]

        inputs_xyz_square = tf.reduce_sum(tf.square(inputs_xyz), axis=-1, keepdims=True)  # [B, N, 1]
        inputs_xyz_square_transpose = tf.transpose(inputs_xyz_square, perm=[0, 2, 1])  # [B, 1, N]

        inputs_xyz_pairwise_dist = inputs_xyz_square + inputs_xyz_square_transpose - 2 * inputs_xyz_inner  # [B, N, N]

        inputs_xyz_norm = tf.norm(inputs_xyz, axis=-1, keepdims=True)  # [B, N, 1]
        inputs_xyz_norm_transpose = tf.transpose(inputs_xyz_norm, perm=[0, 2, 1])  # [B, 1, N]
        inputs_xyz_angle = inputs_xyz_inner / (inputs_xyz_norm*inputs_xyz_norm_transpose+0.00001)

        inputs_xyz_relation = tf.concat([inputs_xyz_pairwise_dist, inputs_xyz_angle], axis=-1)  # [B, N, 2N]

        inputs_xyz_relation = tf.expand_dims(inputs_xyz_relation, axis=2)  # [B, N, 1, N]
        inputs_xyz_relation = tf_util_pointnet2.conv2d(inputs_xyz_relation, feature_dim, [1, 1],
                                                        padding='VALID', stride=[1, 1], bn=True,
                                                        is_training=is_training,
                                                        scope=scope, bn_decay=None,
                                                        activation_fn=None)  # [B, N, 1, C]
        weights = tf.squeeze(inputs_xyz_relation)  # [B, N, C]
        weights = tf.nn.softmax(weights, axis=1)

        inputs_refine = inputs_features + inputs_features*weights

    return inputs_refine

use_attn = True
print("use_attn:", use_attn)

def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    l0_xyz = point_cloud
    l0_points = None

    # Set abstraction layers
    l1_xyz, l1_points, l1_rotinv_features_raw = pointnet_sa_module_msg_rotinv_v66(l0_xyz, l0_points, 512,
                                                                                [0.1, 0.2, 0.4], [16, 32, 64],
                                                                                [[32, 32, 64], [32, 32, 64],
                                                                                 [64, 64, 128]],
                                                                                is_training, bn_decay,
                                                                                scope='layer1')
    if use_attn:
        l1_points = region_atten_new(l1_xyz, l1_points, 'region_atten_unit1', is_training, bn_decay)

    l2_xyz, l2_points, l2_rotinv_features_raw = pointnet_sa_module_msg_rotinv_v66(l1_xyz, l1_points, 128,
                                                                                    [0.2, 0.4, 0.8], [32, 64, 128],
                                                                                    [[64, 64, 128], [64, 64, 128],
                                                                                     [128, 128, 256]],
                                                                                    is_training, bn_decay,
                                                                                    scope='layer2')
    if use_attn:
        l2_points = region_atten_new(l2_xyz, l2_points, 'region_atten_unit2', is_training, bn_decay)


    l3_xyz, l3_points, _ = pointnet_sa_module_rotinv_v66(l2_xyz, l2_points, npoint=None, radius=None,
                                                           nsample=None,
                                                           mlp=[256, 512, 1024], mlp2=None, group_all=True,
                                                           is_training=is_training, bn_decay=bn_decay,
                                                           scope='layer3')

    # Fully connected layers
    codeword = tf.reshape(l3_points, [batch_size, -1])
    net = tf_util_pointnet2.fully_connected(codeword, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util_pointnet2.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp1')
    net = tf_util_pointnet2.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util_pointnet2.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp2')
    net = tf_util_pointnet2.fully_connected(net, 40, activation_fn=None, scope='fc3')

    # return net, end_points
    return net, codeword


NUM_CATEGORIES = 16

def get_model_seg(point_cloud, cls_label, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    l0_xyz = point_cloud
    l0_points = None

    # Set abstraction layers
    # l1_xyz, l1_points, l1_rotinv_features_raw = pointnet_sa_module_msg_rotinv_v66_robust(l0_xyz, l0_points, 512, [0.1, 0.2, 0.4], [16, 32, 64],
    #                                           [[32, 32, 64], [32, 32, 64], [64, 64, 128]],
    #                                            is_training, bn_decay, scope='layer1', robust=False)
    l1_xyz, l1_points, l1_rotinv_features_raw = pointnet_sa_module_msg_rotinv_v66(l0_xyz, l0_points, 512,
                                                                                         [0.1, 0.2, 0.4], [16, 32, 64],
                                                                                         [[32, 32, 64], [32, 32, 64],
                                                                                          [64, 64, 128]],
                                                                                         is_training, bn_decay,
                                                                                         scope='layer1')
    if use_attn:
        l1_points = region_atten_new(l1_xyz, l1_points, 'region_atten_unit1', is_training, bn_decay)

    # l2_xyz, l2_points, l2_rotinv_features_raw = pointnet_sa_module_msg_rotinv_v66_robust(l1_xyz, l1_points, 128, [0.2, 0.4, 0.8], [32, 64, 128],
    #                                                   [[64, 64, 128], [64, 64, 128], [128, 128, 256]],
    #                                            is_training, bn_decay, scope='layer2', robust=True)
    l2_xyz, l2_points, l2_rotinv_features_raw = pointnet_sa_module_msg_rotinv_v66(l1_xyz, l1_points, 128,
                                                                                         [0.2, 0.4, 0.8], [32, 64, 128],
                                                                                         [[64, 64, 128], [64, 64, 128],
                                                                                          [128, 128, 256]],
                                                                                         is_training, bn_decay,
                                                                                         scope='layer2')

    if use_attn:
        l2_points = region_atten_new(l2_xyz, l2_points, 'region_atten_unit2', is_training, bn_decay)

    # l3_xyz, l3_points, _ = pointnet_sa_module_rotinv_v66_robust(l2_xyz, l2_points, npoint=None, radius=None, nsample=None,
    #                                           mlp=[256,512,1024], mlp2=None, group_all=True,
    #                                           is_training=is_training, bn_decay=bn_decay, scope='layer3', robust=True)
    l3_xyz, l3_points, _ = pointnet_sa_module_rotinv_v66(l2_xyz, l2_points, npoint=None, radius=None,
                                                                nsample=None,
                                                                mlp=[256, 512, 1024], mlp2=None, group_all=True,
                                                                is_training=is_training, bn_decay=bn_decay,
                                                                scope='layer3')

    # Feature propagation layers
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256, 256], is_training, bn_decay,
                                   scope='fa_layer1')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 128], is_training, bn_decay,
                                   scope='fa_layer2')

    cls_label_one_hot = tf.one_hot(cls_label, depth=NUM_CATEGORIES, on_value=1.0, off_value=0.0)
    cls_label_one_hot = tf.reshape(cls_label_one_hot, [batch_size, 1, NUM_CATEGORIES])
    cls_label_one_hot = tf.tile(cls_label_one_hot, [1, num_point, 1])
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, cls_label_one_hot, l1_points, [128, 128], is_training, bn_decay, scope='fp_layer3')

    # FC layers
    net = tf_util_pointnet2.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1',
                                   bn_decay=bn_decay)
    end_points['feats'] = net
    net = tf_util_pointnet2.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util_pointnet2.conv1d(net, 50, 1, padding='VALID', activation_fn=None, scope='fc2')  # [B, N, 50]

    return net, end_points

def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

def get_loss_dgcnn(pred, label, end_points):
  """ pred: B*NUM_CLASSES,
      label: B, """
  labels = tf.one_hot(indices=label, depth=40)
  #labels = tf.one_hot(indices=label, depth=55)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
  classify_loss = tf.reduce_mean(loss)
  return classify_loss

def get_loss_seg(pred, label):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((8,1024,3))
        cls_label = tf.zeros((1,), dtype=tf.int32)
        net, _ = get_model(inputs, tf.constant(True))
        # net, _ = get_model_seg(inputs, cls_label, tf.constant(True))
        print np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print(net)
