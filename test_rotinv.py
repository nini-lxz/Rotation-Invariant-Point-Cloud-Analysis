import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import h5py


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=3, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model', help='Model name: dgcnn [default: dgcnn]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log_y_cls/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_INDEX)
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, './data_mn40/shape_names.txt'))] 

HOSTNAME = socket.gethostname()

f_test = h5py.File('./data_mn40/test_1024.h5')
test_data = f_test['data'][:]
test_label = f_test['label'][:]
f_test.close()
test_label = np.squeeze(test_label)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES)
        loss = MODEL.get_loss_dgcnn(pred, labels_pl, NUM_CLASSES)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss}

    eval_one_epoch(sess, ops, num_votes)

   
def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    is_training = False

    global test_data
    file_size = test_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    print(file_size)

    eval_accu_best = 0.0
    for vote_idx in range(num_votes):
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            cur_batch_size = end_idx - start_idx

            # Aggregating BEG
            batch_loss_sum = 0 # sum of losses for the batch
            batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES)) # score for classes
            batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES)) # 0/1 for classes
            rotated_data = provider.rotate_point_cloud_so(test_data[start_idx:end_idx, :, :])
            feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['labels_pl']: test_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
            batch_pred_sum += pred_val
            batch_pred_val = np.argmax(pred_val, 1)
            for el_idx in range(cur_batch_size):
                batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
            batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))
            pred_val = np.argmax(batch_pred_sum, 1)
            # Aggregating END

            correct = np.sum(pred_val == test_label[start_idx:end_idx])
            total_correct += correct
            total_seen += cur_batch_size
            loss_sum += batch_loss_sum

            for i in range(start_idx, end_idx):
                l = test_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)
                fout.write('%d, %d\n' % (pred_val[i-start_idx], l))

        eval_accu = (total_correct / float(total_seen))
        print eval_accu

        if eval_accu >= eval_accu_best:
            eval_accu_best = eval_accu
            class_accuracies_best = np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float)+0.0000001)

    log_string('eval best accuracy: %f' % eval_accu_best)
    for i, name in enumerate(SHAPE_NAMES):
        log_string('%10s:\t%0.3f' % (name, class_accuracies_best[i]))
    


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=30)  # run 30 times and select the highest accuracy
    LOG_FOUT.close()
