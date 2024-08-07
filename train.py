__author__ = 'yunbo'
__editor__ = 'songhune'
import os.path
import time
import numpy as np
import tensorflow as tf
# v1 only has flag attributes
import tensorflow.compat.v1 as tf
import cv2
import sys
import random
from nets import models_factory
from data_provider import datasets_factory
from utils import preprocess
from utils import metrics
#from skimage.measure import compare_ssim 
#compare_ssim was depricated
from skimage.metrics import structural_similarity
import csv
# -----------------------------------------------------------------------------
FLAGS = tf.app.flags.FLAGS
tf.disable_v2_behavior()
# data I/O
tf.app.flags.DEFINE_string('dataset_name', 'kth',
                           'The name of dataset.')
tf.app.flags.DEFINE_string('train_data_paths',
                           '/Users/songhune/Downloads/data/kth_preprocessed/train/train_dataset_1.npz',
                           'train data paths.')
tf.app.flags.DEFINE_string('valid_data_paths',
                           '/home/work/songhune/data/kth_preprocessed/validation/validation_dataset_1.npz',
                           'validation data paths.')
tf.app.flags.DEFINE_string('save_dir', '/Users/songhune/kth',
                            'dir to store trained net.')
tf.app.flags.DEFINE_string('gen_frm_dir', '/Users/songhune/kth',
                           'dir to store result.')
# model
tf.app.flags.DEFINE_string('model_name', 'predrnn_pp',
                           'The name of the architecture.')
tf.app.flags.DEFINE_string('pretrained_model', '',
                           'file of a pretrained model to initialize from.')
tf.app.flags.DEFINE_integer('input_length', 10,
                            'encoder hidden states.')
tf.app.flags.DEFINE_integer('seq_length', 20,
                            'total input and output length.')
tf.app.flags.DEFINE_integer('img_width', 120,
                            'input image width.')
tf.app.flags.DEFINE_integer('img_height', 160,
                            'input image height.')
tf.app.flags.DEFINE_integer('img_channel', 1,
                            'number of image channel.')
tf.app.flags.DEFINE_integer('stride', 1,
                            'stride of a convlstm layer.')
tf.app.flags.DEFINE_integer('filter_size', 5,
                            'filter of a convlstm layer.')
tf.app.flags.DEFINE_string('num_hidden', '128,64,64,64',
                           'COMMA separated number of units in a convlstm layer.')
tf.app.flags.DEFINE_integer('patch_size', 4,
                            'patch size on one dimension.')
tf.app.flags.DEFINE_boolean('layer_norm', True,
                            'whether to apply tensor layer norm.')
# optimization
tf.app.flags.DEFINE_float('lr', 0.001,
                          'base learning rate.')
tf.app.flags.DEFINE_boolean('reverse_input', True,
                            'whether to reverse the input frames while training.')
tf.app.flags.DEFINE_integer('batch_size', 8,
                            'batch size for training.')
tf.app.flags.DEFINE_integer('max_iterations', 80000,
                            'max num of steps.')
tf.app.flags.DEFINE_integer('display_interval', 1,
                            'number of iters showing training loss.')
tf.app.flags.DEFINE_integer('test_interval', 2000,
                            'number of iters for test.')
tf.app.flags.DEFINE_integer('snapshot_interval', 10000,
                            'number of iters saving models.')
tf.compat.v1.disable_eager_execution()

class Model(object):
    def __init__(self, input_handle):
        self.input_handle = input_handle
        self.x = tf.placeholder(tf.float32,
                                [FLAGS.batch_size,
                                 FLAGS.seq_length,
                                 FLAGS.img_width // FLAGS.patch_size,
                                 FLAGS.img_height // FLAGS.patch_size,
                                 FLAGS.patch_size * FLAGS.patch_size * FLAGS.img_channel])

        self.mask_true = tf.placeholder(tf.float32,
                                        [FLAGS.batch_size,
                                         FLAGS.seq_length,
                                         FLAGS.img_width // FLAGS.patch_size,
                                         FLAGS.img_height // FLAGS.patch_size,
                                         FLAGS.patch_size * FLAGS.patch_size * FLAGS.img_channel])

        grads = []
        loss_train = []
        self.pred_seq = []
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        num_hidden = [int(x) for x in FLAGS.num_hidden.split(',')]
        num_layers = len(num_hidden)
        with tf.compat.v1.variable_scope(tf.get_variable_scope()):
            output_list = models_factory.construct_model(
                FLAGS.model_name, self.x,
                self.mask_true,
                num_layers, num_hidden,
                FLAGS.filter_size, FLAGS.stride,
                None,  # Use None for dynamic sequence length
                FLAGS.input_length,
                FLAGS.layer_norm)
            gen_ims = output_list[0]
            loss = output_list[1]
            pred_ims = gen_ims[:, FLAGS.input_length - 1:]
            self.loss_train = loss / FLAGS.batch_size
            all_params = tf.trainable_variables()
            grads.append(tf.gradients(loss, all_params))
            self.pred_seq.append(pred_ims)

        self.train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables)
        init = tf.global_variables_initializer()
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config=configProt)
        self.sess.run(init)
        if FLAGS.pretrained_model:
            self.saver.restore(self.sess, FLAGS.pretrained_model)

    def train(self, inputs, lr, mask_true):
        feed_dict = {self.x: inputs}
        feed_dict.update({self.tf_lr: lr})
        feed_dict.update({self.mask_true: mask_true})
        loss, _ = self.sess.run((self.loss_train, self.train_op), feed_dict)
        return loss

    def test(self, inputs, mask_true):
        feed_dict = {self.x: inputs}
        feed_dict.update({self.mask_true: mask_true})
        gen_ims = self.sess.run(self.pred_seq, feed_dict)
        return gen_ims

    def save(self, itr):
        checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=itr)
        print('saved to ' + FLAGS.save_dir)
# #0326/한승현 save directory for the TESNSORBOARD results
# log_dir = '/home/work/log/mnist_predrnn_pp/'
# os.makedirs(log_dir, exist_ok=True)
# numeric_dir = '/home/work/outs/mnist_predrnn_pp/'
#0326/한승현 create a function to save the results to a csv file
def save_results_to_text_file(file_path, metrics_names, metrics_values):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(metrics_names)
        writer.writerow(metrics_values)

def main(argv=None):
    if tf.gfile.Exists(FLAGS.save_dir):
        tf.gfile.DeleteRecursively(FLAGS.save_dir)
    tf.gfile.MakeDirs(FLAGS.save_dir)
    if tf.gfile.Exists(FLAGS.gen_frm_dir):
        tf.gfile.DeleteRecursively(FLAGS.gen_frm_dir)
    tf.gfile.MakeDirs(FLAGS.gen_frm_dir)

    train_input_handle, test_input_handle = datasets_factory.data_provider(
        FLAGS.dataset_name, FLAGS.train_data_paths, FLAGS.valid_data_paths,
        FLAGS.batch_size, FLAGS.img_width, FLAGS.img_height)

    print("Initializing models")
    model = Model(train_input_handle)
    lr = FLAGS.lr

    delta = 0.00002
    base = 0.99998
    eta = 1
    writer = tf.compat.v1.summary.FileWriter(log_dir)
    writer.add_graph(tf.compat.v1.get_default_graph())
    
    for itr in range(1, FLAGS.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
        ims = train_input_handle.get_batch()
        ims = preprocess.reshape_patch(ims, FLAGS.patch_size)

        if itr < 50000:
            eta -= delta
        else:
            eta = 0.0
        random_flip = np.random.random_sample(
            (FLAGS.batch_size, train_input_handle.current_output_length - 1))
        true_token = (random_flip < eta)
        ones = np.ones((FLAGS.img_width // FLAGS.patch_size,
                        FLAGS.img_width // FLAGS.patch_size,
                        FLAGS.patch_size ** 2 * FLAGS.img_channel))
        zeros = np.zeros((FLAGS.img_width // FLAGS.patch_size,
                          FLAGS.img_width // FLAGS.patch_size,
                          FLAGS.patch_size ** 2 * FLAGS.img_channel))
        mask_true = []
        for i in range(FLAGS.batch_size):
            for j in range(train_input_handle.current_output_length - 1):
                if true_token[i, j]:
                    mask_true.append(ones)
                else:
                    mask_true.append(zeros)
        mask_true = np.array(mask_true)
        mask_true = np.reshape(mask_true, (FLAGS.batch_size,
                                           train_input_handle.current_output_length - 1,
                                           FLAGS.img_width // FLAGS.patch_size,
                                           FLAGS.img_width // FLAGS.patch_size,
                                           FLAGS.patch_size ** 2 * FLAGS.img_channel))
        cost = model.train(ims, lr, mask_true)
        if FLAGS.reverse_input:
            ims_rev = ims[:, ::-1]
            cost += model.train(ims_rev, lr, mask_true)
            cost = cost / 2

        if itr % FLAGS.display_interval == 0:
            print('itr: ' + str(itr))
            print('training loss: ' + str(cost))
            summary = tf.compat.v1.Summary()
            summary.value.add(tag='Training Loss', simple_value=float(cost))
            writer.add_summary(summary, itr)

        if itr % FLAGS.test_interval == 0:
            print('test...')
            test_input_handle.begin(do_shuffle=False)
            res_path = os.path.join(FLAGS.gen_frm_dir, str(itr))
            os.mkdir(res_path)
            avg_mse = 0
            batch_id = 0
            img_mse, ssim, psnr, fmae, sharp = [], [], [], [], []
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                img_mse.append(0)
                ssim.append(0)
                psnr.append(0)
                fmae.append(0)
                sharp.append(0)
            mask_true = np.zeros((FLAGS.batch_size,
                                  FLAGS.seq_lenth-FLAGS.input_length - 1,
                                  FLAGS.img_width // FLAGS.patch_size,
                                  FLAGS.img_height // FLAGS.patch_size,
                                  FLAGS.patch_size ** 2 * FLAGS.img_channel))
            while (test_input_handle.no_batch_left() == False):
                batch_id = batch_id + 1
                test_ims = test_input_handle.get_batch()
                test_dat = preprocess.reshape_patch(test_ims, FLAGS.patch_size)
                img_gen = model.test(test_dat, mask_true)

                img_gen = np.concatenate(img_gen)
                img_gen = preprocess.reshape_patch_back(img_gen, FLAGS.patch_size)
                for i in range(FLAGS.seq_length - FLAGS.input_length):
                    x = test_ims[:, i + FLAGS.input_length, :, :, 0]
                    gx = img_gen[:, i, :, :, 0]
                    fmae[i] += metrics.batch_mae_frame_float(gx, x)
                    gx = np.maximum(gx, 0)
                    gx = np.minimum(gx, 1)
                    mse = np.square(x - gx).sum()
                    img_mse[i] += mse
                    avg_mse += mse

                    real_frm = np.uint8(x * 255)
                    pred_frm = np.uint8(gx * 255)
                    psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
                    for b in range(FLAGS.batch_size):
                        sharp[i] += np.max(
                            cv2.convertScaleAbs(cv2.Laplacian(pred_frm[b], 3)))
                        score, _ = structural_similarity(pred_frm[b], real_frm[b], full=True)
                        ssim[i] += score

                if batch_id <= 10:
                    path = os.path.join(res_path, str(batch_id))
                    os.mkdir(path)
                    for i in range(FLAGS.seq_length):
                        name = 'gt' + str(i+1) + '.png'
                        file_name = os.path.join(path, name)
                        img_gt = np.uint8(test_ims[0,i,:,:,:] * 255)
                        cv2.imwrite(file_name, img_gt)
                    for i in range(FLAGS.seq_length-FLAGS.input_length):
                        name = 'pd' + str(i+1+FLAGS.input_length) + '.png'
                        file_name = os.path.join(path, name)
                        img_pd = img_gen[0,i,:,:,:]
                        img_pd = np.maximum(img_pd, 0)
                        img_pd = np.minimum(img_pd, 1)
                        img_pd = np.uint8(img_pd * 255)
                        cv2.imwrite(file_name, img_pd)
                test_input_handle.next()
            avg_mse = avg_mse // (batch_id * FLAGS.batch_size)
            print('mse per seq: ' + str(avg_mse))
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(img_mse[i] // (batch_id * FLAGS.batch_size))
            psnr = np.asarray(psnr, dtype=np.float32) // batch_id
            fmae = np.asarray(fmae, dtype=np.float32) // batch_id
            ssim = np.asarray(ssim, dtype=np.float32) // (FLAGS.batch_size * batch_id)
            sharp = np.asarray(sharp, dtype=np.float32) // (FLAGS.batch_size * batch_id)
            print('psnr per frame: ' + str(np.mean(psnr)))
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(psnr[i])
            print('fmae per frame: ' + str(np.mean(fmae)))
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(fmae[i])
            print('ssim per frame: ' + str(np.mean(ssim)))
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(ssim[i])
            print('sharpness per frame: ' + str(np.mean(sharp)))
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(sharp[i])
            os.makedirs(numeric_dir, exist_ok=True)
            file_name = f'results_itr_{itr}.csv'
            file_path = os.path.join(numeric_dir, file_name)
            metrics_names = ['PSNR', 'FMAE', 'SSIM', 'Sharpness']
            metrics_values = [psnr.tolist(), fmae.tolist(), ssim.tolist(), sharp.tolist()]
            save_results_to_text_file(file_path, metrics_names, metrics_values)

        if itr % FLAGS.snapshot_interval == 0:
            model.save(itr)

        train_input_handle.next()

if __name__ == '__main__':
    tf.app.run()
