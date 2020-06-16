from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import namedtuple
import random
from module import *
from utils import *
import utils

import cv2

class vae(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        #self.image_size = args.fine_size
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir
        self.alpha = args.alpha

        self.network = network
        self.Angular_Softmax_Loss = Angular_Softmax_Loss

        self._build_model(args)
        
        self.saver = tf.train.Saver(max_to_keep=100)
        
        if args.phase == 'train':
            self.ds, self.other = self._load_dataset('D:/Dataset/CAM11_12_NEW_for_cutmix/')
            
        
    def _load_dataset(self, path):
        class_name_list = {'fanout':0,'impedance':1}
        #class_name_list = {'fanout':0,'impedance':1, 'others':2}
        class_list = os.listdir(path)
        img_list = []
        label_list = []
        other_list = []
        rand_label_list = []
        for i in range(len(class_list)):
            if class_list[i] in class_name_list:
                class_path = path+class_list[i]
                temp_img_list = [class_path + '/' + j for j in os.listdir(class_path)]
                temp_label_list = list(np.zeros(len(temp_img_list)) + class_name_list[class_list[i]])
                img_list.extend(temp_img_list)
                label_list.extend(temp_label_list)
            else:
                temp_other_list = [class_path + '/' + j for j in os.listdir(class_path)]
                temp_rand_label_list = np.random.randint(0,2,len(temp_other_list))
                other_list.extend(temp_other_list)
                rand_label_list.extend(temp_rand_label_list)

        print(np.shape(img_list))
        print(np.shape(label_list))
        data_list = list(zip(img_list, label_list))
        other_list = list(zip(other_list, rand_label_list))
        random.shuffle(data_list)
        random.shuffle(other_list)
        return data_list, other_list

    def _load_batch(self, dataset, idx):
        
        filename_list = dataset[idx * self.batch_size:(idx + 1) * self.batch_size]

        # input batch (2d binary image)
        input_batch = []
        target_batch = []
        for i in range(len(filename_list)):
            temp_img = cv2.imread(filename_list[i][0], 0)
            temp_img = temp_img/128 - 1
            #temp_img = tf.dtypes.cast(temp_img, tf.float32)
            input_batch.append(temp_img)
            temp_target = filename_list[i][1]
            #temp_target = tf.dtypes.cast(temp_target, tf.int64)
            target_batch.append(temp_target)
        

        input_batch = np.array(input_batch, dtype=np.float32)
        target_batch = np.array(target_batch, dtype=np.int64)
        input_batch = np.expand_dims(input_batch, axis=3)

        return input_batch, target_batch, filename_list


    def _build_model(self, args):


        self.input = tf.placeholder(tf.float32, [None, 128, 128, 1], name='input')
        self.label = tf.placeholder(tf.int64, [None,], name='label')

        # labeled data sequence
        self.embeddings = self.network(self.input)
        self.pred_prob, self.loss, self.cos_theta, self.orgina_logits = Angular_Softmax_Loss(self.embeddings, self.label)

        
        self.loss_summary = tf.summary.scalar("loss", self.loss)

        self.t_vars = tf.trainable_variables()
        print("trainable variables : ")
        print(self.t_vars)
        

    def train(self, args):
        
        global_step = tf.Variable(0, trainable=False)        
        decay_lr = tf.train.exponential_decay(args.lr, global_step, 500, 0.9)
        optimizer = tf.train.AdamOptimizer(decay_lr)
        train_op = optimizer.minimize(self.loss)


        print("initialize")
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(args.checkpoint_dir): 
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):

            random.shuffle(self.ds)
            random.shuffle(self.other)
            
            batch_idxs = len(self.ds) // self.batch_size

            #ds_1 = self.ds.sample(frac=1)
            
            for idx in range(0, batch_idxs):

                input_batch, target_batch, _ = self._load_batch(self.ds, idx)
                

                test_input_batch, test_target_batch, _ = self._load_batch(self.other, idx)

                # Update network
                cos_theta, loss, _ = self.sess.run([self.cos_theta, self.loss, train_op], feed_dict={self.input: input_batch, self.label: target_batch})

                #self.writer.add_summary(summary_str, counter)

                counter += 1
                if idx%10==0:
                    print(("Epoch: [%2d] [%4d/%4d] time: %4.4f loss: %4.4f cos(theta): %4.4f" % (
                        #epoch, idx, batch_idxs, time.time() - start_time, loss, np.sum(cos_theta[0]))))
                        epoch, idx, batch_idxs, time.time() - start_time, loss, cos_theta[0])))

                if idx%100==0:
                    print("OTHERS TEST")

                    cos_theta_t, loss_t, origina_logits_t = self.sess.run([self.cos_theta, self.loss, self.orgina_logits], feed_dict={self.input: test_input_batch, self.label: test_target_batch})
                    rand_int_t = int(np.random.randint(10))
                    print(("Epoch: [%2d] [%4d/%4d] time: %4.4f loss: %4.4f cos(theta): %4.4f %4.4f" % (
                        epoch, idx, batch_idxs, time.time() - start_time, loss_t, cos_theta_t[rand_int_t], test_target_batch[rand_int_t])))
                    #epoch, idx, batch_idxs, time.time() - start_time, loss, np.sum(cos_theta_t[int(np.random.randint(10))]))))

                    print(origina_logits_t)
                    print(np.shape(origina_logits_t))
                    print(rand_int_t)

                if np.mod(counter, args.save_freq) == 20:
                    self.save(args.checkpoint_dir, counter)


    def save(self, checkpoint_dir, step):
        model_name = "dnn.model"
        model_dir = "%s" % (self.dataset_dir)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s" % (self.dataset_dir)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt)
            ckpt_paths = ckpt.all_model_checkpoint_paths    #hcw
            print(ckpt_paths)
            #ckpt_name = os.path.basename(ckpt_paths[-1])    #hcw # default [-1]
            temp_ckpt = 'dnn.model-80520'
            ckpt_name = os.path.basename(temp_ckpt)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


    def test(self, args):

        start_time = time.time()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        counter = 0


        if not os.path.exists(os.path.join(args.test_dir,'input')):
            os.mkdir(os.path.join(args.test_dir,'input'))


        batch_idxs = len(self.ds) // self.batch_size

        #ds_1 = self.ds.sample(frac=1)
        ds_1 = self.ds
        #print(ds_1.iloc[:,4:][0:10].values.tolist())
        
        loss_list = []

        df_param_target_all = pd.DataFrame()
        df_param_pred_all = pd.DataFrame()

        for idx in range(0, batch_idxs):

            input_batch, target_batch, _ = self._load_batch(ds_1, idx)

            geo_pred, pred, loss = self.sess.run([self.geo_reconstructed_l, self.spectra_l_predicted, self.total_loss],
                                                feed_dict={self.geo_labeled: input_batch, self.spectrum_target: target_batch})


            loss_list.append(loss)

            counter += 1
            if idx%1==0:
                print(("Step: [%4d/%4d] time: %4.4f" % (
                    idx, batch_idxs, time.time() - start_time)))
                #df_param = pd.DataFrame(np.squeeze(input_batch), columns={'param1','param2','param3','param4','param5'}) 
                df_pred = pd.DataFrame(np.squeeze(pred))
                df_target = pd.DataFrame(np.squeeze(target_batch))
                #df_geo_pred =  np.squeeze(geo_pred)

                #df_param_pred = pd.concat([df_param, df_pred], axis=1, sort=False)
                #df_param_target = pd.concat([df_param, df_target], axis=1, sort=False)
                #df_param_param = pd.concat([df_param, df_geo_pred], axis=1, sort=False)
                
                df_param_target_all = pd.concat([df_param_target_all, df_target], axis=0, sort=False)
                df_param_pred_all = pd.concat([df_param_pred_all, df_pred], axis=0, sort=False)


            df_param_target_all.to_csv('./test/result_test_target.csv', index=False)
            df_param_pred_all.to_csv('./test/result_test_prediction.csv', index=False)

            #print(np.shape(geo_pred))
            geo_pred = np.squeeze(geo_pred)
            #print(geo_pred)
            #cv2.imwrite('./test/reconstructed/test'+str(idx)+'.bmp',(geo_pred+1)*128)
            
        print("loss")
        print(np.mean(loss_list))
        print("total time")
        print(time.time() - start_time)


    def test_reconstruction(self, args):

        self.batch_size = 1

        start_time = time.time()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        counter = 0


        if not os.path.exists(os.path.join(args.test_dir,'input')):
            os.mkdir(os.path.join(args.test_dir,'input'))


        batch_idxs = len(self.ds) // self.batch_size

        #ds_1 = self.ds.sample(frac=1)
        ds_1 = self.ds
        #print(ds_1.iloc[:,4:][0:10].values.tolist())
        
        loss_list = []

        for idx in range(0, batch_idxs):

            input_batch, target_batch, filename_list = self._load_batch(ds_1, idx)

            for j in range(5):
                latent_vector = list(np.random.normal(0,3,5))
                #for k in range(5):
                #    latent_vector[k] = j*0.5 - 2.5
                print(latent_vector)
                latent_vector = np.expand_dims(latent_vector, 0)
                geo_recon = self.sess.run([self.geo_reconstructed], 
                                            feed_dict={self.latent_vector: latent_vector, self.spectrum_target: target_batch})


                #print(np.shape(geo_pred))
                geo_recon = np.squeeze(geo_recon)
                cv2.imwrite('./test/reconstruction/'+str(filename_list)+'_'+str(latent_vector)+'.bmp',(geo_recon+1)*128)
            