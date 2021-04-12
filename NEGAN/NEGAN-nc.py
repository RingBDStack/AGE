# -*- coding: utf-8 -*-

import warnings
import logging
logging.basicConfig(level=logging.INFO, filename='./nc-log.log', filemode='w')
warnings.filterwarnings('ignore')


class Config():
    def __init__(self):
        self.name = '1'
        self.pretrain_name = ''
        
        self.g_batch_size = 64
        self.d_batch_size = 64
        self.lambda_gen = 1e-5
        self.lambda_dis = 1e-5
        self.lr_gen = 0.0001#1e-3
        self.lr_dis = 0.0001#1e-4
        self.n_epoch = 70
        self.sig = 1.0
        self.label_smooth = 0.0
        self.d_epoch = 15
        self.g_epoch = 15
        self.n_emb = 64
        self.pre_d_epoch = 0
        self.pre_g_epoch = 0
        self.neg_weight = [1, 1]
        self.walk_num = 10
        self.walk_len = 10
        self.adv = False
        self.rand = False
        
        self.dataset = 'cora'
        self.experiment = 'link_prediction'
        self.train_filename = '../data/'
        self.test_filename = '../data/'
        self.test_tp = 0
        self.last_ckpt_dir = ''
        # last_ckpt_dir = '../results/twitter/1/'
        self.dis_pretrain_node_emb_filenames = []
        self.gen_pretrain_node_emb_filenames = []
        self.save = False
        self.save_path = '../results/%s/%s/' % (self.dataset, self.name)
        self.verbose = 1
        self.log = True
config = ''


import numpy as np

class Utils():
    def read_graph(self, train_filename):

        # Get Dataset
        filename_list = train_filename.split('/')
        dataset = filename_list[3]
        nodes_dict = { 'cora': 2708, 'citeseer': 3327, 'facebook': 6637 }

        nodes = set()
        egs = []
        graph = {}

        with open(train_filename) as infile:
            for line in infile.readlines():
                line = line.strip()
                u, v = line.split(' ')
                u = int(u)
                v = int(v)

                nodes.add(u)
                nodes.add(v)
                egs.append([u, v])
                egs.append([v, u])

                if u not in graph:
                    graph[u] = []
                if v not in graph:
                    graph[v] = []

                graph[u].append(v)
                graph[v].append(u)

        # n_node = max(list(nodes)) + 1
        n_node = nodes_dict[dataset]
        print(n_node)
        return graph, n_node, list(nodes), egs

    def str_list_to_float(self, str_list):
        return [float(item) for item in str_list]

    def read_embeddings(self, filename, n_node, n_embed):
        embedding_matrix = np.random.rand(n_node, n_embed)
        i = -1
        with open(filename) as infile:
            for line in infile.readlines()[1:]:
                i += 1
                emd = line.strip().split()
                embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
        return embedding_matrix
    
    def log(self, info):
        if config.log == False:
            return
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        log_path = '%slog.txt' % config.save_path
        with open(log_path, 'a') as log_file:
            log_file.write(str(info))



'''Evaluate models'''
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import math
from collections import defaultdict
from scipy.stats import hmean
from sklearn.model_selection import cross_val_score
import nmslib
from annoy import AnnoyIndex


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LinkPrediction():
    def __init__(self):
        self.links = []
        with open(config.test_filename) as infile:
            for line in infile.readlines():
                u, v, label = [int(item) for item in line.strip().split()]
                self.links.append([u, v, label])

    def evaluate(self, embedding_matrix):
        test_y = []
        pred_y = []
        pred_label = []
        for u, v, label in self.links:
            test_y.append(label)
            pred_y.append(embedding_matrix[u].dot(embedding_matrix[v]))
            if pred_y[-1] >= 0:
                pred_label.append(1)
            else:
                pred_label.append(0)
                
        auc = roc_auc_score(test_y, pred_y)
        return auc

class NodeClassification():
    def __init__(self):
        self.node_label = {}
        with open(config.test_filename) as infile:
            for line in infile.readlines():
                line = line.strip()
                line = line.split(',')
                s = int(line[0])
                label = int(line[1])
                self.node_label[s] = label   

    def evaluate(self, embedding_matrix):
        embedding_list = embedding_matrix.tolist()
        X = []
        Y = []
        for s in self.node_label:
            X.append(embedding_list[s] + embedding_list[s])
            Y.append(self.node_label[s])

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        lr = LogisticRegression()
        lr.fit(X_train, Y_train)

        Y_pred = lr.predict(X_test)
        micro_f1 = f1_score(Y_test, Y_pred, average='micro')
        macro_f1 = f1_score(Y_test, Y_pred, average='macro')
        return micro_f1.tolist(), macro_f1.tolist()



import tensorflow as tf

class Generator():
    def __init__(self, n_node, node_emd_init):
        self.n_node = n_node
        self.emd_dim = config.n_emb
        self.node_emd_init = node_emd_init

        #with tf.variable_scope('generator'):
        if node_emd_init:
            self.node_embedding_matrix = tf.get_variable(name = 'gen_node_embedding',
                                                       shape = self.node_emd_init.shape,
                                                       initializer = tf.constant_initializer(self.node_emd_init),
                                                       trainable = True)
        else:
            self.node_embedding_matrix = tf.get_variable(name = 'gen_node_embedding',
                                                       shape = [self.n_node, self.emd_dim],
                                                       initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                                       trainable = True)

        self.gen_w_1 = tf.get_variable(name = 'gen_w',
                                       shape = [self.emd_dim, self.emd_dim],
                                       initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                       trainable = True)
        self.gen_b_1 = tf.get_variable(name = 'gen_b',
                                       shape = [self.emd_dim],
                                       initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                       trainable = True)
        self.gen_w_2 = tf.get_variable(name = 'gen_w_2',
                                       shape = [self.emd_dim, self.emd_dim],
                                       initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                       trainable = True)
        self.gen_b_2 = tf.get_variable(name = 'gen_b_2',
                                       shape = [self.emd_dim],
                                       initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                       trainable = True)
        #self.bias_vector = tf.Variable(tf.zeros([self.n_node]))

        self.node_ids = tf.placeholder(tf.int32, shape = [None])

        self.noise_embedding = tf.placeholder(tf.float32, shape = [None, self.emd_dim])

        self.dis_node_embedding = tf.placeholder(tf.float32, shape = [None, self.emd_dim])
        
        self.node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, self.node_ids)

        
        self.node_fake_embedding = self.generate_node(self.node_embedding, self.noise_embedding)

        self.score = tf.reduce_sum(tf.multiply(self.dis_node_embedding, self.node_fake_embedding), axis = 1)

        self.neg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(                        labels=tf.ones_like(self.score) * (1.0 - config.label_smooth), logits=self.score))                         + config.lambda_gen * (tf.nn.l2_loss(self.node_embedding) + tf.nn.l2_loss(self.gen_w_1))

        self.loss = self.neg_loss

        optimizer = tf.train.AdamOptimizer(config.lr_gen)
        #optimizer = tf.train.RMSPropOptimizer(config.lr_gen)
        self.g_updates = optimizer.minimize(self.loss)

    def generate_node(self, node_embedding, noise_embedding):
        #node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, node_id)
        #relation_embedding = tf.nn.embedding_lookup(self.relation_embedding_matrix, relation_id)

        input = tf.reshape(node_embedding, [-1, self.emd_dim])
        #input = tf.concat([input, noise_embedding], axis = 1)
        input = input + noise_embedding

        output = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_1) + self.gen_b_1)
        #input = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_1) + self.gen_b_1)# +  relation_embedding
        #output = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_2) + self.gen_b_2)
        #output = node_embedding + relation_embedding + noise_embedding

        return output




import tensorflow as tf

class Discriminator():
    def __init__(self, n_node, node_emd_init):
        self.n_node = n_node
        self.emd_dim = config.n_emb
        self.node_emd_init = node_emd_init

        #with tf.variable_scope('disciminator'):
        if node_emd_init:
            self.node_embedding_matrix = tf.get_variable(name = 'dis_node_embedding',
                                                         shape = self.node_emd_init.shape,
                                                         initializer = tf.constant_initializer(self.node_emd_init),
                                                         trainable = True)
        else:
            self.node_embedding_matrix = tf.get_variable(name = 'dis_node_embedding',
                                                         shape = [self.n_node, self .emd_dim],
                                                         initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                                         trainable = True)

        self.pos_node_ids = tf.placeholder(tf.int32, shape = [None])

        self.pos_node_neighbor_ids = tf.placeholder(tf.int32, shape = [None])
        
        self.neg_node_neighbor_ids = tf.placeholder(tf.int32, shape = [None])

        self.node_fake_embedding = tf.placeholder(tf.float32, shape = [None, self.emd_dim])
        
        self.pos_node_embedding = tf.reshape(tf.nn.embedding_lookup(self.node_embedding_matrix, self.pos_node_ids), [-1, self.emd_dim])
        self.pos_node_neighbor_embedding = tf.reshape(tf.nn.embedding_lookup(self.node_embedding_matrix, self.pos_node_neighbor_ids),
                                                      [-1, self.emd_dim])

        self.neg_node_neighbor_embedding = tf.reshape(tf.nn.embedding_lookup(self.node_embedding_matrix, self.neg_node_neighbor_ids),
                                                      [-1, self.emd_dim])

        self.pos_score = tf.reduce_sum(tf.multiply(self.pos_node_embedding, self.pos_node_neighbor_embedding), axis = 1)
        self.pos_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.pos_score),
                                                                               logits=self.pos_score))
        
        self.neg_loss = [0, 0]
        node_emb = [self.neg_node_neighbor_embedding, self.node_fake_embedding]
        for i in range(2):
            _neg_score = tf.reduce_sum(tf.multiply(self.pos_node_embedding, node_emb[i]), axis = 1)
            self.neg_loss[i] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(_neg_score),
                                                                                   logits=_neg_score))

        self.loss = self.pos_loss
        if config.adv:
            for i in range(2):
                self.loss += self.neg_loss[i] * config.neg_weight[i]
        else:
            self.loss += self.neg_loss[0] * config.neg_weight[0]

        optimizer = tf.train.AdamOptimizer(config.lr_dis)
        #optimizer = tf.train.GradientDescentOptimizer(config.lr_dis)
        #optimizer = tf.train.RMSPropOptimizer(config.lr_dis)
        self.d_updates = optimizer.minimize(self.loss)
        #self.reward = tf.log(1 + tf.exp(tf.clip_by_value(self.score, clip_value_min=-10, clip_value_max=10)))




import os
import tensorflow as tf
import time
import numpy as np
import random


class Model():
    def __init__(self):
        t = time.time()
#         print "reading graph..."
        self.utils = Utils()
        self.graph, self.n_node, self.node_list, self.egs = self.utils.read_graph(config.train_filename)
        self.node_emd_shape = [self.n_node, config.n_emb]
#         print '[%.2f] reading graph finished. #node = %d' % (time.time() - t, self.n_node)

        self.dis_node_embed_init = None
        self.gen_node_embed_init = None
        if config.pretrain_name:
            t = time.time()
#             print "read initial embeddings..."
            dis_node_embed_init = np.array([utils.read_embeddings(filename=x, n_node=self.n_node, n_embed=config.n_emb)                                             for x in [config.dis_pretrain_node_emb_filenames]])
            gen_node_embed_init = np.array([utils.read_embeddings(filename=x, n_node=self.n_node, n_embed=config.n_emb)                                             for x in [config.gen_pretrain_node_emb_filenames]])
#             print "[%.2f] read initial embeddings finished." % (time.time() - t)

#         print "build GAN model..."
        self.discriminator = None
        self.generator = None
        self.build_generator()
        self.build_discriminator()
        if config.experiment == 'lp':
            self.link_prediction = LinkPrediction()
        if config.experiment == 'gr':
            self.graph_reconstruction = GraphReconstruction()
        if config.experiment == 'gr2':
            self.graph_reconstruction = GraphReconstruction2()
        if config.experiment == 'nr':
            self.node_recommendation = NodeRecommendation2()
        if config.experiment == 'nc':
            self.node_classification = NodeClassification()

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = self.config)
        self.saver = tf.train.Saver(max_to_keep=0)
        if config.last_ckpt_dir:
#             print 'restore...'
            latest_checkpoint = tf.train.latest_checkpoint(config.last_ckpt_dir)
            self.saver.restore(self.sess, latest_checkpoint)
        else:
#             print 'initial...'
            self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self.sess.run(self.init_op)

    def build_discriminator(self):
        #with tf.variable_scope("discriminator"):
        self.discriminator = Discriminator(n_node = self.n_node,
                                                         node_emd_init = self.dis_node_embed_init)
    def build_generator(self):
        #with tf.variable_scope("generator"):
        self.generator = Generator(n_node = self.n_node,
                                             node_emd_init = self.gen_node_embed_init)

    def train_dis(self, dis_loss, pos_loss, neg_loss, dis_cnt):
        np.random.shuffle(self.node_list)
        
        info = ''
        for index in range(math.floor(len(self.node_list) / config.d_batch_size)):
            #t1 = time.time()
            pos_node_ids, pos_node_neighbor_ids, neg_node_neighbor_ids, node_fake_embedding = self.prepare_data_for_d(index, self.node_list)
            #t2 = time.time()
            #print t2 - t1
            _, _loss, _pos_loss, _neg_loss = self.sess.run([self.discriminator.d_updates, self.discriminator.loss,                                             self.discriminator.pos_loss, self.discriminator.neg_loss],
                                            feed_dict = {self.discriminator.pos_node_ids : np.array(pos_node_ids),
                                                         self.discriminator.pos_node_neighbor_ids : np.array(pos_node_neighbor_ids),
                                                         self.discriminator.neg_node_neighbor_ids : np.array(neg_node_neighbor_ids),
                                                         self.discriminator.node_fake_embedding : np.array(node_fake_embedding)})

            dis_loss += _loss
            pos_loss += _pos_loss
            neg_loss[0] += _neg_loss[0]
            neg_loss[1] += _neg_loss[1]
            dis_cnt += 1
            # print '\r%s' % (' ' * 150),
            info = 'dis_loss=%.4f pos_loss=%.4f neg_loss0=%.4f neg_loss1=%.4f' %                 (dis_loss / dis_cnt, pos_loss / dis_cnt, neg_loss[0] / dis_cnt, neg_loss[1] / dis_cnt)
            self.my_print(info, True, 1)
        self.utils.log(info + '\n')
        return (dis_loss, pos_loss, neg_loss, dis_cnt)

    def train_gen(self, gen_loss, neg_loss, gen_cnt):
        np.random.shuffle(self.node_list)
        
        info = ''
        for index in range(math.floor(len(self.node_list) / config.g_batch_size)):
            node_ids, noise_embedding, dis_node_embedding = self.prepare_data_for_g(index, self.node_list)
            _, _loss, _neg_loss = self.sess.run([self.generator.g_updates, self.generator.loss, self.generator.neg_loss],
                                                 feed_dict = {self.generator.node_ids : np.array(node_ids),
                                                              self.generator.noise_embedding : np.array(noise_embedding),
                                                              self.generator.dis_node_embedding : np.array(dis_node_embedding)})

            gen_loss += _loss
            neg_loss += _neg_loss
            gen_cnt += 1
            # print '\r%s' % (' ' * 150),
            info = 'gen_loss=%.4f neg_loss=%.4f' % (gen_loss / gen_cnt, neg_loss / gen_cnt)
            self.my_print(info, True, 1)
        self.utils.log(info + '\n')
        return (gen_loss, neg_loss, gen_cnt)

    def train(self):
        best_auc = [[0], [0], [0]]
        best_epoch = [-1, -1, -1]

        self.my_print('start traning...', False, 1)
        for epoch in range(config.n_epoch):
            info = 'epoch %d' % epoch
            self.my_print(info, False, 1)
            self.utils.log(info + '\n')
            t = time.time()

            dis_loss = 0.0
            dis_pos_loss = 0.0
            dis_neg_loss = [0.0, 0.0]
#             sim_loss = [0.0, 0.0]
            dis_cnt = 0

            gen_loss = 0.0
            gen_neg_loss = 0.0
            gen_cnt = 0

            #D-step
            #t1 = time.time()
            for d_epoch in range(config.d_epoch):
                dis_loss, dis_pos_loss, dis_neg_loss, dis_cnt = self.train_dis(dis_loss, dis_pos_loss, dis_neg_loss, dis_cnt)
                self.my_print('', False, 1)
#             self.evaluate_node_classification()
#                 auc = self.evaluate_link_prediction()
#             scores = self.evaluate_graph_reconstruction()
#                 if d_epoch and d_epoch % 3 == 0:
                auc = self.evaluate()
                for i in range(len(auc)):
                    if np.mean(auc[i]) > np.mean(best_auc[i]):
                        best_auc[i] = auc
                        best_epoch[i] = epoch
            
            #G-step
            if config.adv:
                for g_epoch in range(config.g_epoch):
                    gen_loss, gen_neg_loss, gen_cnt = self.train_gen(gen_loss, gen_neg_loss, gen_cnt)
                    self.my_print('', False, 1)

            if config.save:
                self.write_embeddings_to_file(epoch)
            #os.system('python ../evaluation/lp_evaluation_2.py')
#         print "training completes"
        return (best_auc, best_epoch)

    def random_walk(self, s):
        walk = []
        p = s
        while len(walk) < config.walk_len:
            if p not in self.graph or len(self.graph[p]) == 0:
                break
            p = random.choice(self.graph[p])
            walk.append(p)
        return walk

    def prepare_data_for_d(self, index, node_list):
        pos_node_ids = []
        pos_node_neighbor_ids = []
        neg_node_neighbor_ids = []

        for node_id in node_list[index * config.d_batch_size : (index + 1) * config.d_batch_size]:
            for k in range(config.walk_num):
                walk = self.random_walk(node_id)
                for t in walk:
                    pos_node_ids.append(node_id)
                    pos_node_neighbor_ids.append(t)
                    neg = random.choice(node_list)
                    neg_node_neighbor_ids.append(neg)

        # generate fake node
        noise_embedding = np.random.normal(0.0, config.sig, (len(pos_node_ids), config.n_emb))
        if config.adv:
            node_fake_embedding = self.sess.run(self.generator.node_fake_embedding,
                                                feed_dict = {self.generator.node_ids : np.array(pos_node_ids),
                                                             self.generator.noise_embedding : np.array(noise_embedding)})
        else:
            node_fake_embedding = noise_embedding

#         print(np.array(node_fake_embedding).shape)
        return pos_node_ids, pos_node_neighbor_ids, neg_node_neighbor_ids, node_fake_embedding

    def prepare_data_for_g(self, index, node_list):
        node_ids = []

        for node_id in node_list[index * config.g_batch_size : (index + 1) * config.g_batch_size]:
            # n_sample = min(self.graph[node_id][1], config.n_sample_max)
#             n_sample = config.n_sample_max
#             for i in range(n_sample):
            node_ids.append(node_id)

        noise_embedding = np.random.normal(0.0, config.sig, (len(node_ids), config.n_emb))
        
        dis_node_embedding = self.sess.run(self.discriminator.pos_node_embedding,
                                            feed_dict = {self.discriminator.pos_node_ids : np.array(node_ids)})
        return node_ids, noise_embedding, dis_node_embedding
    
    def evaluate(self):
        if config.experiment == 'lp':
            return self.evaluate_link_prediction()
        if config.experiment == 'gr':
            return self.evaluate_graph_reconstruction()
        if config.experiment == 'gr2':
            return self.evaluate_graph_reconstruction()
        if config.experiment == 'nr':
            return self.evaluate_node_recommendation()
        if config.experiment == 'nc':
            return self.evaluate_node_classification()
    
    def evaluate_link_prediction(self):
        embedding_matrix = self.sess.run(self.discriminator.node_embedding_matrix)
        auc = self.link_prediction.evaluate(embedding_matrix)
        info = 'auc=%.4f' % auc
        self.my_print(info, False, 1)
        self.utils.log(info + '\n')
        return auc
    
    def evaluate_graph_reconstruction(self):
        embedding_matrix = self.sess.run(self.discriminator.node_embedding_matrix)
        scores = self.graph_reconstruction.evaluate(embedding_matrix)
        info = []
        for score in scores:
            info.append('@%d=%.4f,%.4f' % (score[0], score[1], score[2]))
        info = '\n'.join(info)
        self.my_print(info, False, 1)
        self.utils.log(info + '\n')
        return scores
    
    def evaluate_node_recommendation(self):
        embedding_matrix = self.sess.run(self.discriminator.node_embedding_matrix)
        aucs = self.node_recommendation.evaluate(embedding_matrix)
        info = []
        for auc in aucs:
            info.append('@%d=%.4f,%.4f' % (auc[0], auc[1], auc[2]))
        info = '\n'.join(info)
#         for auc in aucs:
#             info.append('@%d,A=%.4f,P=%.4f,R=%.4f,F=%.4f' % (auc[0], auc[1], auc[2], auc[3], auc[4]))
#         info = '\n'.join(info)
        self.my_print(info, False, 1)
        self.utils.log(info + '\n')
        return aucs
    
    def evaluate_node_classification(self):
        embedding_matrix = self.sess.run(self.discriminator.node_embedding_matrix)
        micro_f1, macro_f1 = self.node_classification.evaluate(embedding_matrix)
        micro_f1 = [round(item, 6) for item in micro_f1]
        macro_f1 = [round(item, 6) for item in macro_f1]
        logging.info('micro-f1={} macro_f1={}'.format(micro_f1, macro_f1))
        info = 'micro-f1={} macro_f1={}'.format(micro_f1, macro_f1)
        self.my_print(info, False, 1)
        self.utils.log(info + '\n')
        return [micro_f1, macro_f1]

    def write_embeddings_to_file(self, epoch):
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        models = [self.generator, self.discriminator]
        emb_filenames = ['gen.emb', 'dis_s.emb', 'dis_t.emb']
        embedding_matrix = [self.sess.run(self.generator.node_embedding_matrix)]
        embedding_matrix.extend([self.sess.run(self.discriminator.node_embedding_matrix)[0],
                                 self.sess.run(self.discriminator.node_embedding_matrix)[1]])
        for i in range(3):
            index = np.array(range(self.n_node)).reshape(-1, 1)
            t = np.hstack([index, embedding_matrix[i]])
            embedding_list = t.tolist()
            embedding_str = [str(int(emb[0])) + ' ' + ' '.join([str(x) for x in emb[1:]]) + '\n' for emb in embedding_list]

            file_path = '%s%d-%s' % (config.save_path, epoch, emb_filenames[i])
            with open(file_path, 'w') as f:
                lines = [str(self.n_node) + ' ' + str(config.n_emb) + '\n'] + embedding_str
                f.writelines(lines)
        self.saver.save(self.sess, config.save_path + 'model.ckpt', global_step=epoch)
    
    def my_print(self, info, r_flag, verbose):
        if verbose == 1 and config.verbose == 0:
            return
        if r_flag:
            print('\r%s' % info, end='')
        else:
            print('%s' % info)




if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', metavar='DATASET',
                            help='training dataset')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
    help='learning rate')
    args = parser.parse_args()

    tf.reset_default_graph()
    config = Config()
    config.g_batch_size = 64
    config.d_batch_size = 64
    config.lambda_gen = 1e-5
    config.lambda_dis = 1e-5
    config.lr_gen = args.lr
    config.lr_dis = args.lr
    config.n_epoch = 70
    config.sig = 1.0
    config.label_smooth = 0.0
    config.d_epoch = 5
    config.g_epoch = 5
    config.n_emb = 128
    config.pre_d_epoch = 0
    config.pre_g_epoch = 0
    config.neg_weight = [1, 1]
    config.walk_num = 10
    config.walk_len = 10
    config.adv = False
    config.rand = False

    config.name = '3'
    # node_classification
    config.experiment = 'nc'
    config.dataset = args.dataset
    # config.dataset = 'citeseer'
    config.train_filename = './data/' + config.experiment + '/' + config.dataset + '/train_1'
    config.test_filename = './data/' + config.experiment + '/' + config.dataset + '/test_1'
    config.save_path = './results/%s/%s/%s/' % (config.experiment, config.dataset, config.name)
    config.save = False
    config.verbose = 1
    config.log = True

    model = Model()
    best_auc, best_epoch = model.train()
    for i in range(3):
        print("{} {}".format(best_auc[i], best_epoch[i]))
    del model
    del config
