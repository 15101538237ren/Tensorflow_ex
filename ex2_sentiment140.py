# -*- coding: utf-8 -*-
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pickle,os,random
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import OrderedDict
import sys
reload(sys)
sys.setdefaultencoding('latin-1')

org_train_file = 'data/sentiment140/training.1600000.processed.noemoticon.csv'
org_test_file = 'data/sentiment140/testdata.manual.2009.06.14.csv'


# 提取文件中有用的字段
def usefull_filed(org_file, output_file):
    output = open(output_file, 'w')
    with open(org_file, buffering=10000) as f:
        try:
            for line in f:                # "4","2193601966","Tue Jun 16 08:40:49 PDT 2009","NO_QUERY","AmandaMarie1028","Just woke up. Having no school is the best feeling ever "
                line = line.replace('"', '')
                clf = line.split(',')[0]   # 4
                if clf == '0':
                    clf = [0, 0, 1]  # 消极评论
                elif clf == '2':
                    clf = [0, 1, 0]  # 中性评论
                elif clf == '4':
                    clf = [1, 0, 0]  # 积极评论

                tweet = line.split(',')[-1] #内容

                outputline = str(clf) + ':%:%:%:' + tweet
                output.write(outputline)  # [0, 0, 1]:%:%:%: that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D
        except Exception as e:
            print(e)
    output.close()  # 处理完成，处理后文件大小127.5M
training_file_path = 'data/sentiment140/training.csv'
usefull_filed(org_train_file, training_file_path)
testing_file_path = 'data/sentiment140/tesing.csv'
usefull_filed(org_test_file, testing_file_path)

# 创建词汇表
def create_lexicon(train_file):
    lex = []
    lemmatizer = WordNetLemmatizer()
    with open(train_file, buffering=10000) as f:
        try:
            count_word = {}  # 统计单词出现次数
            for line in f:
                tweet = line.split(':%:%:%:')[1]
                words = word_tokenize(tweet.lower())
                for word in words:
                    word = lemmatizer.lemmatize(word)
                    if word not in count_word:
                        count_word[word] = 1
                    else:
                        count_word[word] += 1

            count_word = OrderedDict(sorted(count_word.items(), key=lambda t: t[1]))
            for word in count_word:
                if count_word[word] < 100000 and count_word[word] > 100:  # 过滤掉一些词
                    lex.append(word)
        except Exception as e:
            print(e)
    return lex

#lex = create_lexicon(training_file_path)
pickle_file_path = 'data/sentiment140/lexcion.pkl'

# with open(pickle_file_path, 'wb') as f:
#     pickle.dump(lex, f)


"""
# 把字符串转为向量
def string_to_vector(input_file, output_file, lex):
    output_f = open(output_file, 'w')
    lemmatizer = WordNetLemmatizer()
    with open(input_file, buffering=10000) as f:
        for line in f:
            label = line.split(':%:%:%:')[0]
            tweet = line.split(':%:%:%:')[1]
            words = word_tokenize(tweet.lower())
            words = [lemmatizer.lemmatize(word) for word in words]

            features = np.zeros(len(lex))
            for word in words:
                if word in lex:
                    features[lex.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大

            features = list(features)
            output_f.write(str(label) + ":" + str(features) + '\n')
    output_f.close()


f = open('lexcion.pickle', 'rb')
lex = pickle.load(f)
f.close()

# lexcion词汇表大小112k,training.vec大约112k*1600000  170G  太大，只能边转边训练了
# string_to_vector('training.csv', 'training.vec', lex)
# string_to_vector('tesing.csv', 'tesing.vec', lex)
"""

f = open(pickle_file_path, 'rb')
lex = pickle.load(f)
f.close()


def get_random_line(file, point):
    file.seek(point)
    file.readline()
    return file.readline()
# 从文件中随机选择n条记录
def get_n_random_line(file_name, n=150):
    lines = []
    file = open(file_name)
    total_bytes = os.stat(file_name).st_size
    for i in range(n):
        random_point = random.randint(0, total_bytes)
        lines.append(get_random_line(file, random_point))
    file.close()
    return lines

def get_test_dataset(test_file):
    with open(test_file) as f:
        test_x = []
        test_y = []
        lemmatizer = WordNetLemmatizer()
        for line in f:
            label = line.split(':%:%:%:')[0]
            tweet = line.split(':%:%:%:')[1]
            words = word_tokenize(tweet.lower())
            words = [lemmatizer.lemmatize(word) for word in words]
            features = np.zeros(len(lex))
            for word in words:
                if word in lex:
                    features[lex.index(word)] = 1

            test_x.append(list(features))
            test_y.append(eval(label))
    return test_x, test_y

test_x, test_y = get_test_dataset(testing_file_path)

#######################################################################

input_size = len(lex)  # 输入层

num_classes = 3

X = tf.placeholder(tf.int32, [None, input_size])
Y = tf.placeholder(tf.float32, [None, num_classes])
dropout_keep_prob = tf.placeholder(tf.float32)

batch_size = 90

def neural_network():
    # embedding layer
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        embedding_size = 128
        W = tf.Variable(tf.random_uniform([input_size, embedding_size], -1.0, 1.0))
        embedded_chars = tf.nn.embedding_lookup(W, X)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
    # convolution + maxpool layer
    num_filters = 128
    filter_sizes = [3,4,5]
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
            conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID")
            h = tf.nn.relu(tf.nn.bias_add(conv, b))
            pooled = tf.nn.max_pool(h, ksize=[1, input_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
            pooled_outputs.append(pooled)

    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    # dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
    # output
    with tf.name_scope("output"):
        W = tf.get_variable("W", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        output = tf.nn.xw_plus_b(h_drop, W, b)

    return output
model_path = 'model/senti_model.ckpt'
def train_neural_network():
    output = neural_network()

    optimizer = tf.train.AdamOptimizer(1e-3)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        lemmatizer = WordNetLemmatizer()
        i = 0
        while True:
            batch_x = []
            batch_y = []

            #if model.ckpt文件已存在:
            #    saver.restore(session, 'model.ckpt')  恢复保存的session
            try:
                lines = get_n_random_line(training_file_path, batch_size)
                for line in lines:
                    label = line.split(':%:%:%:')[0]
                    tweet = line.split(':%:%:%:')[1]
                    words = word_tokenize(tweet.lower())
                    words = [lemmatizer.lemmatize(word) for word in words]

                    features = np.zeros(len(lex))
                    for word in words:
                        if word in lex:
                            features[lex.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大

                    batch_x.append(list(features))
                    batch_y.append(eval(label))

                _, loss_ = sess.run([train_op, loss], feed_dict={X:batch_x, Y:batch_y, dropout_keep_prob:0.5})
                print(loss_)
            except Exception as e:
                print(e)

            if i % 10 == 0:
                predictions = tf.argmax(output, 1)
                correct_predictions = tf.equal(predictions, tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
                accur = sess.run(accuracy, feed_dict={X:test_x[0:50], Y:test_y[0:50], dropout_keep_prob:1.0})
                saver.save(sess, model_path)
                print('准确率:', accur)

            i += 1
train_neural_network()

def prediction(tweet_text):
    predict = neural_network(X)

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        saver.restore(session, model_path)

        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(tweet_text.lower())
        words = [lemmatizer.lemmatize(word) for word in words]

        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1

        #print(predict.eval(feed_dict={X:[features]})) [[val1,val2,val3]]
        res = session.run(tf.argmax(predict.eval(feed_dict={X:[features]}),1 ))
        return res
#prediction("I am very happy")



