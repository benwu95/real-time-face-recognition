from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import pickle
from scipy import misc
import facenet
import detect_face

img_size = 160


def align_mtcnn(img, pnet, rnet, onet, minsize):
    threshold = [0.6, 0.7, 0.9]  # three steps's threshold
    factor = 0.709  # scale factor

    if img.ndim == 2:
        img = facenet.to_rgb(img)
    img = img[:, :, 0:3]

    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    num_of_faces = bounding_boxes.shape[0]
    aligned_list = []
    for face_pos in bounding_boxes:
        face_pos = np.rint(face_pos)
        aligned_list.append(face_pos.astype(int))

    return aligned_list


def restore_mtcnn_model():
    gpu_memory_fraction = 1.0
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, 'models/mtcnn/')
    return pnet, rnet, onet


def restore_facenet_model(model_path):
    # load the model
    print('Loading feature extraction model')
    facenet.load_model(model_path)


def restore_classifier(classifier_path):
    # load the classifier
    print('Loading face classifier')
    if os.path.isdir(classifier_path):
        classifier = []
        class_names = []
        for root, dirs, files in os.walk(classifier_path):
            for file in files:
                with open(os.path.join(root, file), 'rb') as infile:
                    tmp1, tmp2 = pickle.load(infile)
                    classifier.append(tmp1)
                    class_names.append(tmp2)

    if os.path.isfile(classifier_path):
        with open(classifier_path, 'rb') as infile:
            classifier, class_names = pickle.load(infile)

    return classifier, class_names


def get_face_in_frame(frame, aligned_list):
    images = np.zeros((len(aligned_list), img_size, img_size, 3))
    i = 0
    for face_pos in aligned_list:
        if face_pos[0] < 0 or face_pos[1] < 0:
            continue
        else:
            img = frame[face_pos[1]:face_pos[3], face_pos[0]:face_pos[2], ]
            if img.ndim == 2:
                img = facenet.to_rgb(img)
            img = misc.imresize(img, (img_size, img_size), interp='bilinear')
            img = facenet.prewhiten(img)
            img = facenet.crop(img, False, img_size)
            images[i, :, :, :] = img
            i += 1
    return images


def identify_face(sess, frame, aligned_list, classifier, class_names):
    # get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]

    # run forward pass to calculate embeddings
    emb_array = np.zeros((len(aligned_list), embedding_size))
    faces = get_face_in_frame(frame, aligned_list)
    feed_dict = {images_placeholder: faces, phase_train_placeholder: False}
    emb_array = sess.run(embeddings, feed_dict=feed_dict)

    # classify faces
    predict = classifier.predict_proba(emb_array)
    best_class_indices = np.argmax(predict, axis=1)
    best_class_prob = predict[np.arange(len(best_class_indices)), best_class_indices]

    name_list = []
    for i in range(len(best_class_indices)):
        if class_names[best_class_indices[i]] == 'others':
            print('<ERROR> %s: %.3f' % (class_names[best_class_indices[i]], best_class_prob[i]))
        else:
            print('%s: %.3f' % (class_names[best_class_indices[i]], best_class_prob[i]))
        name_list.append(class_names[best_class_indices[i]])
    print('-----')

    return name_list
