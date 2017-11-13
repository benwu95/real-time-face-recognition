import time
import tensorflow as tf
import argparse
import sys
from scipy import misc
import cv2
import identify_face


def main(args):
    stamp = 0
    prev_time = 0

    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 50
    config.inter_op_parallelism_threads = 5
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            # restore mtcnn model
            pnet, rnet, onet = identify_face.restore_mtcnn_model()

            # restore the classifier
            classifier, class_names = identify_face.restore_classifier(args.classifier)

            # load the camera
            cap = cv2.VideoCapture(0)

            if args.mode == 'ONLY_DETECT':
                while cap.isOpened():
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # face detect
                    cur_time = time.time()
                    aligned_list = identify_face.align_mtcnn(rgb, pnet, rnet, onet, args.minsize)

                    if len(aligned_list) > 0:
                        # boxed the face
                        for face_pos in aligned_list:
                            cv2.rectangle(frame, (face_pos[0], face_pos[1]), (face_pos[2], face_pos[3]), (0, 255, 0), 2)

                    # Display the resulting frame
                    cv2.imshow('frame', frame)
                    prev_time = cur_time

                    # keyboard event
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('s'):
                        print('save')
                        misc.imsave('%s.jpg' % time.strftime("%Y%m%d_%H%M%S", time.localtime()), rgb)
                    elif k == ord('q'):
                        break

            if args.mode == 'ALL':
                # restore facenet model
                identify_face.restore_facenet_model(args.model)

                while cap.isOpened():
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # face detect
                    cur_time = time.time()
                    aligned_list = identify_face.align_mtcnn(rgb, pnet, rnet, onet, args.minsize)

                    if len(aligned_list) > 0:
                        # boxed the face
                        for face_pos in aligned_list:
                            cv2.rectangle(frame, (face_pos[0], face_pos[1]), (face_pos[2], face_pos[3]), (0, 255, 0), 2)

                        # face identify
                        if stamp % args.interval == 0:
                            name_list = identify_face.identify_face(sess, rgb, aligned_list, classifier, class_names)
                            # show users' name
                            show_name(frame, aligned_list, name_list)
                            print(cur_time - prev_time)

                        # add the stamp
                        stamp += 1

                    # Display the resulting frame
                    cv2.imshow('frame', frame)
                    prev_time = cur_time

                    # keyboard event
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('s'):
                        print('save')
                        misc.imsave('%s.jpg' % time.strftime("%Y%m%d_%H%M%S", time.localtime()), rgb)
                    elif k == ord('q'):
                        break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


def show_name(frame, aligned_list, name_list):
    if len(name_list) > 0:
        i = 0
        for face_pos in aligned_list:
            if name_list[i] == 'others':
                cv2.putText(frame, name_list[i], (face_pos[0], face_pos[1] - 30),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2, lineType=2)
                cv2.rectangle(frame, (face_pos[0], face_pos[1]), (face_pos[2], face_pos[3]), (0, 0, 255), 2)
            else:
                cv2.putText(frame, name_list[i], (face_pos[0], face_pos[1] - 30),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), thickness=2, lineType=2)
            i += 1


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['ONLY_DETECT', 'ALL'],
                        help='Indicates if the program only detects faces or' +
                        ' recognizes faces from the camera.',
                        default='ALL')
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier', type=str,
                        help='Classifier model file name as a pickle (.pkl) file.')
    parser.add_argument('--interval', type=int,
                        help='Frame interval of each face recognition event.',
                        default=5)
    parser.add_argument('--minsize', type=int,
                        help='Minimum size (height, width) of face in pixels.',
                        default=80)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
