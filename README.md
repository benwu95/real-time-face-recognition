# Real Time Face Recognition using FaceNet and OpenCV
This is a real time face recognition project based on [FaceNet](https://github.com/davidsandberg/facenet) and OpenCV.

## Compatibility
The code is tested using Tensorflow 1.3 with GPU support under Fedora 26 with Python 2.7 and Python 3.6.

## Requirements
* NumPy
* SciPy
* scikit-learn
* Pillow
* OpenCV-Python
* TensorFlow

## Pre-processing
### The dataset structure
```
face_DB/raw
├── ID1
│     ├── ID1_001.jpg
│     ├── ID1_002.jpg
│     ├── ID1_003.jpg
│     ├── ID1_004.jpg
│     └── ID1_005.jpg
├── ID2
│     ├── ID2_001.jpg
│     ├── ID2_002.jpg
│     ├── ID2_003.jpg
│     ├── ID2_004.jpg
│     └── ID2_005.jpg
├── ID3
│     ├── ID3_001.jpg
...
...
```
### Pre-trained models
Use the Pre-trained models from [davidsandberg/facenet](https://github.com/davidsandberg/facenet#pre-trained-models)
### Align the dataset
```
python align_dataset_mtcnn.py <raw_img_dir> <aligned_img_dir>
```
**Example**
```
python align_dataset_mtcnn.py Face_db/raw Face_db/align_160
```
### Train a classifier
```
python classifier.py TRAIN <aligned_img_dir> <facenet_model_path> <classifier_path>
```
**Example**
```
python classifier.py TRAIN Face_db/align_160/ models/20170512-110547/20170512-110547.pb models/classifier/test_classifier.pkl
```
## Run
```
python camera.py <mode> <facenet_model_path> <classifier_path> --interval=5 --minsize=80
```
* mode
    * ONLY_DETECT: Only detects faces from the camera
    * ALL: Recognizes faces from the camera
* interval: Frame interval of each face recognition event, default value is 5
* minsize: Minimum size (height, width) of face in pixels, default value is 80

**Example**
```
python camera.py ALL models/20170512-110547/20170512-110547.pb models/classifier/test_classifier.pkl --interval=5 --minsize=80
```

## Inspiration
* [davidsandberg/facenet](https://github.com/davidsandberg/facenet)
The following codes and files was taken from this repository:
    * faceney.py
    * detect_face.py
    * align_dataset_mtcnn.py
    * classifier.py
    * models/mtcnn/
* [shanren7/real_time_face_recognition](https://github.com/shanren7/real_time_face_recognition)
The workflow was inspired by here.
