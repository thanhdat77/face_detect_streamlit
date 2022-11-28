import argparse
import av
import cv2 as cv
import os
from io import StringIO
import numpy as np
from streamlit_option_menu import option_menu
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import sys
import subprocess
import joblib
from PIL import Image
import time
import uuid
from pathlib import  Path
import logging
import pandas as pd
from typing import List, NamedTuple
# from object_detection import postprocess

with st.sidebar: 
    selected = option_menu("Chọn model muốn dùng", ["Facebook","Yolo3", 'DNN_opencv'
        ,"DNN_opencv_caffee"], 
        icons=["archive", "activity","app" ,"app-indicator" ], menu_icon="cast", default_index=0)
# the facebook page 
if selected == "Facebook":
    print(selected)
    def str2bool(v):
        if v.lower() in ['on', 'yes', 'true', 'y', 't']:
            return True
        elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
            return False
        else:
            raise NotImplementedError

    parser = argparse.ArgumentParser()
    parser.add_argument('--image1', '-i1', type=str, help='Path to the input image1. Omit for detecting on default camera.')
    parser.add_argument('--image2', '-i2', type=str, help='Path to the input image2. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm.')
    parser.add_argument('--video', '-v', type=str, help='Path to the input video.')
    parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
    parser.add_argument('--face_detection_model', '-fd', type=str, default='model_face/face_detection_yunet_2022mar.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
    parser.add_argument('--face_recognition_model', '-fr', type=str, default='model_face/face_recognition_sface_2021dec.onnx', help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
    parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
    parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
    parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
    parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
    args = parser.parse_args()

    svc = joblib.load('model_face/svc.pkl')
    
    mydict = ['BanGiang', 'ThayDuc']
    def visualize(input, faces, fps, thickness=2):
        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

                coords = face[:-1].astype(np.int32)
                img = cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
                img = cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
                img = cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
                img = cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
                img = cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
                img = cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
        img = cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img

    detector = cv.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (320, 320),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )
    recognizer = cv.FaceRecognizerSF.create(
    args.face_recognition_model,"")

    tm = cv.TickMeter()

    cap = cv.VideoCapture(0)
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    dem = 0

    st.header(selected+" Page")
    FRAME_WINDOW = st.image([])
    tm = cv.TickMeter()
    cap  = cv.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        tm.start()
        faces = detector.detect(frame) # faces is a tuple
      
        frame_pre = visualize(frame, faces, tm.getFPS())
        
        tm.stop()
        if faces[1] is not None:
            face_align = recognizer.alignCrop(frame, faces[1][0])
        face_feature = recognizer.feature(face_align)
        test_predict = svc.predict(face_feature)
        result = mydict[test_predict[0]]
        frame_pre = cv.putText(frame,result,(1,50),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        frame_pre = cv.cvtColor(frame_pre, cv.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_pre)
        
if selected == "Yolo3":
    st.header(selected+"Page")
    print(selected)
    
    @st.cache
    def load_image(image_file):
        img = Image.open(image_file)
        return img
    
    image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])
    
    if image_file is not None:
        file_details = {"FileName":image_file.name,"FileType":image_file.type}
        img = load_image(image_file)
        img.save(image_file.name)
        st.image(img)
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
        # print(image_file.name)
        new_dir = str(uuid.uuid4())+'.jpg'
        subprocess.run("python object_detection.py  --thr={thr} --input={inp}  --model=model_yolo/yolov4-tiny.weights --config=model_yolo/yolov4-tiny.cfg --classes=model_yolo/coco.names --sdir={sdir} --width=416 --height=416 --scale=0.00392\n".format(thr=confidence_threshold,inp = image_file.name,sdir=new_dir))
        st.image(load_image(new_dir))
        os.remove(image_file.name)
        os.remove(new_dir)
        
    
if selected == "DNN_opencv":
    st.header(selected+"Page")
    print(selected)
    HERE = Path(__file__).parent    
    ROOT = HERE.parent

    logger = logging.getLogger(__name__)


    MODEL_LOCAL_PATH =  "model_DNN/res10_300x300_ssd_iter_140000.caffemodel"
    PROTOTXT_LOCAL_PATH =  "model_DNN/opencv_face_detector3.prototxt"

    CLASSES = [
        'person'
    ]


    @st.experimental_singleton  
    def generate_label_colors():
        return np.random.uniform(0, 255, size=(len(CLASSES), 3))


    COLORS = generate_label_colors()

    DEFAULT_CONFIDENCE_THRESHOLD = 0.85


    class Detection(NamedTuple):
        name: str
        prob: float

    def add_argument(zoo, parser, name, help, required=False, default=None, type=None, action=None, nargs=None):
        if len(sys.argv) <= 1:
            return

        modelName = sys.argv[1]

        if os.path.isfile(zoo):
            fs = cv.FileStorage(zoo, cv.FILE_STORAGE_READ)
            node = fs.getNode(modelName)
            if not node.empty():
                value = node.getNode(name)
                if not value.empty():
                    if value.isReal():
                        default = value.real()
                    elif value.isString():
                        default = value.string()
                    elif value.isInt():
                        default = int(value.real())
                    elif value.isSeq():
                        default = []
                        for i in range(value.size()):
                            v = value.at(i)
                            if v.isInt():
                                default.append(int(v.real()))
                            elif v.isReal():
                                default.append(v.real())
                            else:
                                print('Unexpected value format')
                                exit(0)
                    else:
                        print('Unexpected field format')
                        exit(0)
                    required = False

        if action == 'store_true':
            default = 1 if default == 'true' else (0 if default == 'false' else default)
            assert(default is None or default == 0 or default == 1)
            parser.add_argument('--' + name, required=required, help=help, default=bool(default),
                                action=action)
        else:
            parser.add_argument('--' + name, required=required, help=help, default=default,
                                action=action, nargs=nargs, type=type)

    def add_preproc_args(zoo, parser, sample):
        aliases = []
        if os.path.isfile(zoo):
            fs = cv.FileStorage(zoo, cv.FILE_STORAGE_READ)
            root = fs.root()
            for name in root.keys():
                model = root.getNode(name)
                if model.getNode('sample').string() == sample:
                    aliases.append(name)

        parser.add_argument('alias', nargs='?', choices=aliases,
                            help='An alias name of model to extract preprocessing parameters from models.yml file.')
        add_argument(zoo, parser, 'model', required=True,
                    help='Path to a binary file of model contains trained weights. '
                        'It could be a file with extensions .caffemodel (Caffe), '
                        '.pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet), .bin (OpenVINO)')
        add_argument(zoo, parser, 'config',
                    help='Path to a text file of model contains network configuration. '
                        'It could be a file with extensions .prototxt (Caffe), .pbtxt or .config (TensorFlow), .cfg (Darknet), .xml (OpenVINO)')
        add_argument(zoo, parser, 'mean', nargs='+', type=float, default=[0, 0, 0],
                    help='Preprocess input image by subtracting mean values. '
                        'Mean values should be in BGR order.')
        add_argument(zoo, parser, 'scale', type=float, default=1.0,
                    help='Preprocess input image by multiplying on a scale factor.')
        add_argument(zoo, parser, 'width', type=int,
                    help='Preprocess input image by resizing to a specific width.')
        add_argument(zoo, parser, 'height', type=int,
                    help='Preprocess input image by resizing to a specific height.')
        add_argument(zoo, parser, 'rgb', action='store_true',
                    help='Indicate that model works with RGB input images instead BGR ones.')
        add_argument(zoo, parser, 'classes',
                    help='Optional path to a text file with names of classes to label detected objects.')


    backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_HALIDE, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv.dnn.DNN_BACKEND_OPENCV,
                cv.dnn.DNN_BACKEND_VKCOM, cv.dnn.DNN_BACKEND_CUDA)
    targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD, cv.dnn.DNN_TARGET_HDDL,
            cv.dnn.DNN_TARGET_VULKAN, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yml'),
                        help='An optional path to file with preprocessing parameters.')
    parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
    parser.add_argument('--out_tf_graph', default='graph.pbtxt',
                        help='For models from TensorFlow Object Detection API, you may '
                            'pass a .config file which was used for training through --config '
                            'argument. This way an additional .pbtxt file with TensorFlow graph will be created.')
    parser.add_argument('--framework', choices=['caffe', 'tensorflow', 'torch', 'darknet', 'dldt'],
                        help='Optional name of an origin framework of the model. '
                            'Detect it automatically if it does not set.')
    parser.add_argument('--thr', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.4, help='Non-maximum suppression threshold')
    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                        help="Choose one of computation backends: "
                            "%d: automatically (by default), "
                            "%d: Halide language (http://halide-lang.org/), "
                            "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                            "%d: OpenCV implementation, "
                            "%d: VKCOM, "
                            "%d: CUDA" % backends)
    parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                        help='Choose one of target computation devices: '
                            '%d: CPU target (by default), '
                            '%d: OpenCL, '
                            '%d: OpenCL fp16 (half-float precision), '
                            '%d: NCS2 VPU, '
                            '%d: HDDL VPU, '
                            '%d: Vulkan, '
                            '%d: CUDA, '
                            '%d: CUDA fp16 (half-float preprocess)' % targets)
    parser.add_argument('--async', type=int, default=0,
                        dest='asyncN',
                        help='Number of asynchronous forwards at the same time. '
                            'Choose 0 for synchronous mode')
    args, _ = parser.parse_known_args()
    add_preproc_args(args.zoo, parser, 'object_detection')
    parser = argparse.ArgumentParser(parents=[parser],
                                    description='Use this script to run object detection deep learning networks using OpenCV.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    nmsThreshold = args.nms
    args.width, args.height = 300, 300 



    cache_key = "PhatHienKhuonMat_DNN_opencv"
    if cache_key in st.session_state:
        net = st.session_state[cache_key]
    else:
        net = cv.dnn.readNet("model_DNNCaffe/opencv_face_detector_uint8.pb", 'model_DNNCaffe/opencv_face_detector.pbtxt', args.framework)
        net.setPreferableBackend(args.backend)
        net.setPreferableTarget(args.target)
        outNames = net.getUnconnectedOutLayersNames()
        st.session_state[cache_key] = net

    streaming_placeholder = st.empty()

    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )
    

    def _annotate_image(frame, outs):
        txtconf = 0
        frame_pre = None
        result: List[Detection] = []
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        def drawPred(classId, conf, left, top, right, bottom):
            
            rec = cv.rectangle(frame,(int(left), int(top)),(int(right), int(bottom)), (0,222,0))

            label = 'person %.2f' % conf    

            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            rec1 = cv.rectangle(rec, (int(left), int(top - labelSize[1])), (int(left + labelSize[0]), int(top + baseLine)), (255, 255, 255), cv.FILLED)
            rec2 = cv.putText(rec1, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            return rec2

        layerNames = net.getLayerNames()
        lastLayerId = net.getLayerId(layerNames[-1])
        lastLayer = net.getLayer(lastLayerId)

        classIds = []
        confidences = []
        boxes = []
        if lastLayer.type == 'DetectionOutput':
        
            change_fame_W = (640-300)/100
            change_fame_H = (460 -300)/100
            
            for out in outs:
                for detection in out[0, 0]:
                    confidence = detection[2]
                    if confidence > confidence_threshold:
                        left = int(detection[3]) + change_fame_W
                        top = int(detection[4]) + change_fame_H
                        right = int(detection[5])- change_fame_W
                        bottom = int(detection[6])- change_fame_H
                        width = right - left + 5
                        height = bottom - top + 5
                        if width <= 2 or height <= 2:
                            left = int(detection[3] * frameWidth)+ 5
                            top = int(detection[4] * frameHeight)+ 5
                            right = int(detection[5] * frameWidth)- 5
                            bottom = int(detection[6] * frameHeight)- 5
                            width = right - left + 5
                            height = bottom - top + 5
                        classIds.append(int(detection[1]) - 1)  
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])
                        
        elif lastLayer.type == 'Region':
        
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > confidence_threshold:
                        center_x = int(detection[0] * frameWidth)
                        center_y = int(detection[1] * frameHeight)
                        width = int(detection[2] * frameWidth)
                        height = int(detection[3] * frameHeight)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)
                        classIds.append(classId)
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])
        else:
            print('Unknown output layer type: ' + lastLayer.type)
            exit()

        if len(net.getUnconnectedOutLayersNames()) > 1 or lastLayer.type == 'Region' and args.backend != cv.dnn.DNN_BACKEND_OPENCV:
            indices = []
            classIds = np.array(classIds)
            boxes = np.array(boxes)
            confidences = np.array(confidences)
            unique_classes = set(classIds)
            for cl in unique_classes:
                class_indices = np.where(classIds == cl)[0]
                conf = confidences[class_indices]
                box  = boxes[class_indices].tolist()
                nms_indices = cv.dnn.NMSBoxes(box, conf, confidence_threshold, nmsThreshold)
                nms_indices = nms_indices[:, 0] if len(nms_indices) else []
                indices.extend(class_indices[nms_indices])
        else:
            indices = np.arange(0, len(classIds))

        for i in indices:
            box = boxes[i]
            left = box[0]+14
            top = box[1]+14
            width = box[2]+14
            height = box[3]+5
            frame_pre = drawPred("person", confidences[i], int(left)  , int(top), int(left + width), int(top + height))
            txtconf = confidences[i]
        
        return frame_pre, ["persion",txtconf]



    def handle_callback(frame ,result_queue) :
    
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        
        if not frame is None:
            # Create a 4D blob from a frame.
            
            # image = frame.to_ndarray(format="bgr24")
        
            blob = cv.dnn.blobFromImage(frame, size=(frameWidth, frameHeight), swapRB=True, ddepth=cv.CV_8U)
            

            # Run a model
            
            net.setInput(blob)
            if net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
                frame = cv.resize(frame, (300, 300))
                net.setInput(np.array([[300, 300, 1.6]], dtype=np.float32), 'im_info')

            if args.asyncN:
                    net.forwardAsync()
            else:
                outs = net.forward(net.getUnconnectedOutLayersNames())
                annotated_image, result = _annotate_image(frame, outs)
                result_queue.append(result)
                annotated_image =  np.array(annotated_image)
                
        

        return annotated_image, result_queue

    result_queue = []    
    FW = st.image([])
    cap = cv.VideoCapture(0)
    cbxCamera = st.checkbox("Camera",key="mocamere1")
    cbxLabel = st.checkbox("Hiện nhãn", value=True)
    tblConf = st.table([])
    df = None
    print(cbxCamera)
    while cbxCamera:
        _, frame = cap.read()
        # cap.set(cv.CAP_PROP_ZOOM, 0x8004)
        frame_pre, result_queue = handle_callback(frame, result_queue)
        
        if len(frame_pre.shape)==2 or len(frame_pre.shape)==3:
            frame_pre = cv.cvtColor(frame_pre, cv.COLOR_BGR2RGB)
            FW.image(frame_pre) 
            if cbxLabel:
                result_queue = sorted(result_queue, key = lambda x: x[1])
                df = pd.DataFrame(result_queue[-2:],columns=( ["name", "confiden"]))
                tblConf.table(df)
        
    st.markdown(
        "bài làm deloy detection with streamlit"
        
    )
    
if selected == "DNN_opencv_caffee":
    st.header(selected+"Page")
    print(selected)
    HERE = Path(__file__).parent
    ROOT = HERE.parent

    logger = logging.getLogger(__name__)


    MODEL_LOCAL_PATH =  "res10_300x300_ssd_iter_140000.caffemodel"
    PROTOTXT_LOCAL_PATH =  "opencv_face_detector3.prototxt"

    CLASSES = [
        'person'
    ]


    @st.experimental_singleton  
    def generate_label_colors():
        return np.random.uniform(0, 255, size=(len(CLASSES), 3))


    COLORS = generate_label_colors()

    DEFAULT_CONFIDENCE_THRESHOLD = 0.85


    class Detection(NamedTuple):
        name: str
        prob: float

    def add_argument(zoo, parser, name, help, required=False, default=None, type=None, action=None, nargs=None):
        if len(sys.argv) <= 1:
            return

        modelName = sys.argv[1]

        if os.path.isfile(zoo):
            fs = cv.FileStorage(zoo, cv.FILE_STORAGE_READ)
            node = fs.getNode(modelName)
            if not node.empty():
                value = node.getNode(name)
                if not value.empty():
                    if value.isReal():
                        default = value.real()
                    elif value.isString():
                        default = value.string()
                    elif value.isInt():
                        default = int(value.real())
                    elif value.isSeq():
                        default = []
                        for i in range(value.size()):
                            v = value.at(i)
                            if v.isInt():
                                default.append(int(v.real()))
                            elif v.isReal():
                                default.append(v.real())
                            else:
                                print('Unexpected value format')
                                exit(0)
                    else:
                        print('Unexpected field format')
                        exit(0)
                    required = False

        if action == 'store_true':
            default = 1 if default == 'true' else (0 if default == 'false' else default)
            assert(default is None or default == 0 or default == 1)
            parser.add_argument('--' + name, required=required, help=help, default=bool(default),
                                action=action)
        else:
            parser.add_argument('--' + name, required=required, help=help, default=default,
                                action=action, nargs=nargs, type=type)

    def add_preproc_args(zoo, parser, sample):
        aliases = []
        if os.path.isfile(zoo):
            fs = cv.FileStorage(zoo, cv.FILE_STORAGE_READ)
            root = fs.root()
            for name in root.keys():
                model = root.getNode(name)
                if model.getNode('sample').string() == sample:
                    aliases.append(name)

        parser.add_argument('alias', nargs='?', choices=aliases,
                            help='An alias name of model to extract preprocessing parameters from models.yml file.')
        add_argument(zoo, parser, 'model', required=True,
                    help='Path to a binary file of model contains trained weights. '
                        'It could be a file with extensions .caffemodel (Caffe), '
                        '.pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet), .bin (OpenVINO)')
        add_argument(zoo, parser, 'config',
                    help='Path to a text file of model contains network configuration. '
                        'It could be a file with extensions .prototxt (Caffe), .pbtxt or .config (TensorFlow), .cfg (Darknet), .xml (OpenVINO)')
        add_argument(zoo, parser, 'mean', nargs='+', type=float, default=[0, 0, 0],
                    help='Preprocess input image by subtracting mean values. '
                        'Mean values should be in BGR order.')
        add_argument(zoo, parser, 'scale', type=float, default=1.0,
                    help='Preprocess input image by multiplying on a scale factor.')
        add_argument(zoo, parser, 'width', type=int,
                    help='Preprocess input image by resizing to a specific width.')
        add_argument(zoo, parser, 'height', type=int,
                    help='Preprocess input image by resizing to a specific height.')
        add_argument(zoo, parser, 'rgb', action='store_true',
                    help='Indicate that model works with RGB input images instead BGR ones.')
        add_argument(zoo, parser, 'classes',
                    help='Optional path to a text file with names of classes to label detected objects.')


    backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_HALIDE, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv.dnn.DNN_BACKEND_OPENCV,
                cv.dnn.DNN_BACKEND_VKCOM, cv.dnn.DNN_BACKEND_CUDA)
    targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD, cv.dnn.DNN_TARGET_HDDL,
            cv.dnn.DNN_TARGET_VULKAN, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yml'),
                        help='An optional path to file with preprocessing parameters.')
    parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
    parser.add_argument('--out_tf_graph', default='graph.pbtxt',
                        help='For models from TensorFlow Object Detection API, you may '
                            'pass a .config file which was used for training through --config '
                            'argument. This way an additional .pbtxt file with TensorFlow graph will be created.')
    parser.add_argument('--framework', choices=['caffe', 'tensorflow', 'torch', 'darknet', 'dldt'],
                        help='Optional name of an origin framework of the model. '
                            'Detect it automatically if it does not set.')
    parser.add_argument('--thr', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.4, help='Non-maximum suppression threshold')
    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                        help="Choose one of computation backends: "
                            "%d: automatically (by default), "
                            "%d: Halide language (http://halide-lang.org/), "
                            "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                            "%d: OpenCV implementation, "
                            "%d: VKCOM, "
                            "%d: CUDA" % backends)
    parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                        help='Choose one of target computation devices: '
                            '%d: CPU target (by default), '
                            '%d: OpenCL, '
                            '%d: OpenCL fp16 (half-float precision), '
                            '%d: NCS2 VPU, '
                            '%d: HDDL VPU, '
                            '%d: Vulkan, '
                            '%d: CUDA, '
                            '%d: CUDA fp16 (half-float preprocess)' % targets)
    parser.add_argument('--async', type=int, default=0,
                        dest='asyncN',
                        help='Number of asynchronous forwards at the same time. '
                            'Choose 0 for synchronous mode')
    args, _ = parser.parse_known_args()
    add_preproc_args(args.zoo, parser, 'object_detection')
    parser = argparse.ArgumentParser(parents=[parser],
                                    description='Use this script to run object detection deep learning networks using OpenCV.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    nmsThreshold = args.nms
    args.width, args.height = 300, 300 



    cache_key = "PhatHienKhuonMat_DNN_opencv_caffee"
    if cache_key in st.session_state:
        net = st.session_state[cache_key]
    else:
        net = cv.dnn.readNet("model_DNNCaffe/opencv_face_detector_uint8.pb", 'model_DNNCaffe/opencv_face_detector.pbtxt', args.framework)
        net.setPreferableBackend(args.backend)
        net.setPreferableTarget(args.target)
        outNames = net.getUnconnectedOutLayersNames()
        st.session_state[cache_key] = net

    streaming_placeholder = st.empty()

    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )
    

    def _annotate_image(frame, outs):
        txtconf = 0
        frame_pre = None
        result: List[Detection] = []
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        def drawPred(classId, conf, left, top, right, bottom):
            
            rec = cv.rectangle(frame,(int(left), int(top)),(int(right), int(bottom)), (0,222,0))

            label = 'person %.2f' % conf    

            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            rec1 = cv.rectangle(rec, (int(left), int(top - labelSize[1])), (int(left + labelSize[0]), int(top + baseLine)), (255, 255, 255), cv.FILLED)
            rec2 = cv.putText(rec1, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            return rec2

        layerNames = net.getLayerNames()
        lastLayerId = net.getLayerId(layerNames[-1])
        lastLayer = net.getLayer(lastLayerId)

        classIds = []
        confidences = []
        boxes = []
        if lastLayer.type == 'DetectionOutput':
        
            change_fame_W = (640-300)/100
            change_fame_H = (460 -300)/100
            
            for out in outs:
                for detection in out[0, 0]:
                    confidence = detection[2]
                    if confidence > confidence_threshold:
                        left = int(detection[3]) + change_fame_W
                        top = int(detection[4]) + change_fame_H
                        right = int(detection[5])- change_fame_W
                        bottom = int(detection[6])- change_fame_H
                        width = right - left + 5
                        height = bottom - top + 5
                        if width <= 2 or height <= 2:
                            left = int(detection[3] * frameWidth)+ 5
                            top = int(detection[4] * frameHeight)+ 5
                            right = int(detection[5] * frameWidth)- 5
                            bottom = int(detection[6] * frameHeight)- 5
                            width = right - left + 5
                            height = bottom - top + 5
                        classIds.append(int(detection[1]) - 1)  
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])
                        
        elif lastLayer.type == 'Region':
        
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > confidence_threshold:
                        center_x = int(detection[0] * frameWidth)
                        center_y = int(detection[1] * frameHeight)
                        width = int(detection[2] * frameWidth)
                        height = int(detection[3] * frameHeight)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)
                        classIds.append(classId)
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])
        else:
            print('Unknown output layer type: ' + lastLayer.type)
            exit()

        if len(net.getUnconnectedOutLayersNames()) > 1 or lastLayer.type == 'Region' and args.backend != cv.dnn.DNN_BACKEND_OPENCV:
            indices = []
            classIds = np.array(classIds)
            boxes = np.array(boxes)
            confidences = np.array(confidences)
            unique_classes = set(classIds)
            for cl in unique_classes:
                class_indices = np.where(classIds == cl)[0]
                conf = confidences[class_indices]
                box  = boxes[class_indices].tolist()
                nms_indices = cv.dnn.NMSBoxes(box, conf, confidence_threshold, nmsThreshold)
                nms_indices = nms_indices[:, 0] if len(nms_indices) else []
                indices.extend(class_indices[nms_indices])
        else:
            indices = np.arange(0, len(classIds))

        for i in indices:
            box = boxes[i]
            left = box[0]+14
            top = box[1]+14
            width = box[2]+14
            height = box[3]+5
            frame_pre = drawPred("person", confidences[i], int(left)  , int(top), int(left + width), int(top + height))
            txtconf = confidences[i]
        
        return frame_pre, ["persion",txtconf]



    def handle_callback(frame ,result_queue) :
    
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        
        if not frame is None:
            # Create a 4D blob from a frame.
            
            # image = frame.to_ndarray(format="bgr24")
        
            blob = cv.dnn.blobFromImage(frame, size=(frameWidth, frameHeight), swapRB=True, ddepth=cv.CV_8U)
            

            # Run a model
            
            net.setInput(blob)
            if net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
                frame = cv.resize(frame, (300, 300))
                net.setInput(np.array([[300, 300, 1.6]], dtype=np.float32), 'im_info')

            if args.asyncN:
                    net.forwardAsync()
            else:
                outs = net.forward(net.getUnconnectedOutLayersNames())
                annotated_image, result = _annotate_image(frame, outs)
                result_queue.append(result)
                annotated_image =  np.array(annotated_image)
                
        

        return annotated_image, result_queue

    result_queue = []    
    FW = st.image([])
    cap = cv.VideoCapture(0)
    cbxCamera = st.checkbox("Camera",key="mocamere1")
    cbxLabel = st.checkbox("Hiện nhãn", value=True)
    tblConf = st.table([])
    df = None
    print(cbxCamera)
    while cbxCamera:
        _, frame = cap.read()
        # cap.set(cv.CAP_PROP_ZOOM, 0x8004)
        frame_pre, result_queue = handle_callback(frame, result_queue)
        
        if len(frame_pre.shape)==2 or len(frame_pre.shape)==3:
            frame_pre = cv.cvtColor(frame_pre, cv.COLOR_BGR2RGB)
            FW.image(frame_pre) 
            if cbxLabel:
                result_queue = sorted(result_queue, key = lambda x: x[1])
                df = pd.DataFrame(result_queue[-2:],columns=( ["name", "confiden"]))
                tblConf.table(df)
        
    st.markdown(
        "bài làm deloy detection with streamlit"
        
    )
    