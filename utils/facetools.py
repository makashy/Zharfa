import numpy as np
import dlib
import pandas as pd
from utils.inception_resnet_v1 import InceptionResNetV1

pd.set_option('mode.chained_assignment', 'raise')


def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    y = (x - mean) / std_adj
    return y


def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(
        np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def compare(vec1, vec2):
    return np.linalg.norm(vec1 - vec2) < 0.55


def compare2(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

class WatchDog():
    def __init__(self, detector_path, predictor_path, recognizer_path, device_id):
        dlib.cuda.set_device(device_id)
        self.detector = dlib.cnn_face_detection_model_v1(
            detector_path)  #dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.recognizer = InceptionResNetV1(weights_path=recognizer_path, device_id=str(device_id))
        #dlib.face_recognition_model_v1(face_rec_model_path)

    def identify(self, image):  # TODO : images
        detected_faces = self.detector(image)
        if len(detected_faces) < 1:
            return None

        face_points = dlib.full_object_detections()
        detection_confidences = []
        detection_rects = []
        for _, person in enumerate(detected_faces):
            face_points.append(self.predictor(image, person.rect))
            detection_confidences.append(person.confidence)
            detection_rects.append(person.rect)

        face_chips = np.array(
            dlib.get_face_chips(image, face_points, 160, 0.25))
        # print(np.array(face_chips).shape)
        # if len(np.array(face_chips).shape)<4:
        #     face_chips = np.expand_dims(np.array(face_chips),axis=0)
        embeddings = self.recognizer.predict(prewhiten(face_chips))

        embeddings = l2_normalize(embeddings)

        result = {
            'Size': len(detected_faces),
            'DetectionConfidences': detection_confidences,
            'DetectionRects': detection_rects,
            'FacePoints': face_points,
            'FaceChips': face_chips,
            'RecognitionID': embeddings
        }

        return result
