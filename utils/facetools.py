import time
from multiprocessing import JoinableQueue, Process

import dlib
import numpy as np
import pandas as pd

from utils.advanced_queues import AdvancedQueue
from utils.inception_resnet_v1 import InceptionResNetV1
from utils.chronometer import Chronometer

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

#TODO: remove
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


class DetectionProcess(Process):
    def __init__(self, detector_path, device_id, input_queue):
        super().__init__()
        self.detector_path = detector_path
        self.device_id = device_id
        self.input_q = input_queue
        self.output_q = AdvancedQueue(1)
        self.stop_signal_q = JoinableQueue(1)
        self.chronometer = Chronometer()

    def run(self):
        dlib.cuda.set_device(self.device_id)
        detector = dlib.cnn_face_detection_model_v1(self.detector_path)
        while True:
            ################################################################
            if self.stop_signal_q.empty() is False:
                break

            ################################################################
            usable, item = self.input_q.empty_and_get()
            if usable:
                self.chronometer.start()
                print("============= DetectionProcess: ", time.time())
                self.input_q.task_done()
                frame = item
                if frame is None:
                    detection_result = None
                else:
                    detection_result = detector(frame)
                    detection_confidences = []
                    detection_rects = []
                    for _, person in enumerate(detection_result):
                        detection_confidences.append(person.confidence)
                        detection_rects.append(person.rect)

                    if len(detection_result) < 1:
                        detection_result = None
                    else:
                        detection_result = {"detection_confidences":detection_confidences,
                                            "detection_rects":detection_rects}
                                            
                self.output_q.empty_and_put([frame, detection_result])
                print("D============ DetectionProcess: ", self.chronometer.average_elapsed())
                self.chronometer.start()
                self.output_q.join()
                print("D============: ", self.chronometer.give_elapsed())
                

    def end_process(self):
        self.stop_signal_q.put_nowait(True)
        time.sleep(1)  #TODO : change duration or remove!
        self.terminate()
        self.join()
        self.close()


class FaceEditionProcess(Process):
    """ Makes the necessary edits for face recognition:
    1. Predicts human face pose and identifies the location of important facial
        land marks (such as the corners of the mouth and eyes, tip of the nose)
    2. Crops faces, rotates them upright and scales them to 160x160 pixels
    """
    def __init__(self, predictor_path, device_id, input_queue):
        super().__init__()
        self.predictor_path = predictor_path
        self.device_id = device_id
        self.input_q = input_queue
        self.output_q = AdvancedQueue(1)
        self.stop_signal_q = JoinableQueue(1)
        self.chronometer = Chronometer()

    def run(self):
        """ Method representing the process’s activity.
        """
        dlib.cuda.set_device(self.device_id)
        predictor = dlib.shape_predictor(self.predictor_path)
        while True:
            ################################################################
            if self.stop_signal_q.empty() is False:
                break

            ################################################################
            usable, item = self.input_q.empty_and_get()
            if usable:
                self.chronometer.start()
                print("############# FaceCorrectionProcess: ", time.time())
                self.input_q.task_done()
                frame, detection_result = item
                if detection_result is None:
                    result = None
                else:
                    
                    face_points = dlib.full_object_detections()
                    detection_confidences = detection_result["detection_confidences"]
                    detection_rects = detection_result["detection_rects"]
                    for _, rect in enumerate(detection_rects):
                        face_points.append(predictor(frame, rect))
                    face_chips = np.array(dlib.get_face_chips(frame, face_points, 160, 0.25))
                    prewhiten_face_chips = prewhiten(face_chips)

                    result = {
                        'Size': len(detection_confidences),
                        'DetectionConfidences': detection_confidences,
                        'DetectionRects': detection_rects,
                        'FacePoints': face_points,
                        'FaceChips': face_chips,
                        'prewhiten_face_chips': prewhiten_face_chips
                    }
                self.output_q.empty_and_put([frame, result])
                print("D############ FaceCorrectionProcess: ", self.chronometer.average_elapsed())
                self.chronometer.start()
                self.output_q.join()
                print("D############: ", self.chronometer.give_elapsed())

    def end_process(self):
        self.stop_signal_q.put_nowait(True)
        time.sleep(1)  #TODO : change duration or remove!
        self.terminate()
        self.join()
        self.close()


class IdentificationProcess(Process):
    def __init__(self, recognizer_path, device_id, input_queue):
        super().__init__()
        self.recognizer_path = recognizer_path
        self.device_id = device_id
        self.input_q = input_queue
        self.output_q = AdvancedQueue(1)
        self.stop_signal_q = JoinableQueue(1)
        self.chronometer = Chronometer()

    def run(self):
        recognizer = InceptionResNetV1(weights_path=self.recognizer_path, device_id=str(self.device_id))
        while True:
            ################################################################
            if self.stop_signal_q.empty() is False:
                break

            ################################################################
            usable, item = self.input_q.empty_and_get()
            if usable:
                self.chronometer.start()
                print("+++++++++++++IdentificationProcess: ", time.time())
                self.input_q.task_done()
                frame, edited_result = item
                if edited_result is None:
                    result = None
                else:
                    prewhiten_face_chips = edited_result["prewhiten_face_chips"]
                    embeddings = recognizer.predict(prewhiten_face_chips)
                    embeddings = l2_normalize(embeddings)
                    edited_result.update({'RecognitionID': embeddings})
                    del edited_result["prewhiten_face_chips"]
                    result = edited_result
                print("D++++++++++++IdentificationProcess: ", self.chronometer.average_elapsed())
                self.output_q.empty_and_put([frame, result])

    def end_process(self):
        self.stop_signal_q.put_nowait(True)
        time.sleep(1)  #TODO : change duration or remove!
        self.terminate()
        self.join()
        self.close()
