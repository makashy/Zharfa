import time
import numpy as np
import dlib
import pandas as pd
from utils.inception_resnet_v1 import InceptionResNetV1
from multiprocessing import JoinableQueue, Process
from utils.advanced_queues import AdvancedQueue

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
                print("22222222222222222222 Det starts at: ", time.time())
                self.input_q.task_done()
                frame = item
                if frame is None:
                    self.output_q.empty_and_put(None)
                    print("33333333333333333333 Det finishes at: ", time.time(), "NONE")
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
                    print("33333333333333333333 Det finishes at: ", time.time())
                self.output_q.join()

    def end_process(self):
        self.stop_signal_q.put_nowait(True)
        time.sleep(1)  #TODO : change duration or remove!
        self.terminate()
        self.join()
        self.close()


class IdentificationProcess(Process):
    def __init__(self, predictor_path, recognizer_path, device_id, input_queue):
        super().__init__()
        self.predictor_path = predictor_path
        self.recognizer_path = recognizer_path
        self.device_id = device_id
        self.input_q = input_queue
        self.output_q = AdvancedQueue(1)
        self.stop_signal_q = JoinableQueue(1)

    def run(self):
        dlib.cuda.set_device(self.device_id)
        predictor = dlib.shape_predictor(self.predictor_path)
        recognizer = InceptionResNetV1(weights_path=self.recognizer_path, device_id=str(self.device_id))
        while True:
            ################################################################
            if self.stop_signal_q.empty() is False:
                break

            ################################################################
            usable, item = self.input_q.empty_and_get()
            if usable:
                print("44444444444444444444 Iden starts at: ", time.time())
                self.input_q.task_done()
                if item is None:
                    self.output_q.empty_and_put(None)
                    print("5555555555555555555 Iden finishes at: ", time.time(), "NONEEEE")
                else:
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
                        embeddings = recognizer.predict(prewhiten(face_chips))
                        embeddings = l2_normalize(embeddings)
                        result = {
                            'Size': len(detection_confidences),
                            'DetectionConfidences': detection_confidences,
                            'DetectionRects': detection_rects,
                            'FacePoints': face_points,
                            'FaceChips': face_chips,
                            'RecognitionID': embeddings
                        }
                    self.output_q.empty_and_put([frame, result])
                    print("5555555555555555555 Iden finishes at: ", time.time())

    def end_process(self):
        self.stop_signal_q.put_nowait(True)
        time.sleep(1)  #TODO : change duration or remove!
        self.terminate()
        self.join()
        self.close()