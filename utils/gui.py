import time

from kivy.graphics.texture import Texture #pylint: disable=no-name-in-module

import cv2 #pylint: disable=import-error


def check_source(mode, root_widget):#TODO
    if mode == 'Camera':
        root_widget.ids['source_address'].disabled = True
        root_widget.ids['camera_number'].disabled = False
    elif mode == 'IP Camera':
        root_widget.ids['source_address'].disabled = False
        root_widget.ids['camera_number'].disabled = True
    elif mode == 'Video':
        root_widget.ids['source_address'].disabled = False
        root_widget.ids['camera_number'].disabled = True
    elif 'Demo' in mode:
        root_widget.ids['source_address'].disabled = True
        root_widget.ids['camera_number'].disabled = True

def add_face_boxes(frame, rect, color=(0, 255, 0), thickness=3):
    cv2.rectangle(frame, (rect.right(), rect.top()),
                  (rect.left(), rect.bottom()), color, thickness)
    return frame

def add_name(frame,
             text,
             rect,
             font_face=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=1,
             color=(255, 255, 255),
             thickness=2,
             line_type=cv2.LINE_AA):
    cv2.putText(frame, text, (rect.left(), rect.bottom() + 10), font_face,
                font_scale, color, thickness, line_type)  #TODO : 10?
    return frame

def add_id(frame,
           text,
           rect,
           font_face=cv2.FONT_HERSHEY_SIMPLEX,
           font_scale=1,
           color=(255, 0, 0),
           thickness=2,
           line_type=cv2.LINE_AA):
    cv2.putText(frame, text, (rect.left(), rect.bottom() + 20), font_face,
                font_scale, color, thickness, line_type)  #TODO : 20?
    return frame

def get_kivy_texture(frame):
    buf = cv2.flip(frame, 0)
    buf = buf.tostring()
    texture = Texture.create(size=(frame.shape[1], frame.shape[0]),
                             colorfmt='bgr')
    texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    return texture

def open_cv2_window():
    cv2.namedWindow("Zharfa  (Large view)")

def update_cv2_window(frame):
    cv2.imshow("Zharfa  (Large view)", frame)

def close_cv2_window():
    cv2.destroyAllWindows()
