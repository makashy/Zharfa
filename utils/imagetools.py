from kivy.graphics.texture import Texture

import cv2


class InputImage():
    def __init__(self, initial_source):
        self.source = cv2.VideoCapture(initial_source)

    def change_source(self, new_source):
        self.source = cv2.VideoCapture(new_source)
        self.source.set(cv2.CAP_PROP_FRAME_WIDTH, 1920 / 2)
        self.source.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080 / 2)

    def get_frame(self):
        for _ in range(5):
            ret, frame = self.source.read()
            if ret:
                break
            print('No data!')
            time.sleep(0.002)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ret, frame

    def add_face_boxes(self, frame, rect, color=(0, 255, 0), thickness=3):
        cv2.rectangle(frame, (rect.right(), rect.top()),
                      (rect.left(), rect.bottom()), color, thickness)
        return frame

    def add_name(self,
                 frame,
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

    def get_kivy_texture(self, frame):
        buf = cv2.flip(frame, 0)
        buf = buf.tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]),
                                 colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture

    def open_cv2_window(self):
        cv2.namedWindow("Zharfa  (Large view)")

    def update_cv2_window(self, frame):
        cv2.imshow("FA", frame)

    def close_cv2_window(self):
        cv2.destroyAllWindows()
