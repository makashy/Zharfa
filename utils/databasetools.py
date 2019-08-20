import time
import os

import numpy as np
import pandas as pd

import cv2

pd.set_option('mode.chained_assignment', 'raise')

def compare(vec1, vec2):
    return np.linalg.norm(vec1 - vec2) < 1


class DataBase():

    def __init__(self, database_address):
        self.database_address = database_address

        if 'database.pkl' in os.listdir(database_address):
            self.dataframe = pd.read_pickle(database_address + 'database.pkl')
        else:
            self.dataframe = pd.DataFrame()

    def check_face(self, embedding):
        for i in range(self.dataframe.index.size):
            if compare(self.dataframe.RecognitionID[i], embedding):
                return i
        return None

    def add_person(self,
                   id_num,
                   detect_time,
                   face_chip,
                   recognition_id,
                   first_name='Unknown',
                   last_name='Unknown'):

        image_address = self.database_address + 'images/' + str(
            id_num) + '.jpg'
        cv2.imwrite(image_address, face_chip)
        new_person = {
            'ID': id_num,
            'FirstName': first_name,
            'LastName': last_name,
            'History': [np.array(detect_time)],
            'TimeFlag': detect_time,
            'RecognitionID': [recognition_id],
            'imageAddress': image_address
        }
        self.dataframe = self.dataframe.append(new_person, ignore_index=True)

    def add_history(self, index, current_time):
        #TODO : improve the logic
        if current_time - self.dataframe.TimeFlag[index] > 15:
            self.dataframe.at[index, 'History'] = np.append(
                self.dataframe.History[index], current_time)
        self.dataframe.at[index, 'TimeFlag'] = current_time

    def update(self, result_list):
        list_size = result_list['Size']
        add_list = []
        update_list = []
        correspondence_dict = {}
        for i in range(list_size):
            id_num = self.check_face(result_list['RecognitionID'][i])
            if id_num is None:
                add_list.append(i)
            else:
                update_list.append(id_num)
                correspondence_dict[i] = id_num

        current_time = time.time()
        for i, index in enumerate(add_list, start=100):
            id_num = int(current_time) * 1000 + i
            self.add_person(id_num, current_time,
                            result_list['FaceChips'][index],
                            result_list['RecognitionID'][index])
            correspondence_dict[index] = self.dataframe.index[-1]

        for index in update_list:
            self.add_history(index, current_time)

        return correspondence_dict

    def save_data_frame(self):
        self.dataframe.to_pickle(self.database_address + 'database.pkl')
