import torch
import cv2
import socket
import numpy as np


class Backend:

    def __int__(self):
        self.model = self.load_moddel()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice used:", self.device)

    @staticmethod
    def socket_send(dups):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind("192.168.178.122", 6969)
        s.listen(5)

        while True:
            clientsocket, adress = s.accept()
            print(f"Duplicate was send to {adress}!")
            clientsocket.send(bytes(dups, encoding="utf-8"))

    @staticmethod
    def load_model():
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='../Model/best.pt', force_reload=True)
        return model

    def score_frame(self):
        cap_receive = cv2.VideoCapture('udpsrc port=4009 ! application/x-rtp, clock-rate=90000, encoding-name=JPEG,'
                                       'payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink',
                                       cv2.CAP_GSTREAMER)

        while True:
            ret, frame = cap_receive.read()
            if not ret:
                print("empty frame!")

            self.model.to(self.device)
            results = self.model(frame)
            dups_name, panda = self.find_duplicate(results)
            coordinates = self.get_coordinates(dups_name, panda)
            self.socket_send(coordinates)

    @staticmethod
    def find_duplicates(results):
        panda = results.panda().xyxy[0]
        dups = str(panda.loc[panda['name'].duplicated(keep=False), 'name'].unique()[0])
        return dups, panda

    @staticmethod
    def get_coordinates(dups, panda, frame):
        cords = []
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        index = panda.index
        condition = panda['name'] == dups
        dups_index = index[condition].tolist()

        x_min_1 = panda.iat[dups_index[0], 0]
        y_min_1 = panda.iat[dups_index[0], 1]
        x_max_1 = panda.iat[dups_index[0], 2]
        y_max_1 = panda.iat[dups_index[0], 3]

        x_min_2 = panda.iat[dups_index[1], 0]
        y_min_2 = panda.iat[dups_index[1], 1]
        x_max_2 = panda.iat[dups_index[1], 2]
        y_max_2 = panda.iat[dups_index[1], 3]

        cords = cords.append(int(x_min_1*x_shape), int(y_min_1*y_shape), int(x_max_1*x_shape), int(y_max_1*y_shape))
        cords = cords.append(int(x_min_2*x_shape), int(y_min_2*y_shape), int(x_max_2*x_shape), int(y_max_2*y_shape))

        return cords

    def __call__(self):
        while True:
            self.score_frame()


detection = Backend()
detection()
