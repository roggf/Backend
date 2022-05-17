import torch
import cv2
import socket


class Backend:

    def __init__(self):
        self.model = self.load_model()
        self.pipeline = self.load_pipeline()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice used:", self.device)

    @staticmethod
    def socket_send(message):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('192.168.178.124', 6969))
        s.listen(5)

        while True:
            clientsocket, address = s.accept()
            print(f"Duplicate was send to {address}!")
            clientsocket.send(bytes(message, encoding="utf-8"))

    @staticmethod
    def load_model():
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='Model/last.pt', force_reload=True)
        return model

    @staticmethod
    def load_pipeline():
        cap_receive = cv2.VideoCapture(
            'udpsrc port=1900 ! application/x-rtp, clock-rate=90000, encoding-name=JPEG, payload=26 ! rtpjpegdepay ! '
            'jpegdec ! videoconvert ! appsink ! drop=1 ',
            cv2.CAP_GSTREAMER)
        print('pipeline loaded')
        return cap_receive

    def score_frame(self, frame):
        self.model.to(self.device)
        results = self.model(frame)
        panda = results.pandas().xyxy[0]

        return panda

    @staticmethod
    def find_duplicates(panda):
        dups = str(panda.loc[panda['name'].duplicated(keep=False), 'name'].unique())  # *[0]
        if not dups:
            dups = 'empty'
            return dups
        else:
            return dups

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

        cords = cords.append(int(x_min_1 * x_shape), int(y_min_1 * y_shape), int(x_max_1 * x_shape),
                             int(y_max_1 * y_shape))
        cords = cords.append(int(x_min_2 * x_shape), int(y_min_2 * y_shape), int(x_max_2 * x_shape),
                             int(y_max_2 * y_shape))

        return cords

    def __call__(self):
        print('Backend is up')
        print(self.pipeline)
        if not self.pipeline.isOpened():
            print('VideoCapture not opened')
        while True:
            ret, frame = self.pipeline.read()
            print(ret)
            print(frame)
            if not ret:
                print("empty frame!")
            if ret:
                panda = self.score_frame(frame)
                duplicates = self.find_duplicates(panda)
                if duplicates:
                    coordinates = self.get_coordinates(duplicates, panda, frame)
                    self.socket_send(coordinates)
    cv2.destroyAllWindows()


detection = Backend()
detection()
