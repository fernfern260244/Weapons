import torch
import numpy as np
import cv2
import time


class ObjectDetection:
    
    def __init__(self, capture_index, model_name):

        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device:",self.device)


    def load_model(self,model_name):
        """
        โหลดโมเดล Yolo5 จากฮับ pytorch
        :return: โมเดล Pytorch ที่ผ่านการฝึกอบรม
        """
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model


    def score_frame(self, frame):
        """
        รับเฟรมเดียวเป็นอินพุต และให้เฟรมเรทโดยใช้โมเดล yolo5
        :param frame: ใส่เฟรมในรูปแบบ numpy/list/tuple
        :return: ป้ายกำกับและพิกัดของวัตถุที่ตรวจพบโดยโมเดลในเฟรม
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord


    def class_to_label(self, x):
        """
        สำหรับค่าป้ายกำกับที่กำหนด ให้ส่งคืนป้ายกำกับสตริงที่เกี่ยวข้อง
        :param x: ป้ายตัวเลข
        :return: ป้ายกำกับสตริงที่เกี่ยวข้อง
        """
        return self.classes[int(x)]


    def plot_boxes(self, results, frame):
        """
        รับเฟรมและผลลัพธ์เป็นอินพุต และพล็อตกล่องที่มีขอบเขตและป้ายกำกับไปที่เฟรม
        :param results: มีป้ายกำกับและพิกัดที่คาดการณ์โดยโมเดลในเฟรมที่กำหนด
        :param frame: เฟรมที่ได้คะแนนแล้ว
        :return: เฟรมที่มีกรอบล้อมรอบและป้ายกำกับที่วาดไว้
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3 :
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame


    def __call__(self):
        """
        ฟังก์ชันนี้ถูกเรียกเมื่อคลาสทำงาน มันรันลูปเพื่ออ่านวิดีโอทีละเฟรม
        และเขียนผลลัพธ์ลงในไฟล์ใหม่
        :return: void
        """
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time.perf_counter()
            fps = 1 / np.round(end_time - start_time, 3)
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
            cv2.imshow("img", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

#สร้างวัตถุใหม่และดำเนินการ
detection = ObjectDetection(capture_index=0,model_name='best.pt')
detection()