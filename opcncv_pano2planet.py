import os
import sys
import cv2
import numpy as np
import dlib
import time
from PIL import Image

class Equirectangular:
    Delta_position_x = 0
    Delta_position_y = 0
    def __init__(self, img_name):
        self._img = img_name
        #self._img = cv2.imread(img_name, cv2.IMREAD_ANYCOLOR)
        [self._height, self._width, _] = self._img.shape
        print(self._width,self._height)

    def GetPerspective(self, FOV, THETA, PHI, height, width, RADIUS=128):
        # THETA is left/right angle, PHI is up/down angle, both in degree

        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wFOV = FOV
        hFOV = float(height) / width * wFOV

        c_x = (width - 1) / 2.0
        c_y = (height - 1) / 2.0

        wangle = (180 - wFOV) / 2.0
        w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
        w_interval = w_len / (width - 1)

        hangle = (180 - hFOV) / 2.0
        h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
        h_interval = h_len / (height - 1)
        x_map = np.zeros([height, width], np.float32) + RADIUS
        y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
        z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
        D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
        xyz = np.zeros([height, width, 3], np.float64)
        xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
        xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
        xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2] / RADIUS)
        lon = np.zeros([height * width], np.float64)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(np.bool_)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool_)

        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy
        outImage = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_WRAP)
        return outImage

    def face_detection(image):
        # 获得检测器
        face_detector = dlib.get_frontal_face_detector()

        # 加载关键点模型
        shape_detection = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将照片转化为灰度

        faces = face_detector(image_gray, 1)  # 检测人脸数，1为放大后检测

        # 绘制矩形框和特征点
        for face in faces:
            # 绘制矩形框
            cv2.rectangle(image, (int(face.left()), int(face.top())),
                          (int(face.right()), int(face.bottom())), (0, 0, 255), 3)
            shapes = shape_detection(image_gray, face)  # 检测人脸特征点

            # font = cv2.FONT_HERSHEY_SIMPLEX

            # 画出人脸特征点
            for pt in shapes.parts():
                pt_position = (pt.x, pt.y)
                cv2.circle(image, pt_position, 1, (0, 0, 255), -1)

    def facedetect(outImage):
        
        detector = dlib.get_frontal_face_detector()  # 人脸检测画框
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # 标记人物68个关键点
        width = 640  # 定义摄像头获取图像垂直宽度
        height = 480  # 定义摄像头获取水平宽度
        disp_width = 640  # 展示在笔记本中的图像水平宽度
        disp_height = 480  # 展示在笔记本中的图像水平宽度
        color_focus = (255, 0, 0)  # 焦点的检测框为蓝色
        color_figure = (255, 255, 255)  # 非焦点的检测框为白色
        line_width = 3
        N_width = disp_width / width  # 水平宽度归一化
        N_height = disp_height / height  # 垂直宽度归一化
        gray = cv2.cvtColor(outImage, cv2.COLOR_BGR2GRAY)  # BGR到GRAY的转换
        dets = detector(gray)
        i = 0  # 视频中的人数
        center = []  # 存放检测结果的数组
        for det in dets:
            frame_horizontal = (det.right() + det.left()) / 2  # 检测框中点横坐标
            frame_vertical = (det.top() + det.bottom()) / 2  # 检测框中点纵坐标
            frame_area = (det.right() - det.left()) * (det.bottom() - det.top()) / (width * height)  # 检测框面积
            target = ((np.square(frame_horizontal - width / 2) + np.square(frame_vertical - height / 2)) /
                      (pow(width / 2, 2) + pow(height / 2, 2))) * 0.1 + 1 - frame_area  # 确定焦点的指标
            center.append(target)  # append函数会在center数组后加上target产生的元素
            shape = predictor(img, det)  # 定位人脸关键点
            if len(center) == 0:  # 若没有检测到脸
                focus = 100
            else:  # 若检测到脸
                focus = center.index(min(center))

            print("第", i, "张人脸")
            print("人脸坐标为", frame_horizontal, frame_vertical)
            print("检测结果为", target)
            print("焦点为", "第", focus, "张人脸")

            if focus == 100:  # 若未检测到脸
                print(focus)
                pass  # 跳出循环
            else:
                # print(focus)
                if i == focus:  # 检测到脸且为焦点
                    cv2.rectangle(outImage, (round(det.left() * N_width), round(det.top() * N_height)),
                                  (round(det.right() * N_width), round(det.bottom() * N_height)), color_focus,
                                  line_width)
                    cv2.putText(outImage, 'FOCUS', (round(det.left() * N_width), round(det.top() * N_height)),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                else:  # 检测到脸但不是焦点
                    cv2.rectangle(outImage, (round(det.left() * N_width), round(det.top() * N_height)),
                                  (round(det.right() * N_width), round(det.bottom() * N_height)), color_figure,
                                  line_width)
            shape = predictor(img, det)  # 定位人脸关键点
            for p in shape.parts():  # 获取点集合组成形状
                cv2.circle(outImage, (round(p.x * N_width), round(p.y * N_height)), 2, (0, 255, 0), -1)
            i = i + 1


    def CallbackFunc(event, x, y, flags, param):
        i=0
        #p_x1 = 0
        #p_y2 = 0
        if event == cv2.EVENT_LBUTTONDOWN:
            p_x = x
            p_y = y
            while True:
                ret, frame = cap.read()
                equ = Equirectangular(frame)  # Load equirectangular image
                img_1 = equ.GetPerspective(80, p_x, 0, 480, 640)  # Specify parameters(FOV, theta, phi, height, width)
                Equirectangular.facedetect(img_1)
                cv2.imshow("show", img_1)
                cv2.waitKey(1)

if __name__ == '__main__':
    cap = cv2.VideoCapture('more.mp4')
    while (cap.isOpened()):
        ret, frame = cap.read()
        equ = Equirectangular(frame)
        img = equ.GetPerspective(80, 200, -10, 480, 640)  # Specify parameters(FOV, theta, phi, height, width)
        cv2.imshow('show', img)
        print('img:',len(img[1]),len(img))
        cv2.setMouseCallback("show", Equirectangular.CallbackFunc)
        cv2.waitKey(1)





