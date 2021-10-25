import sys
import dlib
import cv2
import numpy as np

width = 640  # 定义摄像头获取图像垂直宽度
height = 480  # 定义摄像头获取水平宽度
disp_width = 1920  # 展示在笔记本中的图像水平宽度
disp_height = 1080  # 展示在笔记本中的图像水平宽度
detector = dlib.get_frontal_face_detector()  # 人脸检测画框
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # 标记人物68个关键点
cam = cv2.VideoCapture('eeeeee.mp4')  # 打开笔记本的内置摄像头
cam.set(3, width)  # 设置图像水平宽度
cam.set(4, height)  # 设置图像垂直宽度
color_focus = (255, 0, 0)  # 焦点的检测框为蓝色
color_figure = (255, 255, 255)  # 非焦点的检测框为白色
line_width = 3
focus = 0
N_width = disp_width/width  # 水平宽度归一化
N_height = disp_height/height  # 垂直宽度归一化

while True:
    ret_val, img = cam.read()
    print("图像分辨率为", len(img[1]), "x", len(img))
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR到RGB的转换
    disp_img = cv2.resize(img, (disp_width, disp_height), interpolation=cv2.INTER_AREA)  # 将原始图像调整为指定大小并进行重采样
    dets = detector(rgb_image)
    i = 0  # 视频中的人数
    center = []  # 存放检测结果的数组
    for det in dets:
        frame_horizontal = (det.right()+det.left())/2  # 检测框中点横坐标
        frame_vertical = (det.top()+det.bottom())/2  # 检测框中点纵坐标
        frame_area = (det.right()-det.left())*(det.bottom()-det.top())/(width*height)  # 检测框面积
        target = ((np.square(frame_horizontal-width/2)+np.square(frame_vertical-height/2)) /
                  (pow(width/2, 2)+pow(height/2, 2)))*0.1+1-frame_area  # 确定焦点的指标
        center.append(target)  # append函数会在center数组后加上target产生的元素

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
                cv2.rectangle(disp_img, (round(det.left()*N_width), round(det.top()*N_height)),
                              (round(det.right()*N_width), round(det.bottom()*N_height)), color_focus, line_width)
                cv2.putText(disp_img, 'FOCUS', (round(det.left()*N_width), round(det.top()*N_height)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), line_width)
            else:  # 检测到脸但不是焦点
                cv2.rectangle(disp_img, (round(det.left()*N_width), round(det.top()*N_height)),
                              (round(det.right()*N_width), round(det.bottom()*N_height)), color_figure, line_width)
        shape = predictor(img, det)  # 定位人脸关键点
        for p in shape.parts():  # 获取点集合组成形状
            cv2.circle(disp_img, (round(p.x*N_width), round(p.y*N_height)), 2, (0, 255, 0), -1)
        i = i+1
    print(" ------------------------------------------------------------------------------------------------------")
    cv2.imshow('my webcam', disp_img)
    if cv2.waitKey(1) == 27:  # waitkey控制着imshow的持续时间
        break  # esc to quit
cv2.destroyAllWindows()



