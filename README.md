# Face-detection-in-panoramic-video
利用python语言通过全景视频的rtmp地址将全景视频拉流，根据等距柱状投影原理，利用numpy库，opencv库和dlib库生成相对于某一个视点的平面图像。
通过参数传递入播放器，可实现全景视频跟随鼠标的拖动确定视点，并且显示该视点的平面图像，达到与PC机VLC播放器相同的效果，进一步加入Dlib的68个特征点的人脸检测模块，对平面图像执行人脸特征点检测。