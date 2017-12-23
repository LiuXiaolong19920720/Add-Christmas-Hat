import numpy as np 
import cv2
import dlib

# 给img中的人头像加上圣诞帽，人脸最好为正脸
def add_hat(img,hat_img):
    # 分离rgba通道，合成rgb三通道帽子图，a通道后面做mask用
    r,g,b,a = cv2.split(hat_img) 
    rgb_hat = cv2.merge((r,g,b))

    cv2.imwrite("hat_alpha.jpg",a)

    # ------------------------- 用dlib的人脸检测代替OpenCV的人脸检测-----------------------
    # # 灰度变换
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    # # 用opencv自带的人脸检测器检测人脸
    # face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")                       
    # faces = face_cascade.detectMultiScale(gray,1.05,3,cv2.CASCADE_SCALE_IMAGE,(50,50))

    # ------------------------- 用dlib的人脸检测代替OpenCV的人脸检测-----------------------

    # dlib人脸关键点检测器
    predictor_path = "shape_predictor_5_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)  

    # dlib正脸检测器
    detector = dlib.get_frontal_face_detector()

    # 正脸检测
    dets = detector(img, 1)

    # 如果检测到人脸
    if len(dets)>0:  
        for d in dets:
            x,y,w,h = d.left(),d.top(), d.right()-d.left(), d.bottom()-d.top()
            # x,y,w,h = faceRect  
            # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2,8,0)

            # 关键点检测，5个关键点
            shape = predictor(img, d)
            # for point in shape.parts():
            #     cv2.circle(img,(point.x,point.y),3,color=(0,255,0))

            # cv2.imshow("image",img)
            # cv2.waitKey()  

            # 选取左右眼眼角的点
            point1 = shape.part(0)
            point2 = shape.part(2)

            # 求两点中心
            eyes_center = ((point1.x+point2.x)//2,(point1.y+point2.y)//2)

            # cv2.circle(img,eyes_center,3,color=(0,255,0))  
            # cv2.imshow("image",img)
            # cv2.waitKey()

            #  根据人脸大小调整帽子大小
            factor = 1.5
            resized_hat_h = int(round(rgb_hat.shape[0]*w/rgb_hat.shape[1]*factor))
            resized_hat_w = int(round(rgb_hat.shape[1]*w/rgb_hat.shape[1]*factor))

            if resized_hat_h > y:
                resized_hat_h = y-1

            # 根据人脸大小调整帽子大小
            resized_hat = cv2.resize(rgb_hat,(resized_hat_w,resized_hat_h))

            # 用alpha通道作为mask
            mask = cv2.resize(a,(resized_hat_w,resized_hat_h))
            mask_inv =  cv2.bitwise_not(mask)

            # 帽子相对与人脸框上线的偏移量
            dh = 0
            dw = 0
            # 原图ROI
            # bg_roi = img[y+dh-resized_hat_h:y+dh, x+dw:x+dw+resized_hat_w]
            bg_roi = img[y+dh-resized_hat_h:y+dh,(eyes_center[0]-resized_hat_w//3):(eyes_center[0]+resized_hat_w//3*2)]

            # 原图ROI中提取放帽子的区域
            bg_roi = bg_roi.astype(float)
            mask_inv = cv2.merge((mask_inv,mask_inv,mask_inv))
            alpha = mask_inv.astype(float)/255

            # 相乘之前保证两者大小一致（可能会由于四舍五入原因不一致）
            alpha = cv2.resize(alpha,(bg_roi.shape[1],bg_roi.shape[0]))
            # print("alpha size: ",alpha.shape)
            # print("bg_roi size: ",bg_roi.shape)
            bg = cv2.multiply(alpha, bg_roi)
            bg = bg.astype('uint8')

            cv2.imwrite("bg.jpg",bg)
            # cv2.imshow("image",img)
            # cv2.waitKey()

            # 提取帽子区域
            hat = cv2.bitwise_and(resized_hat,resized_hat,mask = mask)
            cv2.imwrite("hat.jpg",hat)
            
            # cv2.imshow("hat",hat)  
            # cv2.imshow("bg",bg)

            # print("bg size: ",bg.shape)
            # print("hat size: ",hat.shape)

            # 相加之前保证两者大小一致（可能会由于四舍五入原因不一致）
            hat = cv2.resize(hat,(bg_roi.shape[1],bg_roi.shape[0]))
            # 两个ROI区域相加
            add_hat = cv2.add(bg,hat)
            # cv2.imshow("add_hat",add_hat) 

            # 把添加好帽子的区域放回原图
            img[y+dh-resized_hat_h:y+dh,(eyes_center[0]-resized_hat_w//3):(eyes_center[0]+resized_hat_w//3*2)] = add_hat

            # 展示效果
            # cv2.imshow("img",img )  
            # cv2.waitKey(0)  

            return img

   
# 读取帽子图，第二个参数-1表示读取为rgba通道，否则为rgb通道
hat_img = cv2.imread("hat2.png",-1)

# 读取头像图
img = cv2.imread("test.jpg")
output = add_hat(img,hat_img)

# 展示效果
cv2.imshow("output",output )  
cv2.waitKey(0)  
cv2.imwrite("output.jpg",output)
# import glob as gb 

# img_path = gb.glob("./images/*.jpg")

# for path in img_path:
#     img = cv2.imread(path)

#     # 添加帽子
#     output = add_hat(img,hat_img)

#     # 展示效果
#     cv2.imshow("output",output )  
#     cv2.waitKey(0)  

cv2.destroyAllWindows()  