import os
import cv2 as cv

cnt = 0
OUTPATH = "./out/"
for images in os.listdir("."):
    cnt += 1
    if (images.endswith(".jpg")):
        print(images)
        img = cv.imread(images)
        h, w, c = img.shape
        w = w * 200 / h
        h = 200
        img = cv.resize(img, (int(w), int(h)))
        imgs = []
        for i in range(int(h/w)):
            imgs.append(img[int(i*w):int((i+1)*w), 0:int(w)])
        rest_h = h - int(h/w) * w
        if (rest_h > 0):
            imgs.append(img[int(h-rest_h):int(h), 0:int(w)])
        for i in range(len(imgs)):
            cv.imwrite(OUTPATH+"img"+str(cnt)+"-"+str(i)+".jpg", imgs[i])
        
