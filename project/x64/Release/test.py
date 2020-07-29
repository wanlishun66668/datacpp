
from datacpp import data
import numpy as np
import cv2
import time
'''img=cv2.imread("time1.jpg")
dst=datacpp.test_img(img)
cv2.imshow("dst",dst[0])
cv2.waitKey(0)'''


'''file_name="D:/darknet/DataLoader/data/labels.txt"
label_dir="D:/darknet/DataLoader/data/label/"
image_dir = "D:/darknet/DataLoader/data/img/"
label_type = "xml"
image_type=".bmp"'''
file_name="D:/datasets/VOC/VOCdevkit/VOC2012/ImageSets/Main/train.txt"
label_dir="D:/datasets/VOC/VOCdevkit/VOC2012/Annotations/"
image_dir = "D:/datasets/VOC/VOCdevkit/VOC2012/JPEGImages/"
label_type = "detection"
label_postfix = "xml"
image_type=".jpg"
#result=datacpp.add1(file_name, 512, 512, 4, label_dir, label_type,4,True)
#print(result)
import time
height=512
width=512
batch_size=16

d=data.paramset(image_dir,file_name, height, width, batch_size, label_dir,label_type,label_postfix,image_type, 4,True)
transform=data.transform_image()
transform.vertical_flip(d)
transform.rand_rotate_90(d)
transform.rand_blur(d)
transform.rand_crop(d)
for i in range(1000):
    start=time.time()
    batch=data.next_batch(d)
    end=time.time()
    print("time--{}: {}".format(i,end-start))
    if(d.out_flag):
        d=data.paramset(image_dir,file_name, height, width, batch_size, label_dir, label_type,label_postfix,image_type, 4,True)
        transform.vertical_flip(d)
        transform.rand_rotate_90(d)
        transform.rand_blur(d)
        transform.rand_crop(d)
        continue
    print("batch_count:",d.batch_count)
    images=batch.images
    labels=batch.labels
    '''for i in range(len(images)):
        for j in range(len(labels[i])):
            cv2.rectangle(images[i],(labels[i][j][0],labels[i][j][1]),(labels[i][j][2],labels[i][j][3]),(255,0,0),thickness=1)
        cv2.imshow("images:",images[i])
        cv2.waitKey(0)'''
    
    '''for i in range(len(images)):
        #cv2.imshow("image",images[i])
        dst=cv2.resize(labels[i][0],(512,512))
        #cv2.imshow("label",dst)
        #cv2.waitKey(0)
        dst=cv2.cvtColor(dst,cv2.COLOR_GRAY2BGR)
        dst=(dst*255).astype(np.uint8)
        #cv2.imshow("m",dst)
        #cv2.waitKey(0)
        image=images[i].astype(np.uint8)
        cv2.imshow("src",image)
        dst=cv2.addWeighted(image,0.5,dst,0.5,0)
        cv2.imshow("tt",dst)
        cv2.waitKey(0)'''