```
#將圖片偵測特徵點,並輸出眼睛位置相關資訊至已有紅點資訊文檔中
import cv2
import dlib
import numpy
import numpy as np
import sys
import os
    
#===============讀取檔案內的所有圖片資料===============
def read_directory(directory_name):
    txtNum = 1000 #文件建檔的標記數
    path_list=os.listdir(directory_name)
    path_list.sort(key=lambda x:int(x[0:-4])) #將資料進行排序
    PREDICTOR_PATH = "C:/Users/Lucy/Desktop/Eyesdetect/shape_predictor_81_face_landmarks.dat" 
    
    #===============檔案內的圖片讀取loop=============== 
    for filename in path_list:   
        
        detector = dlib.get_frontal_face_detector()      #使用dlib自帶的frontal_face_detector作為人臉提取器
        predictor = dlib.shape_predictor(PREDICTOR_PATH) #使用官方提供的模型建構特徵提取器 
                                   
        print(filename+" loading...") #目前處理的filename
        with open("C:/Users/Lucy/Desktop/Eyesdetect/text/" + str(txtNum) + ".txt", "r") as text:
            redDotContent = text.read() #先讀取目前紅點資訊並暫存
        im = cv2.imread(directory_name + "/" + filename)
        rects = detector(im,1) #使用detector進行人臉檢測 rects為返回的結果
        
        #輸出人臉數 rects的元素個數為人臉的個數
        if len(rects) >= 1:
            print("{} faces detected".format(len(rects)))
    
        elif len(rects) == 0:
            pass   
        
        #===============以人臉數目為loop結束標準===============
        sitNum = 0 #用來插入眼睛框資訊該放置的位置(寫入文件時必須用到)
        for i in range(len(rects)):  
            landmarks = numpy.matrix([[p.x,p.y] for p in predictor(im,rects[i]).parts()])
            im = im.copy()
            minX = 0
            minY = 0
            maxX = 0
            maxY = 0
              
            #===============透過特徵點序號處理眼睛相關位置資訊===============
            for idx,point in enumerate(landmarks):
                pos = (point[0,0],point[0,1])
                detector = dlib.get_frontal_face_detector()
                if ( idx == 17 ):               #設定初始值(以雙眼最初特徵點序號開始)
                    x = (point[0,0])
                    y = (point[0,1])
                    maxX = x
                    maxY = y
                    minX = x
                    minY = y
            
                if ( idx >= 17 and idx <= 29 ): #雙眼範圍 序號17~29 取最大最小值
                    tempX = (point[0,0])
                    tempY = (point[0,1])
                    maxX = max(maxX,tempX)
                    maxY = max(maxY,tempY)
                    minX = min(minX,tempX)
                    minY = min(minY,tempY)
                    
            size = 5            #為的使眼睛框更符合雙眼外圍
            pointMinX = minX-size
            pointMinY = minY-size
            pointMaxX = maxX+size
            pointMaxY = maxY+size 
        
            point1 = ( pointMinX, pointMinY )
            point2 = ( pointMaxX, pointMaxY )
        
            w = maxX - minX     #眼睛框之 寬
            h = maxY - minY     #眼睛框之 高
       
            #central point
            Cx = (maxX+minX)/2  #眼睛框的中心位置(兩眼中央)_x座標
            Cy = (maxY+minY)/2  #眼睛框的中心位置(兩眼中央)_y座標
        
            point1 = ( minX, minY )
            point2 = ( maxX, maxY )
        
            '''
            #Upper center point
            X = Cx - (h/2)      #眼睛框上方中央位置_x座標
            Y = Cy - (h/2)      #眼睛框上方中央位置_y座標
            '''
        
            w.astype('float64')
            h.astype('float64')
            Cx.astype('float64')
            Cy.astype('float64')
        
            w = w/1280
            h = h/720
            Cx = Cx/1280
            Cy = Cy/720
        
            #畫出眼睛框於圖片上並輸出
        
            pic = cv2.rectangle(im, point1, point2, (255,255,255), 2)
            cv2.imwrite("C:/Users/Lucy/Desktop/Eyesdetect/images/" + str(txtNum) + "_DoubleCheck.png",im)
                  
        #===============以原先文件檔(已存有紅點資料)在其前面插入眼睛位置資訊===============
            with open("C:/Users/Lucy/Desktop/Eyesdetect/text/" + str(txtNum) + ".txt", "r+") as text:                     
                text.seek(sitNum, 0)
                if ( sitNum == 0 ) : #不只表示插入位置更表示這文件有一筆以上的內容
                    text.write("0"+" "+str(round(Cx,5))+" "+str(round(Cy,5))+" "+str(round(w,5))+" "+str(round(h,5))+" "+redDotContent+"\n")         
                elif ( sitNum > 0 ) :
                    dataContent = text.read() #讀取目前文件所有內容並暫存
                    text.write("0"+" "+str(round(Cx,5))+" "+str(round(Cy,5))+" "+str(round(w,5))+" "+str(round(h,5))+" "+redDotContent+"\n")         
                text.close() 
                sitNum = sitNum + 1
        #==============================================================================
        
        txtNum = txtNum + 1
        print(filename+" finish!")
        
        #顯示圖檔
        ''' 
        cv2.namedWindow("im",2)
        cv2.imshow("im",im)
        cv2.waitKey(0)
        '''     
        
read_directory("C:/Users/Lucy/Desktop/Eyesdetect/images")
cv2.destroyAllWindows()  
```
