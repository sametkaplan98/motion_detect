# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 01:07:38 2020

@author: Samet
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 00:11:05 2020

@author: Samet
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import time

MIN_MATCH_COUNT=25



video=cv2.VideoCapture(1)


while True:
    
 _,frame=video.read()
 orjinal=frame
 orjinal=imutils.resize(orjinal,360,360)
 time.sleep(0.1)   
          
 a,frame2=video.read()

 
 img1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 img1=cv2.GaussianBlur(img1,(15,15),0)
 
 img2=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
 img2=cv2.GaussianBlur(img2,(15,15),0)

 sift=cv2.xfeatures2d.SIFT_create()
 
 cv2.imshow('Orjinal Goruntu',orjinal)
 kp1,des1=sift.detectAndCompute(img1,None)

 kp2,des2=sift.detectAndCompute(img2,None) 

 sayac_src=0
 sayac_dst=0
 toplam_dst_x=0
 toplam_src_x=0
 
 toplam_dst_y=0
 toplam_src_y=0
 
 poz1_x=0
 pox2_x=0
 
 if des2 is not None : # Ortak feature bulunmadigi durumda hata vermemesi icin
  

  FLANN_INDEX_KDTREE=0
  index_params=dict(algorithm=FLANN_INDEX_KDTREE)          
  search_params=dict(checks=100)
 
  flann=cv2.FlannBasedMatcher(index_params,search_params)
 
  matches=flann.knnMatch(des1,des2,k=2)
  #print('Matches1 boyutu----------',len(matches))
  
  good=[]

  for m,n in matches:
    m.distance <0.7*n.distance
    good.append(m)
    #print( kp1[m.queryIdx].pt[0])
    #pt2 = kp2[n.trainIdx].pt
    
    #pt1=list(pt1)
    #pt2=list(pt2)
    #print('X Fark: ',pt1[0]-pt2[0])
    #print('Y Fark: ',pt1[1]-pt2[1])
       

  
  if len(good)>MIN_MATCH_COUNT :
   src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)    
   dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
   
   
   
   
   for j in src_pts:                        ## Birinci frame 
       #print(j[0,0])
       toplam_src_x=toplam_src_x+j[0,0]
       toplam_src_y=toplam_src_y+j[0,1]
       sayac_src+=1  
   x_src_ortalama=toplam_src_x/sayac_src
   y_src_ortalama=toplam_src_y/sayac_src
   #print('1.Frame ortalama Y: ',y_src_ortalama)
   
   cv2.circle(img1,(int(x_src_ortalama),int(y_src_ortalama)),6,(255,0,255),2)
   
   
   
   for i in dst_pts:                          ## İkinci frame 
       #print(i[0,0])
       toplam_dst_x=toplam_dst_x+i[0,0]
       toplam_dst_y=toplam_dst_y+i[0,1]
       sayac_dst+=1  
   x_dst_ortalama=toplam_dst_x/sayac_dst
   y_dst_ortalama=toplam_dst_y/sayac_dst
   #print('2.Frame ortalama Y: ',y_dst_ortalama)
   cv2.circle(img2,(int(x_dst_ortalama),int(y_dst_ortalama)),6,(255,0,255),2)

   
   #------------------------------------------------------------------------------------------
   """
   if abs(x_src_ortalama-x_dst_ortalama) <5 and abs(x_src_ortalama-x_dst_ortalama) >0:
       print('X sabit')
   elif (x_src_ortalama-x_dst_ortalama) > 0:
       print('SAG')
   elif (x_src_ortalama-x_dst_ortalama) < 0:
       print('SOL')
   
   
   #print('Y FARK : ',abs(y_src_ortalama-y_dst_ortalama))
   if abs(y_src_ortalama-y_dst_ortalama) <4 and abs(y_src_ortalama-y_dst_ortalama) >0 :
       print('Y sabit')
   elif (y_src_ortalama-y_dst_ortalama)< 0:
       print('YUKARI')
   elif (y_src_ortalama-y_dst_ortalama)> 0:
       print('ASAGI')
  
   
   """ 
    #---------------------------------------------------------------------------------
       
   if abs(x_src_ortalama-x_dst_ortalama) <6 and abs(x_src_ortalama-x_dst_ortalama) >0 and abs(y_src_ortalama-y_dst_ortalama) <6 and abs(y_src_ortalama-y_dst_ortalama) >0:
           print('S A B I T')
    
   elif (x_src_ortalama-x_dst_ortalama) > 0 and (y_src_ortalama-y_dst_ortalama) < 0:
       print('Sola + Asagi')
       
   elif (x_src_ortalama-x_dst_ortalama) > 0 and  (y_src_ortalama-y_dst_ortalama) > 0  :
       print('Sola + Yukari')

   elif (x_src_ortalama-x_dst_ortalama) < 0 and (y_src_ortalama-y_dst_ortalama) < 0:
       print('Saga + Asagi')
       
   elif (x_src_ortalama-x_dst_ortalama) < 0 and  (y_src_ortalama-y_dst_ortalama) > 0  :
       print('Saga + Yukari')
       
   elif abs(x_src_ortalama-x_dst_ortalama) <5 and abs(x_src_ortalama-x_dst_ortalama) >0 and (y_src_ortalama-y_dst_ortalama) < 0 :
       
       print('X:0 + Y Yukari')
       
   elif abs(x_src_ortalama-x_dst_ortalama) <5 and abs(x_src_ortalama-x_dst_ortalama) >0 and (y_src_ortalama-y_dst_ortalama) > 0 :
       
       print('X:0 + Y Asagi')
       
   elif abs(y_src_ortalama-y_dst_ortalama) <4 and abs(y_src_ortalama-y_dst_ortalama) >0 and(x_src_ortalama-x_dst_ortalama) > 0:
       
       print('X Yukari +Y:0')
       
   elif abs(y_src_ortalama-y_dst_ortalama) <4 and abs(y_src_ortalama-y_dst_ortalama) >0 and(x_src_ortalama-x_dst_ortalama) < 0:
       
       print('X Asagi + Y:0')
       
   

   
   M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) 
   
   matchesMask=mask.ravel().tolist()

   h,w=img1.shape
   pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)    

   dst = cv2.perspectiveTransform(pts,M)
   
  
   img2=cv2.polylines(img2,[np.int32(dst)],True,255,3,cv2.LINE_AA)
 
  
  draw_params=dict(matchColor=(0,255,0),singlePointColor=None,
                 matchesMask=matchesMask,
                 flags=2)
  
  
  if(len(matchesMask)==len(matches)):
      
   img3=cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
   img3=imutils.resize(img3,1500,1500)
   cv2.circle(img1, center=(418,64), radius=25, color=(255,255,255))
   cv2.imshow('Sonuc ( Birinci Frame ve Ikinci Frame',img3)
  else:
      cv2.imshow('Sonuc ( Birinci Frame ve Ikinci Frame',img2)

  
  
  if cv2.waitKey(1) &  0xFF == ord('q'):
     break
 else:
     
     print('Feature eslesmesi YOK veya ZAYIF') 

video.release()
cv2.destroyAllWindows()




