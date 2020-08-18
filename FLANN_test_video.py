#Not: xfeatures2d modülü openCV sürümünde hata veriyorsa, ilgili sürüm kaldırılıp "pip install opencv-contrib-python" şeklinde yüklenebilir.
import cv2
import numpy as np
import imutils
import time

MIN_ESLESME=25

video=cv2.VideoCapture(1)

while True:
    
 _,frame=video.read()
 orjinal=frame
 orjinal=imutils.resize(orjinal,360,360)
 
 time.sleep(0.1)     #İkinci frame'den önce 0.1 saniye beklenir
 a,frame2=video.read()
 
 img1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 img1=cv2.GaussianBlur(img1,(15,15),0)
 
 img2=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
 img2=cv2.GaussianBlur(img2,(15,15),0)
 
 cv2.imshow('Orjinal Goruntu',orjinal)

 sift=cv2.xfeatures2d.SIFT_create()
 kp1,des1=sift.detectAndCompute(img1,None)
 kp2,des2=sift.detectAndCompute(img2,None) 

 sayac_src=0
 sayac_dst=0
 toplam_dst_x=0
 toplam_src_x=0
 toplam_dst_y=0
 toplam_src_y=0
 
 
 if des2 is not None : #Yeterli çıkarımda bulunamadığı zaman hata verip
                       #programı sonlandırmaması için
  FLANN_INDEX_KDTREE=0
  index_params=dict(algorithm=FLANN_INDEX_KDTREE)  
  search_params=dict(checks=100)
 
  flann=cv2.FlannBasedMatcher(index_params,search_params)
  matches=flann.knnMatch(des1,des2,k=2)

  good=[]   #Başarılı eşleştirmeler buraya atanacak

  for m,n in matches:
    m.distance <0.7*n.distance
    good.append(m)
  
  if len(good)>MIN_ESLESME :
   src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)    
   dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
   
        
   for j in src_pts:       ## Birinci frame ortalama koordinat hesabı
       #print(j[0,0])
       toplam_src_x=toplam_src_x+j[0,0]
       toplam_src_y=toplam_src_y+j[0,1]
       sayac_src+=1  
   x_src_ortalama=toplam_src_x/sayac_src
   y_src_ortalama=toplam_src_y/sayac_src  
   cv2.circle(img1,(int(x_src_ortalama),int(y_src_ortalama)),6,(255,0,255),2) #ortalama koordinat
   
   
   
   for i in dst_pts:    ## İkinci frame ortalama koordinat hesabı
       #print(i[0,0])
       toplam_dst_x=toplam_dst_x+i[0,0]
       toplam_dst_y=toplam_dst_y+i[0,1]
       sayac_dst+=1  
   x_dst_ortalama=toplam_dst_x/sayac_dst
   y_dst_ortalama=toplam_dst_y/sayac_dst
   cv2.circle(img2,(int(x_dst_ortalama),int(y_dst_ortalama)),6,(255,0,255),2)


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
   
    # Harekete göre çıkarımda bulunma 
       
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
       
   # Eşleştirmelerin görselleştirilmesi aşaması:
       
   M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)        #Hazır alındı
   matchesMask=mask.ravel().tolist()                                     #Hazır alındı
   h,w=img1.shape                                                        
   pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)    
   dst = cv2.perspectiveTransform(pts,M)
    
   img2=cv2.polylines(img2,[np.int32(dst)],True,255,3,cv2.LINE_AA) #eşleşen kısmın etrafına 
                                                                   #dörtgen çizimi
   
  draw_params=dict(matchColor=(0,255,0),singlePointColor=None,                                                          
                 matchesMask=matchesMask,
                 flags=2)                 #eşleştirme çizgileri için parametreler
  
  
  if(len(matchesMask)==len(matches)):
      
   img3=cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
   img3=imutils.resize(img3,1500,1500)
   cv2.imshow('Sonuc ( Birinci Frame ve Ikinci Frame',img3)
   
  else: #Mask Boyutları eşleşmezse orjinal görüntü gösterilir (hatayı çözmek için eklendi)
      cv2.imshow('Sonuc ( Birinci Frame ve Ikinci Frame',img2)
 
  if cv2.waitKey(1) &  0xFF == ord('q'):
     break
 
 else:          # MIN_ESLESME Saglanamazsa
     
     print('Feature eslesmesi YOK veya ZAYIF') 

video.release()
cv2.destroyAllWindows()




