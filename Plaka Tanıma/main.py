import cv2
import numpy as np
import pytesseract
import imutils

img=cv2.imread("4.jpg")
gri=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        #görüntünün nasıl okunması gerektiğini belirtir
filtre=cv2.bilateralFilter(gri,7,200,200)       #görüntüyü karartıyor   
kose=cv2.Canny(filtre,40,200)                   #görüntüye filtre ugulayıp blurluyor


kontur,a=cv2.findContours(kose,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   #köşeleri buluyor
cnt=imutils.grab_contours((kontur,a))                                   #imutils ile dikdörtgenin içindeki alanı buluyor
cnt=sorted(cnt,key=cv2.contourArea,reverse=True)[:10]                   #sıralama

ekran=0
for i in cnt:                                   #kontüre yaklaştırıyor
    eps=0.018*cv2.arcLength(i,True)
    aprx=cv2.approxPolyDP(i,eps,True)
    if len(aprx)==4:                            #kenar yakalarsa
        ekran=aprx
        break

maske=np.zeros(gri.shape,np.uint8)                          #plakayı bir tuval gibi alıyor
yenimaske=cv2.drawContours(maske,[ekran],0,(255,255,255),-1)

yazi=cv2.bitwise_and(img,img,mask=maske)

(x,y)=np.where(maske==255)
(ustx,usty)=(np.min(x),np.min(y))
(altx,alty)=(np.max(x),np.max(y))
kirp=gri[ustx:altx+1,usty:alty+1]

text=pytesseract.image_to_string(kirp,lang="eng")
print(text)

cv2.imshow("2",gri)
# cv2.imshow("3",filtre)
cv2.imshow("4",kose)
# cv2.imshow("5",yenimaske)
cv2.imshow("6",yazi)
cv2.imshow("7",kirp)

cv2.waitKey(0)
cv2.destroyAllWindows(img)
