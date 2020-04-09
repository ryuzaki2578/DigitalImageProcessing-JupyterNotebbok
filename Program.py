#!/usr/bin/env python
# coding: utf-8

# In[54]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mimg


# In[43]:


def pixelVal(pix, r1, s1, r2, s2): 
    if (0 <= pix and pix <= r1): 
        return (s1 / r1)*pix 
    elif (r1 < pix and pix <= r2): 
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1 
    else: 
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2 
img = cv2.imread('test.jpg',0) 
r1 = 70
s1 = 0
r2=140
s2 = 255
pixelVal_vec = np.vectorize(pixelVal)
contrast_stretched = pixelVal_vec(img, r1, s1, r2, s2)
equ=np.hstack((img,contrast_stretched))
plt.title("Original/Contrast Stretched")
plt.imshow(equ,'gray')
plt.show()


# In[44]:


histr = cv2.calcHist([img],[0],None,[256],[0,256]) 
  
# show the plotting graph of an image 
plt.plot(histr) 
plt.show() 


# In[45]:


stretch_near = cv2.resize(img, (780, 540),  
               interpolation = cv2.INTER_NEAREST) 
plt.title('original')
plt.imshow(img,'gray')
plt.show()
plt.imshow(stretch_near,'gray')
plt.show()


# In[46]:


image=img
equ = cv2.equalizeHist(image) 
res = np.hstack((image, equ)) 
plt.title("original/enhaced stacked side by side")
plt.imshow(res,'gray')
plt.show()


# In[47]:


image=cv2.imread("test.jpg",0)
x,y=image.shape
th=np.sum(image)/(x*y)
binary=np.zeros((x,y),np.double)
binary=(image>=th)*255
binary=binary.astype(np.uint8)
plt.title("original/binary")
equ=np.hstack((image,binary))
plt.imshow(equ,'gray')
plt.show()


# In[48]:


image=cv2.imread('test.jpg',0)
x,y=image.shape
z=np.zeros((x,y))
for i in range(0,x):
    for j in range(0,y):
        if(image[i][j]>50 and image[i][j]<150):
            z[i][j]=255
        else:
            z[i][j]=image[i][j]
equ=np.hstack((image,z))
plt.title('Original\Graylevel slicing with background')
plt.imshow(equ,'gray')
plt.show()


# In[49]:


image=cv2.imread('test.jpg',0)
x,y=image.shape
z=np.zeros((x,y))
for i in range(0,x):
    for j in range(0,y):
        if(image[i][j]>50 and image[i][j]<150):
            z[i][j]=255
        else:
            z[i][j]=0
equ=np.hstack((image,z))
plt.title('Original\Graylevel slicing w/o background')
plt.imshow(equ,'gray')
plt.show()


# In[50]:


image=cv2.imread('test.jpg',0)
x,y=image.shape
z=255-image
equ=np.hstack((image,z))
plt.title('Original\Image Negative')
plt.imshow(equ,'gray')
plt.show()


# In[51]:


image=cv2.imread('test.jpg',0)
x,y=image.shape
c=255/(np.log(1+np.max(image)))
z=c*np.log(1+image)
z=np.array(z,dtype=np.uint8)
equ=np.hstack((image,z))
plt.title('Log Transformation')
plt.imshow(equ,'gray')
plt.show()


# In[52]:


img=cv2.imread('test.jpg',0)
x,y=img.shape
z=cv2.blur(img,(3,3))
z1=cv2.blur(img,(5,5))
equ=np.hstack((img,z))        
plt.title('original/Averaging filter3X3')
plt.imshow(equ,'gray')
plt.show()
equ=np.hstack((img,z1))
plt.title('orogina;/averging filter 5X5')
plt.imshow(equ,'gray')
plt.show()


# In[53]:


z=cv2.medianBlur(img,5)
equ=np.hstack((img,z))
plt.title('original/Median Blur')
plt.imshow(equ,'gray')
plt.show()

