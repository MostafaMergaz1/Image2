import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

st.title('Filtter Aplication')
upload=st.file_uploader('choose an image',type=['png','jpg','jpeg'])
import cv2
def black(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

def bluring_dst(img,ksize):
    ksize=abs(int(ksize))
    if ksize%2==0:
        ksize+=1
    blur=cv2.GaussianBlur(img,(ksize,ksize),0,0)
    dst_gray,dst_color=cv2.pencilSketch(blur)
    return dst_gray    

def bluring_color(img,ksize):
    ksize=abs(int(ksize))
    if ksize%2==0:
        ksize+=1
    blur=cv2.GaussianBlur(img,(ksize,ksize),0,0)
    dst_gray,dst_color=cv2.pencilSketch(blur)
    return dst_color    

def bluring(img,ksize):
    ksize=abs(int(ksize))
    if ksize%2==0:
        ksize+=1
    blur=cv2.GaussianBlur(img,(ksize,ksize),0,0)
    return blur

def vintag(img,level):
    if level==0:
        level=1
    hight,width=img.shape[:2]
    x_kernel=cv2.getGaussianKernel(width,width/level)
    y_kernel=cv2.getGaussianKernel(hight,hight/level)

    kernel=y_kernel*x_kernel.T
    mask=kernel/kernel.max()
    image_copy=np.copy(img)
    for i in range(3):
        
        image_copy[:,:,i]=image_copy[:,:,i]*mask
    return image_copy

def HDR(img,level,sigma_s,sigma_r):
    light=cv2.convertScaleAbs(img,beta=level)
    details=cv2.detailEnhance(light,sigma_s=sigma_s,sigma_r=sigma_r)
    return details

def style_img(img,sigma_s,sigma_r,ksize):
    ksize=abs(int(ksize))
    if ksize%2==0:
        ksize+=1
    blur=cv2.GaussianBlur(img,(ksize,ksize),0,0)
    style=cv2.stylization(blur,sigma_s=sigma_s,sigma_r=sigma_r)
    return style

def naration_image(img):
    return 255-img


def sepia_filter(img):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(img, sepia_filter)
    sepia_img = np.clip(sepia_img, 0, 255)
    return sepia_img.astype(np.uint8)



def cartoon_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon




def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(img, -1, kernel)
    return sharp


def sketch_effect(img,ksize):
    ksize=abs(int(ksize))
    if ksize%2==0:
        ksize+=1
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (ksize, ksize), 0,0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return sketch



def enhance_old_image(img):
    
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
   
    contrast_bright = cv2.convertScaleAbs(denoised, alpha=1.3, beta=20)

   
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(contrast_bright, -1, kernel)

    
    enhanced = cv2.detailEnhance(sharpened, sigma_s=10, sigma_r=0.15)

    return enhanced


if upload is not None:
    img=Image.open(upload)
    img=np.array(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    original_image,output_image=st.columns(2)

    with original_image:
        st.header('Original Image')
        st.image(img,channels='BGR',use_column_width=True)
        st.header('choose filter')
        option=st.selectbox('Silict from filter',('None','black','bluring_dst','bluring_color','bluring','vintag','HDR','style_img','naration_image','sepia_filter','cartoon_effect','sharpen_image','sketch_effect','enhance_old_image','old_image'
))

        output_flag=1
        color='BGR'

        if option=='None':
            output_flag=0
            output=img
        elif option=='black':
            output=black(img)
            color='GRAY'
        elif option=='bluring_dst':
            ksize=st.slider('ksize',-50,0,50,step=1)
            output=bluring_dst(img,ksize)

        elif option=='bluring_color':
            ksize=st.slider('ksize',-50,0,50,step=1)
            output=bluring_color(img,ksize)

        elif option=='bluring':
            ksize=st.slider('ksize',-50,0,50,step=1)
            output=bluring(img,ksize)

        elif option == 'HDR':
         level = st.slider('level', -100, 50, 100, step = 1)
         sigma_s = st.slider('sigma_s', 0, 200, 40, step = 10)
         sigma_r = st.slider('sigma_r', 0, 1, 4)
         output = HDR(img, level,sigma_s,sigma_r)


        elif option == 'naration_image':
            output = naration_image(img)

    

        elif option == 'sepia_filter':
            output = sepia_filter(img)

        elif option == 'cartoon_effect':
            output = cartoon_effect(img)

        elif option == 'sharpen_image':
            output = sharpen_image(img)

        elif option == 'vintag':
            level = st.slider('Level', -10,10, 1, step = 1)
            output = vintag(img, level)

        elif option == 'style_img':
         ksize = st.slider('ksize', -100, 50, 100, step = 1)
         sigma_s = st.slider('sigma_s', 0, 200, 40, step = 10)
         sigma_r = st.slider('sigma_r', 0, 1, 4)
         output = style_img(img,sigma_s,sigma_r,ksize )


        elif option == 'sketch_effect':
            ksize=st.slider('ksize',-300,0,300,step=1)
            output = sketch_effect(img,ksize)
            color = 'GRAY'

        elif option == 'enhance_old_image':
            output = enhance_old_image(img)


        elif option == 'old_image':
            output = old_image(img)


        with output_image:
            st.header('Output')
        #imshow image in stremlet
            if color=='BGR':
                output=cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
                st.image(output,use_column_width=True)
            elif color == 'GRAY':
                st.image(output, channels='GRAY', use_column_width=True)
        
            
