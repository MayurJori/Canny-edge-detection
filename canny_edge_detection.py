#Mayur Jori
# mbj282@nyu.edu
# N11534214
# Computer Vision : Project 1 – Canny Edge Detection

#!/usr/bin/env python
# coding: utf-8

import sys #for opening image as commandline argument
import numpy as np  #Numpy for matrix multiplication
import cv2 #cv2 for writing images
from skimage.io import imread #skimage.io for reading images
import math #math for taking square root
import csv #csv for storing matrix values to csv file
np.set_printoptions(threshold=np.nan) #for printing full matrix in output instead for only snippet of the matrix

#Smoothing Operation
class gaussian:
    def __init__(self,img):
        self.img=img #original input image
        self.gaussian_mask=np.array([[1,1,2,2,2,1,1],[1,2,2,4,2,2,1],[2,2,4,8,4,2,2],[2,4,8,16,8,4,2],[2,2,4,8,4,2,2],[1,2,2,4,2,2,1],[1,1,2,2,2,1,1]],dtype=float) #given mask
        self.gaussian_sum=np.sum(self.gaussian_mask) #gaussian sum for normalization
        self.height= len(img) #height for cropping image later
        self.width=len(img[0]) #width for cropping image later
        self.smoothing_img=np.zeros((self.height,self.width)) #matrix for storing smoothing output 
        self.start=int(len(self.gaussian_mask)/2) #start will neglect the undefined region from start
        self.height_end=self.height-self.start #neglect region from undefined height
        self.width_end=self.width-self.start #neglect region from undefined width


    def calc_smoothing(self,xval,yval):
        #convolution operation for smoothing operation
        #result is normalized by dividing the result with gaussian mask matrix sum
        return np.sum(np.multiply(self.img[xval-3:xval+4,yval-3:yval+4],self.gaussian_mask))/self.gaussian_sum


    def gaussian_smoothing(self):
        for i in range(0,self.height): #consider each row from the input image
            for j in range(0,self.width): # consider each column from the input image
                #if i or j lies in undefined region,  then do not perform convolution
                if i<self.start or i>=self.height_end or j<self.start or j>=self.width_end: 
                    continue
                else:
                #perform convolution for the defined pixels, store values in new array
                    self.smoothing_img[i][j]=self.calc_smoothing(i,j)
        return self.saveresult() #store the final smoothing image to local storage and export csv file for pixel values
    def saveresult(self): 
        #crop the image for which the smoothing operation is undefined
        self.smoothing_img=self.smoothing_img[self.start:self.height-self.start, self.start:self.width-self.start]
        out_img=cv2.imwrite('1_gradient_smoothing.bmp',self.smoothing_img)
        np.savetxt("1_grad_smoothing.csv",self.smoothing_img,delimiter=",")
        return self.smoothing_img
    


#Gradient Operation
class gradient:
    def __init__(self,smoothing_img):
        self.smoothing_img=smoothing_img #smoothed image is copied for gradient operation
        self.prewitt_mat_x=np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=float) 
        self.prewitt_mat_y=np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=float)

        self.height=len(self.smoothing_img) #height for cropping image later
        self.width=len(self.smoothing_img[0]) #width for cropping image later
        self.start=int(len(self.prewitt_mat_x)/2) #for neglecting undefined region from start
        self.height_end=self.height-self.start  #for neglecting undefined region from undefined height
        self.width_end=self.width-self.start  #for neglecting undefined region from undefined width

        self.smoothing_img_height=len(self.smoothing_img) # height for looping through gradient operation 
        self.smoothing_img_width=len(self.smoothing_img[0]) ## width for looping through gradient operation

        #zero matrices initialized for storing the result of horizontal gradient, vertical gradient, gradient magnitude, gradient angle and sector values
        self.gradx=np.zeros((self.smoothing_img_height,self.smoothing_img_width),dtype=float)
        self.grady=np.zeros((self.smoothing_img_height,self.smoothing_img_width),dtype=float)
        self.grad_magnitude=np.zeros((self.smoothing_img_height,self.smoothing_img_width),dtype=float)
        self.grad_angle=np.zeros((self.smoothing_img_height,self.smoothing_img_width),dtype=float)
        self.sector=np.zeros((self.smoothing_img_height,self.smoothing_img_width),dtype=float)

    
    def calcgrad(self,xval,yval): #horizontal and vertical gradients are calculated for each defined pixel value
        self.tempgradx=0
        self.tempgrady=0
        #performing gradient operation for each pixel value and taking the normalized value after getting absolute values
        self.tempgradx= abs(np.sum(np.multiply(self.smoothing_img[xval-1:xval+2,yval-1:yval+2],self.prewitt_mat_x)))/3
        self.tempgrady= abs(np.sum(np.multiply(self.smoothing_img[xval-1:xval+2,yval-1:yval+2],self.prewitt_mat_y)))/3
        return self.tempgradx,self.tempgrady
    
    def calcsector(self,val): #sector calculation based on the gradient angle value
        val=val%360 #handling the negative values and converting them to positive for easy comparison
        if 0<=val<22.5 or 157.5 <= val < 202.5 or 337.5<=val<=360: #sector 0 corresponds following gradient angles
            return 0
        elif 22.5<= val< 67.5 or 202.5<=val<247.5: #sector 1 corresponds following gradient angles
            return 1
        elif 67.5<=val < 112.5 or 247.5<=val< 292.5: #sector 2 corresponds following gradient angles
            return 2
        elif 112.5<=val<157.5 or 292.5<=val<337.5: #sector 3 corresponds following gradient angles
            return 3
        
    def gradient_operation(self):      
        for i in range(0,self.smoothing_img_height): #consider each row from the input image
            for j in range(0,self.smoothing_img_width): #consider each column from the input image
                #if i or j lies in undefined region,  then do not perform gradient operation
                if i<self.start or i>=self.height_end or j<self.start or j>=self.width_end: 
                    continue
                else:
                    #perform gradient operation on defined pixel values and calculate horizontal gradient, 
                    #vertical gradient, gradient magnitude and sector value
                    gradx_val,grady_val= self.calcgrad(i,j)
                    self.gradx[i][j]=(gradx_val) #assign vertical gradient value to respective value in output image
                    self.grady[i][j]=(grady_val) #assign horizontal gradient value to respective value in output image
                    grad_magnitude_val=(self.gradx[i][j]**2)+(self.grady[i][j]**2) #calculate gradient magnitude 
                    self.grad_magnitude[i][j]=np.sqrt(grad_magnitude_val)/math.sqrt(2) #normalize the gradient magnitude
                    if self.grady[i][j]==0: #if the value of horizontal gradient is 0, output of inverse tan will be 0 degrees
                        self.grad_angle[i][j]=0
                    elif self.gradx[i][j]==0: #if the value of vertical gradient is 0, value of inverse tan will be 90 degrees
                        self.grad_angle[i][j]=90
                    else: #else performing the inverse tan opeation and finding the gradient angle in degrees
                        self.grad_angle[i][j]=math.atan2(self.grady[i][j],self.gradx[i][j])*(180/np.pi)
                        self.sector[i][j]=self.calcsector(self.grad_angle[i][j]) #sector calculation based on the gradient angle value
        return self.saveresult()

    def saveresult(self):
        #crop the output images where the gradient operation is undefined
        self.gradx=self.gradx[self.start:self.height_end,self.start:self.width_end]
        self.grady=self.grady[self.start:self.height_end,self.start:self.width_end]
        self.grad_magnitude=self.grad_magnitude[self.start:self.height_end,self.start:self.width_end]
        self.grad_angle=self.grad_angle[self.start:self.height_end,self.start:self.width_end]
        self.sector=self.sector[self.start:self.height_end,self.start:self.width_end]

        #save output images to local storage
        grad_image=cv2.imwrite('2_gradient_magnitude_x.bmp',self.gradx)
        grad_image=cv2.imwrite('3_grad_magnitude_y.bmp',self.grady)
        grad_image=cv2.imwrite('4_gradient_magnitude.bmp',self.grad_magnitude)

        #export the pixel values to csv files
        np.savetxt("2_grad_magnitude_x.csv",self.gradx,delimiter=",")
        np.savetxt("3_grad_magnitude_y.csv",self.grady,delimiter=",")
        np.savetxt("4_grad_magnitude.csv",self.grad_magnitude,delimiter=",")
        np.savetxt("5_grad_angle.csv",self.grad_angle,delimiter=",")
        np.savetxt("6_sector.csv",self.sector,delimiter=",")
        return self.grad_magnitude, self.sector #returning gradient magnitude and sector values for future use


#Non-maxima suppression
class nms:
    def __init__(self,gradient_magnitude,sector):
        self.grad_magnitude=gradient_magnitude #copying previously returned gradient matrix
        self.sector=sector ##copying previously returned gradient matrix
        self.sector_x_len=len(self.grad_magnitude) #height for looping through non-maxima suppression operation
        self.sector_y_len=len(self.grad_magnitude[0]) #width for looping through non-maxima suppression operation
        self.grad_magnitude_nms=np.zeros((self.sector_x_len,self.sector_y_len),dtype=float) #initialized matrix for storing the output
        

        self.start=1 #neglecting the undefined start values 
        self.height_end=self.sector_x_len-self.start #neglect region from undefined height 
        self.width_end=self.sector_y_len-self.start#neglect region from undefined width

        
    def nonmaximasuppression(self,ival,jval):
        val=self.grad_magnitude[ival][jval] #gradient magnitude value for comparison
        sectorval=self.sector[ival][jval] #sector value for corresponing pixel location
        if sectorval==0: #comparing the correponding values if sector 0 is identified
            #if value is greater than both neighbors, then only assign it to the new matrix pixel location 
            if val >= self.grad_magnitude[ival][jval-1] and val >= self.grad_magnitude[ival][jval+1]:
                self.grad_magnitude_nms[ival][jval]=math.floor(val)    
        elif sectorval==1: #comparing the correponding values if sector 1 is identified 
            #if value is greater than both neighbors, then only assign it to the new matrix pixel location 
            if val >= self.grad_magnitude[ival+1][jval-1] and val >= self.grad_magnitude[ival-1][jval+1]:
                self.grad_magnitude_nms[ival][jval]=math.floor(val)    
        elif sectorval==2: #comparing the correponding values if sector 2 is identified 
            #if value is greater than both neighbors, then only assign it to the new matrix pixel location 
            if val >= self.grad_magnitude[ival-1][jval] and val >= self.grad_magnitude[ival+1][jval]:
                self.grad_magnitude_nms[ival][jval]=math.floor(val)    
        elif sectorval==3: #comparing the correponding values if sector 3 is identified 
            #if value is greater than both neighbors, then only assign it to the new matrix pixel location 
            if val >= self.grad_magnitude[ival-1][jval-1] and val >= self.grad_magnitude[ival+1][jval+1]:
                self.grad_magnitude_nms[ival][jval]=math.floor(val)
    def nonmaximasupp(self):            
        for i in range(0,self.sector_x_len): #consider each row from the input image
            for j in range(0,self.sector_y_len): #consider each column from the input image
                #if i or j lies in undefined region,  then do not perform non-maxima suppression
                if i<self.start or i>=self.height_end or j<self.start or j>=self.width_end:
                    continue
                else:
                    #perform non-maxima suppression for defined pixels
                    self.nonmaximasuppression(i,j)
        return self.saveresult()
    def saveresult(self): 
        #crop the image where undefined values are present and save output image to local storage, along with exporting the csv for matrix values
        self.grad_magnitude_nms=self.grad_magnitude_nms[self.start:self.height_end,self.start:self.width_end]
        self.grad_image=cv2.imwrite('7_nms_grad_magnitude.bmp',self.grad_magnitude_nms)
        np.savetxt("7_nms_grad_magnitude.csv",self.grad_magnitude_nms,delimiter=",")
        return self.grad_magnitude_nms #return gradient magnitude image after performing non-maxima suppression for future use



#Thresholding 
class thresholding:
    def __init__(self,gradient_magnitude_nms):
        self.nms_dict={} #dictionary for storing the count of pixels at pixel values ranging from (0,255)
        self.grad_magnitude_nms=gradient_magnitude_nms #copying previously returned gradient magnitude matrix
        self.nms_height=len(self.grad_magnitude_nms) #height for looping use
        self.nms_width=len(self.grad_magnitude_nms[0]) #width for looping use
        self.total_no_of_pixels=0 #total number of pixels in an image
        self.minus_zero_pixels=0 #pixel count having zero value
        self.no_of_pixels=0 #pixel count after subtracting the zero value pixel count
        self.p10=0 #threshold value for 10% pixels
        self.p30=0 #threshold value for 30% pixels
        self.p50=0 #threshold value for 50% pixels
        self.p10image=np.zeros((self.nms_height,self.nms_width))
        self.p30image=np.zeros((self.nms_height,self.nms_width))
        self.p50image=np.zeros((self.nms_height,self.nms_width))
        
        #the foreground (edges) are bright, so starting from 255 for threshold calculation 
        self.p10index=255
        self.p30index=255
        self.p50index=255
        
    def initializedict(self): #dictionary initialization to zero values for all key values from (0,255)
        for i in range(0,256):
            self.nms_dict[i]=0
    def removezeropixels(self):
        self.minus_zero_pixels=(self.nms_dict[0]) #pixel count having zero values
        self.no_of_pixels=self.total_no_of_pixels-self.minus_zero_pixels #subtracting the zero pixel value count
        self.p10=int(self.no_of_pixels/10) #10% pixels calculation
        self.p30=int(self.no_of_pixels*3/10) #30% pixels calculation
        self.p50=int(self.no_of_pixels/2) #50% pixels calculation
        
    def calculatedictvalues(self):
        for i in range(0, self.nms_height): #looping through each row
            for j in range(0,self.nms_width): #looping through each column
                #add 1 to corresponding key in dictionary when value for that key is found
                self.nms_dict[self.grad_magnitude_nms[i][j]]=self.nms_dict[self.grad_magnitude_nms[i][j]]+1
    
        for i in self.nms_dict.values():
            self.total_no_of_pixels+=i #total number of pixels for removing the zero value pixels 
            
    def calculatefinalvalues(self):
        #initialized the sum to 0 for each threshold value
        sump10=0
        sump30=0
        sump50=0
        
        while sump10<=self.p10 or sump30<=self.p30 or sump50<=self.p50: #calculation of threshold for each value
            if sump10<=self.p10: #comparing if the current sum is less than calculated 10% threshold
                sump10+=self.nms_dict[self.p10index] #if yes, add it to the pixels count
                self.p10index-=1 
            if sump30<=self.p30: #comparing if the current sum is less than calculated 30% threshold
                sump30+=self.nms_dict[self.p30index] #if yes, add it to the pixels count
                self.p30index-=1
            if sump50<=self.p50: #comparing if the current sum is less than calculated 50% threshold
                sump50+=self.nms_dict[self.p50index] #if yes, add it to the pixels count
                self.p50index-=1
        print ("Threshold value for 10% : ", self.p10index)
        print ("Number of edges detected : ", sump10)
        print ("\n")
        print ("Threshold value for 50% : ", self.p30index)
        print ("Number of edges detected : ", sump30)
        print ("\n")
        print ("Threshold value for 50% : ", self.p50index)
        print ("Number of edges detected : ", sump50)
        
    def build_threshold_images(self):
        for i in range(0,self.nms_height):
            for j in range(0,self.nms_width):
                if self.grad_magnitude_nms[i][j]<=self.p10index:
                        self.p10image[i][j]=0
                else: 
                    self.p10image[i][j]=255
                    
                if self.grad_magnitude_nms[i][j]<=self.p30index:
                        self.p30image[i][j]=0
                else: 
                    self.p30image[i][j]=255
                if self.grad_magnitude_nms[i][j]<=self.p50index:
                        self.p50image[i][j]=0
                else: 
                    self.p50image[i][j]=255
        self.grad_image=cv2.imwrite('8_gradient_magnitude_nms_10percent.bmp',self.p10image)
        self.grad_image=cv2.imwrite('9_gradient_magnitude_nms_30percent.bmp',self.p30image)
        self.grad_image=cv2.imwrite('10_gradient_magnitude_nms_50percent.bmp',self.p50image)
        
img=imread(sys.argv[1],dtype=float) #open image as commanline argument

#Gaussian smoothing operation
tempobj=gaussian(img)
smoothing_img=tempobj.gaussian_smoothing() #smoothed image is returned for future use 

#Gradient operation
tempobj=gradient(smoothing_img)
gradient_magnitude,sector=tempobj.gradient_operation()

#Non-maxima suppression
tempobj=nms(gradient_magnitude,sector)
gradient_magnitude_nms=tempobj.nonmaximasupp()

#Thresholding operation
tempobj=thresholding(gradient_magnitude_nms)
tempobj.initializedict()
tempobj.calculatedictvalues()
tempobj.removezeropixels()
tempobj.calculatefinalvalues()
tempobj.build_threshold_images()