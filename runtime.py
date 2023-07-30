import os
import time
import cv2
import argparse
import numpy as np
import pandas as pd
from functions import *
from tkinter import *
from keras import Model
from tensorflow.keras.models import load_model, model_from_json
import cv2.cv2 as cv2

PATH = "./images"
pred = []
pred_label = []

class MyWindow:
    def __init__(self, win):
        self.btn=Button(win, text="Submit", fg='blue', command=self.submit)
        #self.btn.bind('<Button-1>', self.MyButtonClicked)
        self.btn.place(x=120, y=100)
        self.txtfld=Entry(win, text="Enter the path to folder containing images", bd=3, width=28)
        self.txtfld.place(x=50, y=50)
        self.status=Entry()
    def submit(self):
        self.status.insert(END, str('Running Scripts... Please Wait'))
        self.status.place(x=80,y=150)
        PATH = self.txtfld.get()
        
            
    
def main():

	# Parse Arguments
	parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
	parser.add_argument("--input","-i", help="path to input folder", required=True)
	args = parser.parse_args()
	PATH = args.input

    # User window
	"""window=Tk() 
	mywin=MyWindow(window) 
	window.geometry("400x300+10+10") 
	window.title('Welcome to Lung X-Ray Clasiifier') 
	window.mainloop()"""
    
    # Load Images from Path
	images_array,fileNames = load_images(PATH)
	print(images_array)
    
    # Resize Images
	resized_images = resize_images_of(images_array)
	print(resized_images)
    
    # Segment Images
	segmentor = unet(input_size=(320, 320, 1))
	segmentor.summary()
	segmentor.load_weights('./weights/cxr_reg_weights.hdf5') #content/drive/My Drive/Covid19AI/Pretrained_Model
	segmented_images = do_segmentation(images=resized_images, segmentor=segmentor)
	
	# load json and create model
	json_file = open('./weights/model_arch.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	predictor = model_from_json(loaded_model_json)
	
	# load weights into new model
	predictor.load_weights('./weights/eps=018_valLoss=0.4167.hdf5')
	print("Loaded model from disk")
	
	labels = ['Normal', 'Pnuemonia', 'Others']
	# Run
	for test_img in segmented_images:
		test_img = np.concatenate((test_img, test_img, test_img), axis=-1)
		pred.append(predictor.predict(np.expand_dims(test_img, axis=0)))
		pred_label.append(labels[np.argmax(pred)])
	print(pred)
	
    # Store in CSV file
	output = pd.DataFrame(data={'File Name': fileNames, 'Prediction': pred_label})
	output.to_csv('output.csv',index=False)

if __name__ == "__main__":
    main()
