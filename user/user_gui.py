from configparser import Interpolation
from tkinter import *
import tkinter.messagebox
from PIL import ImageTk, Image
import os
try:
    import Tkinter as tkinter
    import ttk
except ImportError:
    import tkinter
    from tkinter import ttk
import sys
import cv2
import numpy as np
from scipy import stats
import pickle

if __name__ == '__main__':

    def resource_path(relative_path):
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
                base_path = sys._MEIPASS
        except Exception:
                base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)
    
    def from_rgb(rgb):
        """translates an rgb tuple of int to a tkinter friendly color code
        """
        r, g, b = rgb
        return f'#{r:02x}{g:02x}{b:02x}'
    
    ## BackEnd:

    def read_image():
        url = image_info.get()
        name = image_name.get()

        full_name = url + '/' + name

        img = cv2.imread(full_name)

        return img

    def transform_image(img):
        input_image = cv2.resize(img, (48,48), interpolation=cv2.INTER_NEAREST).mean(axis=2).astype(np.ubyte)
        input_x = np.reshape(input_image, (1, 48*48))
        return input_image, input_x

    def show_image():
        img = read_image()
        img_name = 'Imagen Orginal a Procesar'
        img_input = 'Imagen a Procesar'

        cv2.imshow(img_name, img)

        input_image, input_x = transform_image(img)
        cv2.imshow(img_input, input_image)

        print(input_x, input_x.shape)

    def show_result(age, ethnicity, gender):
        #Age:
        label_age = Label(text = 'Age', bg='gray95', fg='Black', width='12', height= '1', font=('Arial', 11, 'bold'))
        label_age.place(x=130 , y=360)
        
        age = round(age, 0)
        result_age = Label(text = age, bg='gray95', fg='gray60', width='15', height= '1', font=('Arial', 10, 'bold'))
        result_age.place(x=125 , y=380)

        #Ethnicity:
        label_eth = Label(text = 'Ethnicity', bg='gray95', fg='Black', width='12', height= '1', font=('Arial', 11, 'bold'))
        label_eth.place(x=330 , y=360)
        
        result_eth = Label(text = ethnicity, bg='gray95', fg='gray60', width='15', height= '1', font=('Arial', 10, 'bold'))
        result_eth.place(x=325 , y=380)

        #Gender:
        label_gen = Label(text = 'Gender', bg='gray95', fg='Black', width='12', height= '1', font=('Arial', 11, 'bold'))
        label_gen.place(x=530 , y=360)
  
        result_gen = Label(text = gender, bg='gray95', fg='gray60', width='15', height= '1', font=('Arial', 10, 'bold'))
        result_gen.place(x=525 , y=380)

    def standardize(array, mean, std):
        return (array - mean) / std

    def get_prediction():
        img = read_image()
        input_image, input_x = transform_image(img)

        meanV = np.load(open( '../train/tokenizer_mean.npy', 'rb' ))
        S = np.load(open( '../train/tokenizer_std.npy', 'rb' ))

        input_x = standardize(input_x, meanV, S)

        #Age
        age_model_1 = pickle.load(open('../train/lin_reg_age.sav', 'rb' ))
        age_model_2 = pickle.load(open( '../train/rf_age.sav', 'rb' ))
        age_model_3 = pickle.load(open( '../train/nn_age.sav', 'rb' ))
        ponderator = np.load(open( '../train/age_pred_weights.npy', 'rb' ))

        pred_age_1 = age_model_1.predict(input_x)[0]
        pred_age_2 = age_model_2.predict(input_x)[0]
        pred_age_3 = age_model_3.predict(input_x)[0]

        age_pred = (pred_age_1*ponderator[0] + pred_age_2*ponderator[1] + pred_age_3*ponderator[2]) / ponderator.sum()

        #Ethnicity:
        
        #Genders:
        gen_model_1 = pickle.load(open('../train/log_reg_gender.sav', 'rb' ))
        gen_model_2 = pickle.load(open( '../train/svm_gender.sav', 'rb' ))
        gen_model_3 = pickle.load(open( '../train/nn_gender.sav', 'rb' ))

        pred_gen_1 = int(gen_model_1.predict(input_x)[0])
        pred_gen_2 = int(gen_model_2.predict(input_x)[0])
        pred_gen_3 = int(gen_model_3.predict(input_x)[0])
        temp = np.array([pred_gen_1,pred_gen_2,pred_gen_3])

        gen_pred = stats.mode(temp)[0][0]

        ethnicity = ['white','black','asian','indian','hispanic']
        genders = ['male','female']

        show_result(age_pred, ethnicity[2], genders[gen_pred])

        print('Done')

        return 200
 
    ## GUI:
    iconiq = resource_path("logo_squared.ico")
    png = resource_path("logo_largo.png")

    formulario = Tk()
    formulario.configure(background='gray95')
    formulario.iconbitmap(iconiq)
    formulario.geometry('770x550')
    formulario.title('Machine Learning - Final Project')

    heading=Label(text = 'Machine Learning - Final Project - PUJ', bg=from_rgb(((23, 50, 104))), fg=from_rgb(((255, 255, 255))), width='50', height= '5', font=('Arial', 12, 'bold'))
    heading.pack(fill="x")

    logo=PhotoImage(file = png)
    logo=logo.subsample(2, 2)
    logo1=Label(image=logo, bg=from_rgb(((23, 50, 104))))
    logo1.image = logo

    logo1.place(x=15 , y=15)

    firma=Label(text = '© 2022 PUJ \n Creado por Juan Vivas - Diego Ruiz - Camila Castiblanco ', fg='gray60', font = "Courier 6 bold italic", bg='gray95')
    firma.place(x=250, y=505)

    ## Labels para entrada:
    label1 = Label(text = 'Path to Image', bg='gray95', fg='Black', width='18', height= '1', font=('Arial', 11, 'bold'))
    label1.place(x=25 , y=130)

    image_info=StringVar()
    image_entry= Entry(textvariable = image_info, width='50', font=('Arial', 9),  fg='gray60')
    image_entry.place(x=50 , y=160)

    label1 = Label(text = 'Image Name', bg='gray95', fg='Black', width='18', height= '1', font=('Arial', 11, 'bold'))
    label1.place(x=25 , y=210)

    image_name=StringVar()
    image_entry= Entry(textvariable = image_name, width='50', font=('Arial', 9),  fg='gray60')
    image_entry.place(x=50 , y=240)

    ## Button to Execute
    script_button= Button(formulario, text='Run', width= '25', height='2' , command = get_prediction, fg='white', bg=from_rgb(((23, 50, 104))), font=('Arial', 10, 'bold'), activeforeground=from_rgb(((23, 50, 104))))
    script_button.place(x=500 , y=230)

    ## Button to Show Image
    image_button= Button(formulario, text='Show Image', width= '25', height='2' , command = show_image, fg='white', bg=from_rgb(((23, 50, 104))), font=('Arial', 10, 'bold'), activeforeground=from_rgb(((23, 50, 104))))
    image_button.place(x=500 , y=150)

    def refresh():
        formulario.update()
        formulario.after(10,refresh)
    
    formulario.mainloop()
