from textwrap import fill
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
import cv2
import mediapipe as mp
import multiprocessing
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from keras_facenet import FaceNet
import datetime

# detection and recognition init
cap = cv2.VideoCapture(0)
pTime = 0
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.45,1)
embedder = FaceNet()
tempSentRecord = {}

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

# recognition dictionary
x_list = list()
photo_list = list()
y_list = list()
name_dict = dict()
path = 'D:/Users/Admin/Desktop/clientFaces/clients3'
#path = "/Users/p2digital/Documents/theface"
i = 0
for folders in os.listdir(path):
    name_dict[i] = folders
    truePath = os.path.join(path,folders)
    for image in os.listdir(truePath):
        img = cv2.imread(os.path.join(truePath,image))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        print(img)
        extracts = embedder.extract(img, threshold=0.95)
        imgEncode = extracts[0]['embedding']
        img = image_resize(img, height = 140)
        x_list.append(imgEncode)
        photo_list.append(img)
        y_list.append(i)
    i += 1

'''# for fullscreen
def get_display_size():
    init_root = Tk()
    init_root.update_idletasks()
    init_root.attributes('-fullscreen', True)
    init_root.state('iconic')
    height = init_root.winfo_screenheight()
    width = init_root.winfo_screenwidth()
    init_root.destroy()
    return height, width

HEIGHT, WIDTH = get_display_size()'''

# mainpage
root = Tk()
root.geometry(f"{800}x{600}")

canvas = Canvas(root)
canvas.pack(fill='both')

# video
wrapper_video = LabelFrame(root)
wrapper_video.place(relx=0.03,rely=0.05,relheight=0.9,relwidth=0.66)

label_video = Label(wrapper_video)
label_video.place(in_=wrapper_video,anchor='c',relx=0.5,rely=0.5)

# side bar
wrapper = LabelFrame(root)
wrapper.place(relx=0.72,rely=0.05,relheight=0.9,relwidth=0.25)

f_canvas = Canvas(wrapper)
#f_canvas.place(relx=0,rely=0,relwidth=1,relheight=1)

# relx calculation
#relx = 0.5-(photo_width/(2*wrapper.winfo_width()))
f_canvas.place(relx=0.26,rely=0,relwidth=1,relheight=1)

yscrollbar = ttk.Scrollbar(wrapper,orient='vertical',command=f_canvas.yview)
yscrollbar.pack(side="right",fill="y")

f_canvas.configure(yscrollcommand=yscrollbar.set)

f_canvas.bind('<Configure>',lambda e: f_canvas.configure(scrollregion = f_canvas.bbox('all')))

frame = Frame(f_canvas)
f_canvas.create_window((0,0),window=frame,anchor='nw')

'''
# img frame
for i in range(40):
    # img
    frame1 = LabelFrame(frame)
    image = Image.open("D:/Users/Admin/Desktop/Ahmad Izhan/Ahmad Izhan_12360_a.jpeg")
    photo = ImageTk.PhotoImage(image.resize((100, 100), Image.ANTIALIAS))
    label = Label(frame1,image=photo)
    label.image = photo
    label.pack(side='left')
    Label(frame1,text='Hello').pack(side='left')
    Button(frame1,text="Lol").pack(side='left')
    frame1.pack(padx=20,pady=20)
'''
# functions
def recognize_face(encodeFace,x_list,y_list,name_dict,threshold):
    dist_list = list()
    for face in x_list:
        dist_list.append(np.linalg.norm(encodeFace-face))
    index1 = np.argmin(dist_list)
    #print(dist_list)
    #print(index1)
    if dist_list[index1] < threshold:
        index2 = y_list[index1]
        return name_dict[index2],index1
    else:
        pass

def face_detection(record,event):
    while True:
        img = cap.read()[1]
        img = image_resize(img, height = int(root.winfo_screenheight()/1.3))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = faceDetection.process(img)
        bbox_data = list()
        if results.detections:
            for ids, detection in enumerate(results.detections):
                #mpDraw.draw_detection(img, detection)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                        int(bboxC.width * iw), int(bboxC.height * ih)
                
                bbox_data.append(img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]])
                cv2.rectangle(img,bbox,(255,0,255),2)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        label_video['image'] = img
        record["bbox_data"] = bbox_data    
        event.set()
        if record["name_with_photo"] != None:
            frame1 = LabelFrame(frame)
            name, photo = record["name_with_photo"]

            # photo
            photo = ImageTk.PhotoImage(Image.fromarray(photo))
            photo_label = Label(frame1,image=photo)
            photo_label.image = photo
            photo_label.pack(side="top",pady=0)

            # name
            Label(frame1,text=name).pack(side='top',pady=5)

            # datetime
            now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            Label(frame1,text=now).pack(side="top",pady=0)
            
            #padx calculation
            #padx = int((wrapper.winfo_width()/2)-(photo_label.winfo_width()/2))
            
            frame1.pack(pady=20,side='top')
            #record["name"] = None
            record['name_with_photo'] = None

        f_canvas.configure(scrollregion = f_canvas.bbox('all'))
        f_canvas.yview_moveto('1.0')
        root.update()

def face_recognition(record,event):
    while True:
        event.wait()
        img_list = record["bbox_data"]
        try:
            for img in img_list:
                encode = embedder.embeddings([img])
                name,idx = recognize_face(encode,x_list,y_list,name_dict,0.71)
                #if name != None:
                if name in tempSentRecord:
                    time_diff = datetime.datetime.now() - tempSentRecord[name]
                    time_sec = time_diff.total_seconds()
                    time_min = time_sec/60
                    if time_min <= 0.1: #minit
                        #print(f'{name} still within 1min cooldown period')
                        pass
                    else:
                        #print(name)
                        #record["name"] = name
                        #record["photo"] = photo_list[idx]
                        record["name_with_photo"] = (name,photo_list[idx])
                        tempSentRecord[name] = datetime.datetime.now()
                else:
                    #print(name)
                    #record["name"] = name
                    #record["photo"] = photo_list[idx]
                    record["name_with_photo"] = (name,photo_list[idx])
                    tempSentRecord[name] = datetime.datetime.now()
        except:
            pass
        event.clear()

# multiprocessing
if __name__ == "__main__":
    
    # creating processes
    manager = multiprocessing.Manager()
    events = multiprocessing.Event()

    records = manager.dict(bbox_data = '',name_with_photo = None)
	
    p1 = multiprocessing.Process(target=face_detection, args=(records,events))
    p2 = multiprocessing.Process(target=face_recognition, args=(records,events))

	# starting process 1
    p1.start()
	# starting process 2
    p2.start()

	# wait until process 1 is finished
    p1.join()
	# wait until process 2 is finished
    p2.join()

	# both processes finished
    print("Done!")


#root.mainloop()