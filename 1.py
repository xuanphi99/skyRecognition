import numpy as np
import cv2
import math
import pandas as pd
from  tkinter import messagebox as mbox
from tkinter.filedialog import *
from PIL import ImageTk, Image
from tkinter import filedialog,Tk,Button,simpledialog,Label
import matplotlib.image as mpimg
import os, sys
from imutils import paths
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import glob





def preImg(img):
        #img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        img = cv2.resize(img,dsize=(224,224))
        
        return img

def loadData():
     #print('gg')
 data = []
 labels = []
 fileName =[]
 for path in glob.glob('datasetSky2/*/**.jpg'):
        _, brand, fn = path.split('\\')
        # _ : datasetSky2, brand : sunset, fn : fileName
        # tiền xử lý Dl
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img,dsize=(224,224))
        fileName.append(path)
        # trich rút đặc trung
        features = featueVector(img)
        # gán nhãn cho data
        data.append(features)
        labels.append(brand)
#  import seaborn as sns
#  plt.figure(figsize=(9,7))
#  plt.style.use("fivethirtyeight")
#  sns.countplot(labels)
#  plt.show()
 return data,labels,fileName


def writeData(str,data,target,fileName):
#     data,target,fileName = loadData()
 #   print(type(target),len(data),len(fileName))

    dataFrame = []
    for item in range(len(data)):
        arr = []
    # arr.append(data[item])
        for i in data[item]:
            arr.append(i)
    
        arr.append(target[item])
        arr.append(fileName[item])
        dataFrame.append(arr)

    #print(dataFrame[0:10])

    df = pd.DataFrame(dataFrame,columns= ['Trung bình B', 'Trung bình G',
    'Trung bình R',"Độ lệch chuẩn B","Độ lệch chuẩn G","Độ lệch chuẩn R","Nhãn",'URL'])
    df.to_csv (str,index = False, header=True,encoding='utf_8_sig')
    print("write data compelete...")
    

def readData(str):
    
    data = pd.read_csv(str)

    data = data[1:]# delete header
    # data = data.drop(columns=['URL'])# delete index
    data = np.array(data)# covert to matrix
    
    np.random.shuffle(data) # shuffle data
    
    trainSet = data[:800] #training data from 1->100
    testSet = data[801:]# the others is testing data
    return trainSet, testSet

def featueVector(img):
        mean,std = cv2.meanStdDev(img)
        features = np.concatenate([mean, std]).flatten()
        return features
        # B, G, R = cv2.split(img)

        # tb1 =np.sum(B)/(img.shape[0]*img.shape[1])
        # tb2 =np.sum(G)/(img.shape[0]*img.shape[1])
        # tb3 =np.sum(R)/(img.shape[0]*img.shape[1])

        # feature = []
        # feature.append(tb1)
        # feature.append(tb2)
        # feature.append(tb3)
        # sum=0
        # sum2=0
        # sum1=0

        # G=G.flatten()

        # for i in G:
        #         sum += ((i-tb2)*(i-tb2))

        # SSG= sum/(img.shape[0]*img.shape[1]-1)

        # B=B.flatten()

        # for i in B:
        #         sum1 += ((i-tb1)*(i-tb1))
        # SSB= sum1/(img.shape[0]*img.shape[1]-1)       
        # R=R.flatten()
        # for i in R:
        #         sum2 += ((i-tb3)*(i-tb3))
        # SSR= sum2/(img.shape[0]*img.shape[1]-1)

        # feature.append(np.sqrt(SSB))
        # feature.append(np.sqrt(SSG))
        # feature.append(np.sqrt(SSR))

        # return feature


def calcDistancs(pointA, pointB, numOfFeature=6):
    tmp = 0
    for i in range(numOfFeature):
        tmp += (float(pointA[i]) - float(pointB[i])) ** 2
    return math.sqrt(tmp)

def kNearestNeighbor(trainSet, point, k):
    distances = []
    for item in trainSet:
        distances.append({
            "label": item[-2], # get label
            "value": calcDistancs(item, point), # get valude dist
            "fileName" : item[-1]
        }) 
    # thuoc list cac dict gom key = label, value = dist
    distances.sort(key=lambda x: x["value"]) # sort Asc by value


    labels = [item["label"] for item in distances] # duyet list da sort lay ra k label
    fileNames = [item["fileName"] for item in distances]
    dist = [item["value"] for item in distances]
    print(dist[:k])
    return labels[:k],fileNames[:k] # return k point nearest

def findMostOccur(arr): # arr is list k  first labels
   
    print("label  ",arr)
    labels = set(arr) # filter key only == set in java
    print(' ',labels)
    ans = ""
    maxOccur = 0
    for label in labels:
        num = arr.count(label) # dem so phan tu value == label in arr
        print(label , " = ", num)
        if num > maxOccur:
            maxOccur = num
            ans = label
    return ans

def fit(x):

        try:

                train,test= readData(x)
                numOfRightAnwser = 0
                for item in test:
                        knn,knn1 = kNearestNeighbor(train, item, 5)
                        print(type(item))
                        answer = findMostOccur(knn)
                        numOfRightAnwser += item[-2] == answer
                        print(item[-1]) # get last index arr
                        print("label: {} -> predicted: {}".format(item[-2], answer))
                        
                print("Accuracy", numOfRightAnwser/len(test)*100)
                mbox.showinfo( "Accuracy", numOfRightAnwser/len(test)*100)
        except :
                mbox.showerror( "Message", "sai duong dan")

def predict(img,train):
#   img = cv2.imread(anh,cv2.IMREAD_UNCHANGED)
    global anh_like 
    img = preImg(img)

    featue = featueVector(img)
    print(" Dac trung dau vao ",featue)
    
    knn,knn1 = kNearestNeighbor(train,featue,5)
    answer = findMostOccur(knn)
    print(answer,knn1)
    anh_like = knn1[0]
    return answer,knn1[0]

def saveFeature():
      
        data,target,fileName = loadData()
        choose =  mbox.askquestion("Question" , "Lưu đặc trưng ??")
        if choose == 'yes':
                nameFile =  simpledialog.askstring(title="Title",prompt="Nhap ten file ")
                
                if nameFile is not None and nameFile != ''  :
                        nameFile = nameFile+".csv"
                        
                        writeData(nameFile,data,target,fileName)
                        fit(nameFile)
                       # mbox.showinfo( "Message", "Thành công!!!")
                else:
                        mbox.showerror( "Message", "Nhap lai ten file")
        
        
def findkNeast():
        gray = plt.imread(anh_like, cv2.IMREAD_UNCHANGED)


        import matplotlib.image as mpimg
        plt.figure(figsize = (12, 4)) # chia khoảng cho 2 ảnh
        plt.subplot(1, 2, 1) # tạo 2 ô gồm 1 dòng 2 cột ảnh 1 ở cột 1
        img = mpimg.imread(path_img_predict)
        imgplot = plt.imshow(img)

        plt.title(" anh du doan : ")

        plt.subplot(1, 2, 2)
        gray = mpimg.imread(anh_like)
        plt.imshow(gray)
        plt.title("anh gan giong")
        plt.show()   


def open_img():
    # Select path  img
    global path_img_predict
    x = openfilename()
    path_img_predict = x
    print('x= ',x)
    # opens the image
    img = Image.open(x)
    print('img',img)
    # resize the image and apply a high-quality down sampling filter
    img = img.resize((250, 250), Image.ANTIALIAS)
  
    # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)
   
    # create a label
    panel = Label(root, image = img)
      
    # set the image as img 
    panel.image = img
    panel.grid(row = 2)

    
    # doc anh 
    image = mpimg.imread(x)
    print('Moi nhan dạng')
    train,test= readData("export_dataframe.csv")
    arrData = []
    arrData = np.array(train)
    arrData = np.append(arrData,test,axis = 0)
    print("*"*30)
    print(arrData.shape)
    str,img_Like = predict(image,arrData)
    res = Label(root,text=" Ảnh dự đoán- "+ str  ).grid( row = 4, columnspan = 4)

def markCenter(root):
  root.update_idletasks()
  width = root.winfo_width()
  height = root.winfo_height()
  x = (root.winfo_screenwidth() // 2) - (width // 2)
  y = (root.winfo_screenheight() // 2) - (height //2)
  print(root.winfo_screenwidth())
  print(root.winfo_screenheight())
  print(width,height)
  print(x,y)
  root.geometry('{}x{}+{}+{}'.format(width,height,x,y))
  

def openfilename():
      
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    filename = filedialog.askopenfilename(title ='"Open')
    return filename


#btn = Button(root, text ='open image', command = open_img).grid( row = 1, columnspan = 4)
# *********************************************
root = Tk()
root.title("phan loai anh bai troi")
root.resizable(height=True,width=True)
root.minsize(height=500,width=500)

btnfeature = Button(root,text ="Train model", command = saveFeature).grid( row = 1, columnspan = 4)

btnpredict = Button(root,text ="open image", command = open_img).grid( row = 9, columnspan = 4)

btnshow = Button(root,text ="Hiện ảnh tương đồng", command = findkNeast).grid( row = 13, columnspan = 4)


markCenter(root)
root.mainloop()

