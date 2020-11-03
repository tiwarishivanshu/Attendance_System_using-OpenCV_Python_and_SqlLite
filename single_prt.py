import os

import cv2

import shutil

import numpy as np

from os import listdir

from os.path import isfile, join

import mysql.connector

import getpass #main package to hide the password entered by the user


print("--------------------------------------------------------------------------------------------------------------------")
def updt1():
    mydb = mysql.connector.connect(

        host="localhost",

        user="root",

        passwd="sucessleads",

        database="student"

    )

    my = mydb.cursor()
    my.execute("show tables")
    for x in my:
        if x!="":
            print("table exist:")
        if x== "":
            my.execute(
        "create table section_d( name varchar(20) PRIMARY KEY, attendance int)")  # table creation names section_d
print("--------------------------------------------------------------------------------------------------------------------")

def updt2():
    mydb = mysql.connector.connect(

        host="localhost",

        user="root",

        passwd="sucessleads",

        database="student"

    )

    my = mydb.cursor()
    # module 2

    # creates the student record
    # this function insert the student name in the database

    def insr(na):  # function to insert the name of the student in the table

        name = str(na)

        sql = "insert into section_d(name,attendance) values('%s',0)" % (name)  # the basic sql query

        my.execute(sql)  # execution of the sql query

        mydb.commit()  # necessary to commit the query or the update will not be retained

        print(my.rowcount, "record inserted")

    x = 0

    this = []  # list

    ch = 677

    while ch != 0:

        ch = eval(input("enter your choice\n1-insert\n2-display\n3-exit"))

        if ch == 1:

            if x == 0:

                print("enter the name of the student")

                name = input()

                this.append(name)

                x += 1

                insr(name)

            elif x > 0:

                print("enter the name of the student")

                name = input()

                if name in this:

                    print("the name is already in the list,record not inserted")

                else:

                    this.append(name)

                    insr(name)

        if ch == 2:

            for n in this:
                print(n)

        if ch == 3:
            break

    mydb.close()

print("--------------------------------------------------------------------------------------------------------------------")

def updt3():
    mydb = mysql.connector.connect(

        host="localhost",

        user="root",

        passwd="sucessleads",

        database="student"

    )

    my = mydb.cursor()
    # module 3

    ##this file creates therespective folder
    def path(yu):

            path = "/home/shivanshu/Desktop/prjct/" + yu

            try:

                os.mkdir(path)

            except OSError:

                print("Creation of the directory %s failed" % path)

            else:

                print("Successfully created the directory %s " % path)

    my.execute("select name from section_d")

    myr = my.fetchall()

    this = myr

    for i in this:
        a = str(i)

        a = a.replace("'", "")

        a = a.replace("(", "")

        a = a.replace(")", "")

        a = a.replace(",", "")
        path(a)

print("--------------------------------------------------------------------------------------------------------------------")

def updt4():
    mydb = mysql.connector.connect(

        host="localhost",

        user="root",

        passwd="sucessleads",

        database="student"

    )

    my = mydb.cursor()
    # module fourt
    # creates the dataset for a particular user
    face_classifier = cv2.CascadeClassifier(
        '/home/shivanshu/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

    def face_extractor(img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]

        return cropped_face

    def dtst(n):
        cap = cv2.VideoCapture(0)
        count = 0
        while True:
            ret, frame = cap.read()
            if face_extractor(frame) is not None:
                count += 1
                face = cv2.resize(face_extractor(frame), (400, 400))
                # face = cv2.cvtColor(face, cv2.COLOR_BGR2BGR)
                file_name_path = '/home/shivanshu/Desktop/prjct/' + n + '/user' + str(count) + '.JPG'
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 204, 102), 2)
                cv2.imshow('face cropper', face)
            else:
                print('face not found')
                pass
            if cv2.waitKey(1) == 13 or count == 100:
                break
        cap.release()
        cv2.destroyAllWindows()
        print('collecting sample complete!!!!!!!!')

    my.execute("select name from section_d")
    myr = my.fetchall()
    this = myr
    ch = 456
    for i in this:
        a = str(i)
        a = a.replace("'", "")
        a = a.replace("(", "")
        a = a.replace(")", "")
        a = a.replace(",", "")
        a = str(a)
        print("the dataset is going to be created for:" + a)
        print("press 1 to create the data set:")
        ch = eval(input())
        if ch == 1:
            dtst(a)
        if ch == 0:
            break

print("--------------------------------------------------------------------------------------------------------------------")

def updt5():
    mydb = mysql.connector.connect(

        host="localhost",

        user="root",

        passwd="sucessleads",

        database="student"

    )

    my = mydb.cursor()
    # module 5.1
    # final modeltraining and detection with multiple updates

    # model trainng




    face_classifier = cv2.CascadeClassifier(
        '/home/shivanshu/opencv/data/haarcascades/haarcascade_frontalface_default.xml')  # the opencv xml file

    def face_detector(img, size=0.5):  # method which identifies the user

        gray = cv2.cvtColor(img,
                            cv2.COLOR_BGR2GRAY)  # converting the inamge into grey scale to reduce the data in the image

        faces = face_classifier.detectMultiScale(gray, 1.3, 5)  # accsing the xml file

        if faces is ():
            return img, []

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (225, 255, 255),
                          1)  # draws the rectangular layout on the face input

            roi = img[y:y + h, x:x + w]

            roi = cv2.resize(roi, (300, 300))

        return img, roi

    # roi =regin of intrest

    def mdltr(n):  # model training

        k = 0

        data_path = '/home/shivanshu/Desktop/prjct/' + n + '/'  # path of the folder containing hte dataset

        onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

        Training_Data, Labels = [], []

        for i, files in enumerate(onlyfiles):
            image_path = data_path + onlyfiles[i]

            images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            Training_Data.append(np.asarray(images, dtype=np.uint8))

            Labels.append(i)

        Labels = np.asarray(Labels, dtype=np.int32)

        model = cv2.face.LBPHFaceRecognizer_create()

        model.train(np.asarray(Training_Data), np.asarray(Labels))

        print("model training complete!!!!!!!!!!")

        print("identifing the face of:" + n)

        cap = cv2.VideoCapture(0)  # shooting the camera for capture

        while True:

            ret, frame = cap.read()

            image, face = face_detector(frame)

            try:

                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                result = model.predict(face)

                n = n.strip("\n")

                if result[1] < 500:
                    confidence = int(100 * (1 - (result[1]) / 300))  # estimation

                    display_string = str(
                        confidence) + " " + n  # displays the estimatinon,if the input is a defined user

                    cv2.putText(image, display_string, (470, 380), cv2.FONT_HERSHEY_COMPLEX, .5, (225, 225, 255), 1)
                    # defines the co-ordinate and font style and color of the text which is to be displayed on the output

                if confidence >= 90:

                    print("present")

                    k += 1
                else:
                    print("absent")

                cv2.putText(image, display_string, (470, 380), cv2.FONT_HERSHEY_COMPLEX, .5, (225, 225, 255), 1)

                if confidence > 75:

                    cv2.putText(image, "User Identified", (490, 400), cv2.FONT_HERSHEY_COMPLEX, .5, (225, 255, 225), 1)

                    cv2.imshow('face cropper', image)

                else:

                    cv2.putText(image, 'User Unknown', (490, 400), cv2.FONT_HERSHEY_COMPLEX, .5, (255, 225, 225), 1)

                    cv2.imshow('face cropper', image)

            except:

                cv2.putText(image, 'Face Not Found', (490, 400), cv2.FONT_HERSHEY_COMPLEX, .5, (225, 225, 255), 1)

                cv2.imshow('face cropper', image)

                pass

            if cv2.waitKey(1) == 13:
                break

        cap.release()  # releases the camera, captured camera function

        cv2.destroyAllWindows()

        return k

    jkl = 0

    def update(jkl, n):

        sql = "update section_d set attendance= '%s' where name like'%s'" % (jkl, n)

        my.execute(sql)

        mydb.commit()
        print("updated")

    def calc(hj):

            df=(int(hj) + 1)
            print("converted")
            return df



    my.execute("select name from section_d")  # basic sql query to select the name in the table

    myr = my.fetchall()

    this = myr

    ch = 456

    for i in this:

        a = str(i)

        a = a.replace("'", "")

        a = a.replace("(", "")

        a = a.replace(")", "")

        a = a.replace(",", "")

        print("the model is going to be trained for:" + a)

        print("press 1 to start:")

        ch = eval(input())

        if ch == 1:

            mn = mdltr(a)
            jk=0
            if mn > 10:
                sql = "select attendance from section_d where name like '%s'" % (a)
                my.execute(sql)
                srt = my.fetchall()
                print(srt)
                for j in srt:
                    int(j[0])
                    j = str(j)
                    j = j.replace("'", "")
                    j = j.replace("(", "")
                    j = j.replace(")", "")
                    j = j.replace(",", "")

                    ty = calc(j)
                    update(ty, a)  # typecasting from string to integer

        if ch == 0:
            break

print("--------------------------------------------------------------------------------------------------------------------")

def updt6():

    my = mydb.cursor()
    mydb = mysql.connector.connect(

        host="localhost",

        user="root",

        passwd="sucessleads",

        database="student"

    )

    my = mydb.cursor()
    # module 60

    ##this file is for providing the teacher n number of menus

    print("plz enter the password for acessing privilages")

    p = getpass.getpass()  # hide the password

    print("password entered:", p)


    if p == "1234":

        ch = 123  # preinitialized choice to 123

        while (ch != 0):

            print(
                "1-Update Attendance\n2-Add a student\n3-Erase Record's\n4-Erase Table\n5-Exit")  # choice for the teacher

            ch = eval(input())  # typecasting the entered coide to int type

            ##if the user entered the option 1 He can eithr mark present or absent of the student by entereing his/her name

            if ch == 1:

                print("enter the name of the student:")

                name = str(input())

                my.execute("select name from section_d")

                rslt = my.fetchall()

                bn = 0

                for r in rslt:

                    st = str(r)

                    st = st.replace("('", "")

                    st = st.replace("',)", "")

                    st = str(st)

                    name = str(name)

                    if st in name:
                        bn = 1

                        break

                    if st != name:
                        bn = 0

                if bn == 1:

                    print("enter your choice\n1-to mark present\n2-to mark absent ")

                    che = eval(input())

                    if che == 1:
                        sql = "update section_d set attendance=1 where name like '%s'" % (name)

                        my.execute(sql)

                        mydb.commit()

                        print("the attendance of " + name + " is updated")

                    if che == 2:
                        sql = "update section_d set attendance =0 where name like '%s'" % (name)

                        my.execute(sql)

                        mydb.commit()

                        print("the attendance of " + name + " is updated")

                if bn == 0:
                    print("error, user does not exist")

            ###if the user enters 2 for inserting new record

            if ch == 2:
                print("enter the name of the student you want to add:")
                nm = str(input())

                sql = "insert into section_d(name) values('%s')" % (nm)

                my.execute(sql)

                mydb.commit()

                print("the student named " + nm + " is added in the record")

                ##if the user enters 3 for deleting the record

            if ch == 3:

                print("enter the name of the student:")

                name = str(input())

                my.execute("select name from section_d")

                path="/home/shivanshu/Desktop/prjct/"+name

                shutil.rmtree(path, ignore_errors=False, onerror=None)

                rslt = my.fetchall()

                for r in rslt:

                    st = str(r)

                    st = st.replace("('", "")

                    st = st.replace("',)", "")

                    st = str(st)

                    name = str(name)

                    if st in name:
                        bn = 1

                        break

                    if st != name:
                        bn = 0

                if bn == 1:
                    sql = "delete from section_d where name like '%s'" % (name)

                    my.execute(sql)

                    mydb.commit()

                    print("the attendance of " + name + " is updated")

                if bn == 0:
                    print("error, user does not exist")

            ##if the user enteres the choice 4

            if ch == 4:

                print("enter 1 if your are sure to delete the data in the table else enter 0:")

                gh = eval(input())

                if gh == 1:
                    print("enter the table you want to delete")

                    nmw = str(input())

                    sql = "truncate table %s" % (nmw)

                    my.execute(sql)

                    print("the table named " + nmw + " sucessfully deleted")

            if ch == 5:
                break

print("--------------------------------------------------------------------------------------------------------------------")

ch1=456
while ch1!=0:
    print(
        "--------------------------------------------------------------------------------------------------------------------")
    print("enter the choice\n1-create the table\n2-insert student record:\n3-create folder\n4-create dataset\n5-detection\n6-menu for teacher")
    ch1=eval(input())
    if ch1==1:
        print("create the table:")
        updt1()
    if ch1 == 2:
        print("insert student records:")
        updt2()
    if ch1 == 3:
        print("create the folder:")
        updt3()
    if ch1 == 4:
        print("create dataset:")
        updt4()
    if ch1 == 5:
        print("detection:")
        updt5()
    if ch1 == 6:
        print("menu:")
        updt6()
    else:
        ch=0
    print(
        "--------------------------------------------------------------------------------------------------------------------")

print("--------------------------------------------------------------------------------------------------------------------")
