import cv2
import mediapipe as mp
import time
import numpy as np
import build_ddnet as ddnet

import random
from tqdm import tqdm
import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt 
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cdist


def data_generator_rt(T,C):
    X_0 = []
    X_1 = []

    T = np.expand_dims(T, axis = 0)
    for i in tqdm(range(len(T))): 
        p = np.copy(T[i])
        p = zoom(p,target_l=C.frame_l,joints_num=C.joint_n,joints_dim=C.joint_d)

        M = get_CG(p,C)

        X_0.append(M)
        p = norm_train2d(p)

        X_1.append(p)

    X_0 = np.stack(X_0)  
    X_1 = np.stack(X_1) 

    return X_0,X_1



# Calculate  features
def zoom(p,target_l=32,joints_num=20,joints_dim=3):
    l = p.shape[0]
    p_new = np.empty([target_l,joints_num,joints_dim]) 
    for m in range(joints_num):
        for n in range(joints_dim):
            p[:,m,n] = medfilt(p[:,m,n],3)
            p_new[:,m,n] = inter.zoom(p[:,m,n],target_l/l)[:target_l]         
    return p_new

def sampling_frame(p,C):
    full_l = p.shape[0] # full length
    if random.uniform(0,1)<0.5: # aligment sampling
        valid_l = np.round(np.random.uniform(0.9,1)*full_l)
        s = random.randint(0, full_l-int(valid_l))
        e = s+valid_l # sample end point
        p = p[int(s):int(e),:,:]    
    else: # without aligment sampling
        valid_l = np.round(np.random.uniform(0.9,1)*full_l)
        index = np.sort(np.random.choice(range(0,full_l),int(valid_l),replace=False))
        p = p[index,:,:]
    p = zoom(p,C.frame_l,C.joint_n,C.joint_d)
    return p

def get_CG(p,C):
    M = []
    iu = np.triu_indices(C.joint_n,1,C.joint_n)
    for f in range(C.frame_l):
        #distance max 
        d_m = cdist(p[f],np.concatenate([p[f],np.zeros([1,C.joint_d])]),'euclidean')       
        d_m = d_m[iu] 
        M.append(d_m)
    M = np.stack(M)   
    return M

def norm_train(p):
    # normolize to start point, use the center for hand case
    # p[:,:,0] = p[:,:,0]-p[:,3:4,0]
    # p[:,:,1] = p[:,:,1]-p[:,3:4,1]
    # p[:,:,2] = p[:,:,2]-p[:,3:4,2]
    # # return p
       
    p[:,:,0] = p[:,:,0]-np.mean(p[:,:,0])
    p[:,:,1] = p[:,:,1]-np.mean(p[:,:,1])
    p[:,:,2] = p[:,:,2]-np.mean(p[:,:,2])
    return p
def norm_train2d(p):
    p[:,:,0] = p[:,:,0]-np.mean(p[:,:,0])
    p[:,:,1] = p[:,:,1]-np.mean(p[:,:,1])
    # p[:,:,2] = p[:,:,2]-np.mean(p[:,:,2])
    return p


#Build DD-Net model
C = ddnet.Config()
DD_Net = ddnet.build_DD_Net(C)
DD_Net.summary()
DD_Net.load_weights('nnkh-8-11-cv.h5')

#10 classes
labels = ['xin chao rat vui duoc gap ban', 'tam biet hen gap lai', 'xin cam on ban that tot bung', 'toi xin loi ban co sao khong', 'toi yeu gia dinh va ban be', 'toi la hoc sinh', 'toi thich dong vat', 'toi an com', 'toi song o viet nam', 'toi la nguoi diec']


colors = [(245,117,16), (117,245,16), (16,117,245)]

#define mediapipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

time0 = 0
sequence = []
sentence = ['']
predictions = []
threshold = 0.6


#access webcam with opencv
cap = cv2.VideoCapture(0)


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()

        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)
        
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        if results.pose_landmarks:
            keypoint = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten()
            keypoint = np.array_split(keypoint, 33)
            #print(keypoint)
            sequence.append(keypoint)
            #print(sequence)
            sequence = sequence[-120:]

        if len(sequence) == 120:
            #print(sequence)
            X_test_rt_1, X_test_rt_2 = data_generator_rt(sequence[-100:], C)
            res = DD_Net.predict([X_test_rt_1, X_test_rt_2])[0]
            # print(labels_text[np.argmax(res)])
            sentence.append(labels[np.argmax(res)])

            sequence.clear()  

            print(sentence)


        # Show fps
        time1 = time.time()
        fps = 1 / (time1 - time0)
        time0 = time1
        cv2.putText(image,'FPS:' + str(int(fps)), (3, 475), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
#         cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, ''.join(sentence[-1:]), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('SIGNTEGRATE', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()




