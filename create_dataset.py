import cv2
import mediapipe as mp
import numpy as np
import time, os
import datetime

videos = []
actions = []
labels = {}
labels_order = []
min_len = 99999
# open labels file and parse for information about videos
with open('gestures/labels/Annot_TrainList.txt', mode='r') as f:
    lines = f.readlines()
    index = 0
    prev_file = ""
    curr_list = []
    curr_file = "not prev"
    file_index = -1


    for line in lines:
        data = [item.strip() for item in line.split(',')]
        curr_file, gesture_name, _, beginning, end, vid_len = data
        actions.append(gesture_name)
        end = int(end)
        beginning = int(beginning)
        vid_len = int(vid_len)
        if min_len > vid_len:
            min_len = vid_len
        info_list = [curr_file, gesture_name, beginning, end, vid_len]
        # keep track of all gestures within the current
        if curr_file != prev_file:
            if file_index != -1:
                videos.append(prev_file)
                labels[prev_file] = curr_list # save current list
                labels_order.append(prev_file)
            index = 0
            file_index += 1
            curr_list = []
        curr_list.append(info_list)
        prev_file = curr_file

print(min_len)

seq_length = min_len - 1

# MediaPipe hands model initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# for file naming purposes
created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

# loop through videos
for i in range(len(videos)):
    # get current video
    name = videos[i]
    video = "gestures/videos/" + name + ".avi"
    print(video)

    # get labels for current video
    list_of_lists = labels[name]
    # action = actions[i]

    # get video loaded
    cap = cv2.VideoCapture(video)

    # count the number of frames
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # calculate duration of the videoq
    seconds = round(frames / fps)
    video_time = datetime.timedelta(seconds=seconds)
    print(f"duration in frames: {frames}")
    print(f"video time: {video_time}")

    # start reading
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    # print(ret)

    # cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
    cv2.imshow('img', img)

    frame = 0
    # curr_begin = 0
    curr_end = 0
    # loop through each action of the video
    for params in list_of_lists:
        data = []
        curr_file, gesture_name, beginning, end, vid_len = params
        frame = 0
        while frame < vid_len:
            ret, img = cap.read()
            if not ret:
                print("video ")
                break

            # image preprocessing
            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None: # hand detection in frame
                # print("made it here")
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints
                    v1 = joint[np.arange(0,20), :3] # Parent joint
                    v2 = joint[np.arange(1,21), :3] # Child joint
                    v = v2 - v1

                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, i)

                    d = np.concatenate([joint.flatten(), angle_label])

                    data.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            frame += 1
            if cv2.waitKey(1) == ord('q'):
                break

        # save processed data
        data = np.array(data)
        np.save(os.path.join('dataset', f'raw_{gesture_name}_{curr_file}_{created_time}'), data)

        full_seq_data = []
        print(data.shape)

        for seq in range(len(data) - seq_length):
            curr = data[seq:seq+seq_length]
            print(data[seq:seq+seq_length].shape)
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        np.save(os.path.join('dataset', f'seq_{gesture_name}_{curr_file}_{created_time}'), full_seq_data)
