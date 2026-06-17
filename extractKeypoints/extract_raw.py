import cv2
import mediapipe as mp
import numpy as np
import os

# holistic 모델 로드
mp_holistic = mp.solutions.holistic

# 입술 40개
LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415
]

# 눈썹  10개
EYEBROW_INDICES = [
    70, 63, 105, 66, 107,      # 왼쪽 눈썹
    336, 293, 334, 296, 300    # 오른쪽 눈썹
]

EYE_INDICES = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, # 왼쪽 눈
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398 # 오른쪽 눈
]

# Pose 8개 (손, 얼굴 중복 제외)
SELECTED_POSE_INDICES = [11, 12, 13, 14, 15, 16, 23, 24]

# 영상에서 Mediapipe Holistic을 통해 원본 좌표만 추출
# 정규화나 좌표 이동(Offset) 연산은 이후에 수행
def extract_raw_features(video_path):
    extracted_data = []
    cap = cv2.VideoCapture(video_path)
    # 파일이 없는 경우에는 빈 배열로 반환
    if not cap.isOpened():
        print(f"[ERROR] 영상 파일을 열지 못했습니다: {video_path}")
        return extracted_data
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    # 모든 영상의 추출 프레임 fps를 30으로 설정
    target_fps = 30.0
    frame_count = 0
    saved_frame_count = 0

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        min_detection_confidence = 0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break

            frame_count += 1

            # 60fps 영상일 경우 30fps로 다운샘플링 (프레임 스킵)
            if original_fps > 50 and frame_count % 2 != 0:
                continue

            saved_frame_count += 1
            timestamp = int((saved_frame_count / target_fps) * 1000) # 학습 데이터에서만 영상의 시간으로 (실시스템에서는 서버 시간 타임 스탬프)

            # MediaPipe 연산
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame)

            # 데이터 초기화 (결측 시 0,0,0)
            face_pts = np.zeros((9, 3)) # 얼굴 9개 키포인트
            pose_pts = np.zeros((8, 3)) # 포즈 8개 키포인트
            lip_pts = np.zeros((40, 3)) # 입술 40개 키포인트
            brow_pts = np.zeros((10, 3)) # 눈썩 10개 키포인트
            eye_pts = np.zeros((32, 3)) # 눈 32개 키포인트 추가
            left_hands_pts = np.zeros((21, 3)) # 왼손 21개 키포인트
            right_hands_pts = np.zeros((21, 3)) # 오른손 21개 키포인트

            # Pose 추출
            if results.pose_landmarks:
                all_pose = np.array([
                    [lm.x * width, lm.y * height, lm.z * width]
                    for lm in results.pose_landmarks.landmark
                    ])
                face_pts = all_pose[0:9] # 0 ~ 8번
                pose_pts = all_pose[SELECTED_POSE_INDICES]

            # Face Mesh 추출
            if results.face_landmarks:
                for i, idx in enumerate(LIP_INDICES):
                    lm = results.face_landmarks.landmark[idx]
                    lip_pts[i] = [lm.x * width, lm.y * height, lm.z * width]
                for i, idx in enumerate(EYEBROW_INDICES):
                    lm = results.face_landmarks.landmark[idx]
                    brow_pts[i] = [lm.x * width, lm.y * height, lm.z * width]
                for i, idx in enumerate(EYE_INDICES):
                    lm = results.face_landmarks.landmark[idx]
                    eye_pts[i] = [lm.x * width, lm.y * height, lm.z * width]

            # Hands 추출 
            if results.left_hand_landmarks:
                left_hands_pts = np.array([
                    [lm.x * width, lm.y * height, lm.z * width] 
                    for lm in results.left_hand_landmarks.landmark
                    ])
                
            if results.right_hand_landmarks:
                right_hands_pts = np.array([
                    [lm.x * width, lm.y * height, lm.z * width]  
                    for lm in results.right_hand_landmarks.landmark
                    ])

            extracted_data.append({
                "frame_number": saved_frame_count,
                "timestamp": timestamp,
                "face_keypoints": face_pts.tolist(),
                "lip_keypoints": lip_pts.tolist(),
                "eyebrow_keypoints": brow_pts.tolist(),
                "eye_keypoints": eye_pts.tolist(),
                "pose_keypoints": pose_pts.tolist(),
                "left_hand_keypoints": left_hands_pts.tolist(),
                "right_hand_keypoints": right_hands_pts.tolist()
            })

    cap.release()
    return extracted_data