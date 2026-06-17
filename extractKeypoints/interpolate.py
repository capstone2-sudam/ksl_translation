import numpy as np
import pandas as pd

KEYPOINT_SIZES = {
    'face_keypoints': 9,
    'lip_keypoints': 40,
    'eyebrow_keypoints': 10,
    'eye_keypoints': 32,
    'pose_keypoints': 8,
    'left_hand_keypoints': 21,
    'right_hand_keypoints': 21
}

# 손 그룹
HAND_GROUPS = [
    'left_hand_keypoints',
    'right_hand_keypoints'
]

# 몸/얼굴 그룹
BODY_FACE_GROUPS = [
    'face_keypoints',
    'lip_keypoints',
    'eyebrow_keypoints',
    'eye_keypoints',
    'pose_keypoints'
]


# 추출된 프레임 데이터의 결측치([0.0, 0.0, 0.0])를 선형 보간
#  손 데이터 최대 보간 프레임 수 (30fps 기준 5프레임 ≈ 0.16초)
def interpolate_landmarks(extracted_data, limit_frames=5):
    if not extracted_data:
        return []
    
    # JSON(3D 데이터)을 Pandas DataFrame으로 평탄화 (Flatten)
    flattened_data = []
    for frame in extracted_data:
        row = {'frame_number': frame['frame_number'], 'timestamp': frame['timestamp']}
        
        for key, expected_size in KEYPOINT_SIZES.items():
            points = np.array(frame.get(key, []), dtype=np.float32)
            
            if len(points) == 0:
                # 데이터가 아예 없는 경우 0 배열로 초기화
                points = np.zeros((expected_size, 3), dtype=np.float32)

            # shape 안정성 보장
            if points.shape != (expected_size, 3):
                points = np.zeros((expected_size, 3), dtype=np.float32)

            # [x, y, z]가 모두 0인 경우를 찾아 NaN으로 치환
            zero_mask = np.isclose(points, 0.0, atol=1e-6).all(axis=1)
            points = points.astype(np.float32)
            points[zero_mask] = np.nan
            
            # x, y, z를 각각의 독립된 열(Column)로 펼침
            for i, (x, y, z) in enumerate(points):
                row[f'{key}_{i}_x'] = x
                row[f'{key}_{i}_y'] = y
                row[f'{key}_{i}_z'] = z
                
        flattened_data.append(row)

    df = pd.DataFrame(flattened_data)

    # 선형 보간법 적용
    cols = [c for c in df.columns if c not in ['frame_number', 'timestamp']]
    
    # 손(Hand) 컬럼 분류
    hand_cols = [c for c in cols 
                 if any(group in c for group in HAND_GROUPS)]
    # 몸통(Pose)과 얼굴(Face, Lip, Eye) 컬럼 분류
    body_face_cols = [c for c in cols 
                      if any(group in c for group in BODY_FACE_GROUPS)]

    # 손의 경우 연속 5프레임 내의 경우에 대해서만 보간 수행
    df[hand_cols] = df[hand_cols].interpolate(method='linear', limit=limit_frames, limit_area='inside', limit_direction='both')
    # 영상 내에 무조건 존재하는 부위이므로 제한 없이 전부 매꾸기
    df[body_face_cols] = df[body_face_cols].interpolate(method='linear', limit_area='inside', limit_direction='both') 
    # 몸통/얼굴 양 끝단 처리: 영상 시작/끝부분의 빈 공간을 앞뒤 값으로 밀어서 채움
    df[body_face_cols] = df[body_face_cols].bfill().ffill()
    # 손 양 끝단 및 긴 결측 처리: 5프레임을 넘어가는 긴 공백은 화면 밖으로 인식하고 0.0 처리
    df[hand_cols] = df[hand_cols].fillna(0.0)

    interpolated_list = []
    for _, row in df.iterrows():
        frame_dict = {
            'frame_number': int(row['frame_number']),
            'timestamp': int(row['timestamp'])
        }
        for key, size in KEYPOINT_SIZES.items():
            pts = []
            for i in range(size):
                x = row[f'{key}_{i}_x']
                y = row[f'{key}_{i}_y']
                z = row[f'{key}_{i}_z']

                # NaN 안전 처리
                x = 0.0 if pd.isna(x) else round(float(x), 6)
                y = 0.0 if pd.isna(y) else round(float(y), 6)
                z = 0.0 if pd.isna(z) else round(float(z), 6)

                pts.append([x, y, z])

            frame_dict[key] = pts
        interpolated_list.append(frame_dict)

    return interpolated_list

