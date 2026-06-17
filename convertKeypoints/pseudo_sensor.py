# convertKeypoints/pseudo_sensor.py
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

class PseudoSensorConverter:
    # 물리 장갑의 센서 위치와 매핑되는 10개의 관절 리스트
    FLEX_JOINTS = [
        [1, 2, 3], [2, 3, 4],      # 엄지 (MCP, IP)
        [0, 5, 6], [5, 6, 7],     # 검지 (MCP, PIP)
        [0, 9, 10], [9, 10, 11],  # 중지 (MCP, PIP)
        [0, 13, 14], [13, 14, 15],# 약지 (MCP, PIP)
        [0, 17, 18], [17, 18, 19] # 소지 (MCP, PIP)
    ]

    """
    MediaPipe 손 랜드마크를 받아 원시 굽힘 각도(Raw Flex Angles) 배열(10개)을 반환합니다.
    (쫙 폈을 때 0도, 굽힐수록 각도 증가)
    """
    
    # 장갑 esp에서 받는 값 형태 
    # 파싱 문자: ntp_timestamp, 엄지_mcp, 엄지_pip, 검지_mcp, 검지_pip, 중지_mcp, 중지_pip, 약지_mcp, 약지_pip, 소지_mcp, 소지_pip
    @classmethod
    def get_flex_sensor_payload(cls, hand_landmarks: list) -> list:


        if not hand_landmarks or len(hand_landmarks) < 21:
            return [0.0] * 10

        landmarks = np.array(hand_landmarks)

        # [0.0, 0.0, 0.0] 으로만 채워진 결측치 방어
        if np.allclose(landmarks, 0.0):
            return [0.0] * 10

        raw_angles = []

        for idx in cls.FLEX_JOINTS:
            p_base, p_joint, p_tip = landmarks[idx[0]], landmarks[idx[1]], landmarks[idx[2]]
            
            v1 = p_base - p_joint
            v2 = p_tip - p_joint
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm == 0 or v2_norm == 0:
                raw_angles.append(0.0)
                continue
                
            dot_product = np.dot(v1, v2)
            cos_theta = np.clip(dot_product / (v1_norm * v2_norm), -1.0, 1.0)
            
            angle_deg = np.degrees(np.arccos(cos_theta))
            raw_flex = 180.0 - angle_deg # 쫙 폄=0도, 꽉 쥠=120도 내외
            
            raw_angles.append(round(float(raw_flex), 4))
            
        return raw_angles
    
    
    """
    손등 평면을 기준으로 Roll, Pitch, Yaw 각도(3개) 추출
    왼손/오른손 여부(is_right_hand)에 따라 Z축 방향을 올바르게 보정합니다.
    """
    @classmethod
    def get_imu_sensor_payload(cls, hand_landmarks: list, is_right_hand: bool = True) -> list:

        error_quat = [0.0, 0.0, 0.0, 1.0]

        if not hand_landmarks or len(hand_landmarks) < 21:
            return error_quat
        
        pts = np.array(hand_landmarks)
        if np.allclose(pts, 0.0):
            return error_quat

        p0 = pts[0]
        p5 = pts[5]
        p9 = pts[9]
        p17 = pts[17]

        # Y축: 손목에서 중지 밑동으로
        v_y = p9 - p0
        if np.linalg.norm(v_y) == 0: 
            return error_quat
        v_y = v_y / np.linalg.norm(v_y)

        # Z축: 외적을 이용한 법선 벡터
        v_5 = p5 - p0
        v_17 = p17 - p0

        # 오른손/왼손에 따라 외적 순서(Z축 방향) 조정 (중요!)
        if is_right_hand:
            v_z = np.cross(v_5, v_17)
        else:
            v_z = np.cross(v_17, v_5)
            
        if np.linalg.norm(v_z) == 0: return error_quat
        v_z = v_z / np.linalg.norm(v_z)

        # X축: Y와 Z의 외적
        v_x = np.cross(v_y, v_z)
        if np.linalg.norm(v_x) == 0: return error_quat
        v_x = v_x / np.linalg.norm(v_x)

        if np.dot(v_x, np.cross(v_y, v_z)) < 0:
            v_z = -v_z  # 좌표계 방향을 강제로 보정

        # 회전 행렬 구성 (3*3)
        rot_matrix = np.column_stack((v_x, v_y, v_z))
        u, _, vh = np.linalg.svd(rot_matrix)
        rot_matrix = u @ vh

        try:
            r = R.from_matrix(rot_matrix)
            qx, qy, qz, qw = r.as_quat()
            
            if qw < 0:
                qx, qy, qz, qw = -qx, -qy, -qz, -qw

            return [round(float(qx), 6), round(float(qy), 6), round(float(qz), 6), round(float(qw), 6)]
        except ValueError:
            # 회전 행렬이 직교하지 않는 등 불량 데이터일 경우 안전하게 0 반환
            return error_quat

    @classmethod
    def convert_to_sensor_data(cls, input_json_path: str):
        """
        원본 MediaPipe JSON 파일을 읽어 센서 데이터셋 포맷으로 변환 후 저장합니다.
        """
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sensor_frames = []

        for frame in data.get("frames", []):
            left_hand = frame.get("left_hand_keypoints", [])
            right_hand = frame.get("right_hand_keypoints", [])

            left_flex = cls.get_flex_sensor_payload(left_hand)
            left_quat = cls.get_imu_sensor_payload(left_hand, is_right_hand=False)

            right_flex = cls.get_flex_sensor_payload(right_hand)
            right_quat= cls.get_imu_sensor_payload(right_hand, is_right_hand=True)

            sensor_frames.append({
                "frame_number": frame["frame_number"],
                "timestamp": frame["timestamp"],
                "left_hand": {
                    "flex_mcp_pip": left_flex, # [10개]
                    "imu_quat": left_quat        # [4개]
                },
                "right_hand": {
                    "flex_mcp_pip": right_flex, # [10개]
                    "imu_quat": right_quat        # [4개]
                }
            })

        output_data = {
        "metadata": data.get("metadata", {}),
        "frames": sensor_frames
        }

        return output_data
    

