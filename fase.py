import cv2
import mediapipe as mp

# Mediapipe 솔루션
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 찾을 수 없습니다.")
    exit()

# Mediapipe 얼굴 메쉬 설정
with mp_face_mesh.FaceMesh(
    max_num_faces=1,  # 탐지할 최대 얼굴 수
    refine_landmarks=True,  # 눈, 입술, 손가락 끝 등 세부 랜드마크
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5  # 특징점 추적의 최소 신뢰도
) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("웹캠을 찾을 수 없습니다.")
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 주요 랜드마크 추출
                nose_tip = face_landmarks.landmark[1]  # 코 끝
                left_eye_outer = face_landmarks.landmark[33]  # 왼쪽 눈 바깥
                right_eye_outer = face_landmarks.landmark[263]  # 오른쪽 눈 바깥

                # 화면 좌표 변환
                h, w, _ = image.shape
                left_eye_coord = (int(left_eye_outer.x * w), int(left_eye_outer.y * h))
                right_eye_coord = (int(right_eye_outer.x * w), int(right_eye_outer.y * h))
                nose_tip_coord = (int(nose_tip.x * w), int(nose_tip.y * h))

                # 눈 중심 계산
                center_x = int((left_eye_outer.x + right_eye_outer.x) / 2 * w)
                
                # 랜드마크 및 중심점 표시
                cv2.circle(image, left_eye_coord, 3, (0, 255, 0), -1)  # 초록색 원
                cv2.circle(image, right_eye_coord, 3, (255, 0, 0), -1)  # 파란색 원
                cv2.circle(image, nose_tip_coord, 3, (0, 0, 255), -1)  # 빨간색 원

                # 얼굴 방향 계산
                nose_x = int(nose_tip.x * w)
                if abs(nose_x - center_x) < 15:  # 세로 중심선과 코의 거리
                    direction = "front"
                elif nose_x < center_x:
                    direction = "left"
                else:
                    direction = "right"


                # 방향 텍스트 표시
                cv2.putText(image, f'{direction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
        cv2.imshow('MediaPipe Face Mesh', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
