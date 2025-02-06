from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
from collections import defaultdict
from utils import get_center_of_bbox, get_bbox_width
from PIL import ImageFont, ImageDraw, Image


from deep_sort_realtime.deepsort_tracker import DeepSort


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # DeepSORT 초기화
        self.tracker = DeepSort(max_age=30,
                                n_init=3,
                                nms_max_overlap=1.0,
                                max_cosine_distance=0.2,
                                nn_budget=None,
                                override_track_class=None,
                                embedder="mobilenet",
                                half=True,
                                bgr=True)
        self.previous_ball_owner = None
        self.previous_team = None
        self.commentary = ""
        self.frame_count = 0

    def detect_frames(self, frames):
        batch_size = 1  # DeepSORT는 프레임별로 처리하므로 배치 크기를 1로 설정
        detections = []
        for frame in frames:
            results = self.model.predict(frame, conf=0.1)
            detections.append(results[0])
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            frame = frames[frame_num]
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            bboxes = detection.boxes.xyxy.cpu().numpy()
            scores = detection.boxes.conf.cpu().numpy()
            class_ids = detection.boxes.cls.cpu().numpy().astype(int)

            # 골키퍼를 플레이어로 통합
            for idx, class_id in enumerate(class_ids):
                if cls_names[class_id] == "goalkeeper":
                    class_ids[idx] = cls_names_inv["player"]

            detections_for_tracker = []
            for bbox, score, class_id in zip(bboxes, scores, class_ids):
                # DeepSORT에 필요한 형식으로 변환
                detections_for_tracker.append({
                    'bbox': bbox,
                    'confidence': score,
                    'class': cls_names[class_id]
                })

            # DeepSORT를 사용하여 트랙킹
            tracks_active = self.tracker.update_tracks(detections_for_tracker, frame=frame)

            # 현재 프레임의 트랙 저장
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for track in tracks_active:
                track_id = track.track_id
                ltrb = track.to_ltrb()
                class_name = track.get_class()

                if class_name == 'player':
                    tracks["players"][frame_num][track_id] = {"bbox": ltrb}
                elif class_name == 'referee':
                    tracks["referees"][frame_num][track_id] = {"bbox": ltrb}
                elif class_name == 'ball':
                    tracks["ball"][frame_num][1] = {"bbox": ltrb}  # 볼은 ID 1로 고정

            self.frame_count += 1

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    def add_positions_to_tracks(self, tracks):
        for current_frame in range(len(tracks['players'])):
            for object_type, object_tracks in tracks.items():
                if current_frame < len(object_tracks):
                    frame_tracks = object_tracks[current_frame]
                    if isinstance(frame_tracks, dict):
                        for track_id, track_info in frame_tracks.items():
                            bbox = track_info['bbox']
                            position = get_center_of_bbox(bbox)
                            track_info['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', [np.nan]*4) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate().bfill()
        ball_positions = [{1: {"bbox": x.tolist()}} for x in df_ball_positions.to_numpy()]
        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # 골키퍼를 플레이어로 통합
            for idx, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[idx] = cls_names_inv["player"]

            # 현재 프레임의 감지 결과를 저장할 딕셔너리
            current_frame_tracks = {
                "players": {},
                "referees": {},
                "ball": {}
            }

            # 각 감지된 객체에 대해 처리
            for idx in range(len(detection_supervision.xyxy)):
                bbox = detection_supervision.xyxy[idx].tolist()
                cls_id = detection_supervision.class_id[idx]
                cls_name = cls_names[cls_id]
                position = get_center_of_bbox(bbox)

                # 이전 프레임의 트랙들과 비교하여 매칭
                matched_track_id = self.match_with_previous_tracks(position, cls_name)

                # 매칭된 트랙 ID가 없으면 새로운 트랙으로 추가
                if matched_track_id is None:
                    matched_track_id = self.next_track_id
                    self.next_track_id += 1

                # 트랙 히스토리 업데이트
                self.track_history[matched_track_id] = {
                    "position": position,
                    "bbox": bbox,
                    "last_seen": frame_num
                }

                # 현재 프레임 트랙에 추가
                if cls_name == 'player':
                    current_frame_tracks["players"][matched_track_id] = {"bbox": bbox}
                elif cls_name == 'referee':
                    current_frame_tracks["referees"][matched_track_id] = {"bbox": bbox}
                elif cls_name == 'ball':
                    current_frame_tracks["ball"][1] = {"bbox": bbox}  # 볼은 항상 ID 1로 유지

            # 오래된 트랙 제거 (예: 30프레임 이상 감지되지 않은 트랙)
            self.remove_stale_tracks(frame_num, max_age=30)

            # 트랙 결과 저장
            tracks["players"].append(current_frame_tracks["players"])
            tracks["referees"].append(current_frame_tracks["referees"])
            tracks["ball"].append(current_frame_tracks["ball"])

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def match_with_previous_tracks(self, position, cls_name, max_distance=50):
        """
        현재 감지된 객체를 이전 트랙들과 비교하여 매칭
        """
        min_distance = float('inf')
        matched_track_id = None

        for track_id, track_info in self.track_history.items():
            if track_info.get('class_name') != cls_name:
                continue
            prev_position = track_info['position']
            distance = np.linalg.norm(np.array(position) - np.array(prev_position))
            if distance < min_distance and distance < max_distance:
                min_distance = distance
                matched_track_id = track_id

        return matched_track_id

    def remove_stale_tracks(self, current_frame_num, max_age=30):
        """
        오래된 트랙을 제거하여 메모리 관리 및 트랙 ID 재사용 방지
        """
        stale_track_ids = []
        for track_id, track_info in self.track_history.items():
            if current_frame_num - track_info['last_seen'] > max_age:
                stale_track_ids.append(track_id)
        for track_id in stale_track_ids:
            del self.track_history[track_id]

    def update_ball_owner(self, player_id, team_id):
        if self.previous_ball_owner is None:
            self.commentary = f"Player {player_id} has the ball."
        elif player_id != self.previous_ball_owner:
            if team_id != self.previous_team:
                self.commentary = f"플레이어 {self.previous_ball_owner}이 공을 뺏겼습니다. 플레이어 {player_id}가 공을 소유중입니다.\n Player {self.previous_ball_owner} lost the ball. Player {player_id} now has it."
            else:
                self.commentary = f"플레이어 {self.previous_ball_owner}가 플레이어 {player_id}에게 패스를 하였습니다.\n Player {self.previous_ball_owner} passed the ball to {player_id}."
        self.previous_ball_owner = player_id
        self.previous_team = team_id

    def draw_text(self, frame, text, y_position, font_size=32, color=(255, 255, 255), font_path="fonts/NanumGothic.ttf"):
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, font_size)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        x_position = (frame.shape[1] - text_width) // 2
        draw.text((x_position, y_position), text, font=font, fill=color)
        return np.array(img_pil)

    def draw_annotations(self, video_frames, tracks, team_ball_control, subtitle_texts, event_texts):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # 플레이어 그리기
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

                # 속도 표시
                speed = player.get('speed', 0)
                position = player.get('position', (0, 0))

            # 심판 그리기
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255), track_id)

            # 볼 그리기
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # 팀 볼 컨트롤 그리기
            frame = self.draw_team_ball_control(frame, frame_num, np.array(team_ball_control))

            # 자막 그리기
            frame = self.draw_subtitle(frame, subtitle_texts[frame_num])

            # 이벤트 자막 그리기
            frame = self.draw_event_subtitle(frame, event_texts[frame_num])

            output_video_frames.append(frame)

        return output_video_frames

    def draw_subtitle(self, frame, subtitle_text):
        # PIL 이미지를 생성
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)

        # 한글 폰트 설정
        font_path = "fonts/NanumGothic.ttf"  # 폰트 파일 경로를 설정하세요
        font = ImageFont.truetype(font_path, 32)  # 글꼴 크기를 조정하세요

        # 자막 위치
        position = (50, 850)  # 비디오 해상도에 따라 조정 필요

        # 반투명 배경 추가
        bbox = font.getbbox(subtitle_text)  # getbbox() 메서드를 사용합니다
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x, y = position
        overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_overlay.rectangle(((x - 10, y - 10), (x + text_width + 10, y + text_height + 10)), fill=(0, 0, 0, 128))
        img_pil = Image.alpha_composite(img_pil.convert('RGBA'), overlay)
        draw = ImageDraw.Draw(img_pil)

        # 텍스트 추가
        draw.text(position, subtitle_text, font=font, fill=(255, 255, 255, 255))

        # OpenCV 이미지로 변환
        frame = np.array(img_pil.convert('RGB'))

        return frame

    def draw_event_subtitle(self, frame, event_text):
        if not event_text:
            return frame

        # PIL 이미지를 생성
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)

        # 한글 폰트 설정
        font_path = "fonts/NanumGothic.ttf"
        font = ImageFont.truetype(font_path, 32)

        # 이벤트 자막 위치
        position = (50, 950)

        # 반투명 배경 추가
        bbox = font.getbbox(event_text)  # getbbox() 메서드를 사용합니다
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x, y = position
        overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_overlay.rectangle(
            ((x - 10, y - 10), (x + text_width + 10, y + text_height + 10)),
            fill=(0, 0, 0, 128)
        )
        img_pil = Image.alpha_composite(img_pil.convert('RGBA'), overlay)
        draw = ImageDraw.Draw(img_pil)

        # 텍스트 추가 (노란색으로 강조)
        draw.text(position, event_text, font=font, fill=(255, 255, 0, 255))

        # OpenCV 이미지로 변환
        frame = np.array(img_pil.convert('RGB'))

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # 반투명한 사각형 그리기
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        # 각 팀이 볼을 소유한 프레임 수 계산
        team_1_num_frames = np.sum(team_ball_control_till_frame == 1)
        team_2_num_frames = np.sum(team_ball_control_till_frame == 2)
        total_frames = team_1_num_frames + team_2_num_frames
        if total_frames > 0:
            team_1 = team_1_num_frames / total_frames
            team_2 = team_2_num_frames / total_frames
        else:
            team_1 = team_2 = 0

        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 3)

        return frame

    def draw_subtitles_and_events(self, frame, subtitle_data, event_data, frame_num):
        text_y_main = frame.shape[0] - 50
        frame = self.draw_text(frame, self.commentary, text_y_main, font_size=32)

        if frame_num < len(subtitle_data):
            subtitle_text = subtitle_data[frame_num]
            text_y_subtitle = text_y_main - 40
            frame = self.draw_text(frame, subtitle_text, text_y_subtitle, font_size=24, color=(255, 255, 255))

        if frame_num < len(event_data):
            event_text = event_data[frame_num]
            text_y_event = text_y_main - 80
            frame = self.draw_text(frame, event_text, text_y_event, font_size=24, color=(255, 255, 0))

        return frame
