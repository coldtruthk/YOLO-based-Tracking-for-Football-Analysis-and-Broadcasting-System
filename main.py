from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
    # 비디오 읽기
    video_frames = read_video('input_videos/video.mp4')

    # 추적기 초기화
    tracker = Tracker('models/best.pt')

    # 객체 추적 정보 가져오기
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/track_stubs.pkl'
    )

    # 객체 위치 추가
    tracker.add_positions_to_tracks(tracks)

    # 카메라 움직임 추정
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # 뷰 변환기
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # 볼 위치 보간
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # 속도 및 거리 추정
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # 팀 할당
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # 볼 소유권 할당
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    previous_player_with_ball = -1
    previous_team_with_ball = None

    # 자막 및 이벤트 데이터 수집
    subtitle_data = []
    event_data = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        ball_speed = tracks['ball'][frame_num].get('speed', 0)
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        # 볼 소유자 업데이트 및 코멘터리 설정
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            current_team_with_ball = tracks['players'][frame_num][assigned_player]['team']
            tracker.update_ball_owner(assigned_player, current_team_with_ball)  # 상태와 코멘터리 업데이트
            team_ball_control.append(current_team_with_ball)
        else:
            current_team_with_ball = previous_team_with_ball
            team_ball_control.append(current_team_with_ball)

        # 자막 생성 (tracker.commentary 활용)
        subtitle_text = tracker.commentary
        subtitle_data.append(subtitle_text)

        # 이벤트 감지 및 자막 생성
        event_texts = []

        if previous_player_with_ball != -1 and assigned_player != previous_player_with_ball:
            if assigned_player != -1:
                event_texts.append(f"패스 성공! 플레이어 {previous_player_with_ball} ➡ 플레이어 {assigned_player}")
        elif assigned_player != -1:
            speed = tracks['players'][frame_num][assigned_player].get('speed', 0)
            if speed > 1.5:
                event_texts.append(f"플레이어 {assigned_player}이 드리블 중입니다.")

        # 태클 감지: 팀이 변경되었을 때
        if previous_team_with_ball is not None and current_team_with_ball != previous_team_with_ball:
            event_texts.append("태클 성공! 상대 팀이 볼을 차단했습니다.")

        # 슛 감지
        if ball_speed > 8:  # 슛 속도 임계값
            event_texts.append("슛! 볼이 빠르게 움직입니다.")

        # 골 감지 (골대 영역 좌표를 설정해야 함)
        goal_area = ((100, 50), (200, 100))  # 예제 좌표
        if goal_area[0][0] < ball_bbox[0] < goal_area[1][0] and goal_area[0][1] < ball_bbox[1] < goal_area[1][1]:
            event_texts.append("골! 볼이 골대에 들어갔습니다!")

        # 이벤트 텍스트 합치기
        event_text = "\n".join(event_texts)
        event_data.append(event_text)

        # 이전 프레임 정보 업데이트 (이벤트 감지 후에 위치)
        previous_player_with_ball = assigned_player
        previous_team_with_ball = current_team_with_ball

    # 출력 그리기
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control, subtitle_data, event_data)

    # 카메라 움직임 그리기
    

    # 비디오 저장
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()
