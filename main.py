# Justin Wong
# melee vision

import cv2 as cv
import custom_detector as dt
import background as bg
import matplotlib.pyplot as plt
import numpy as np
import visualize as vs

def main():
    video_path = 'videos/video_whitebg.avi'
    capture = cv.VideoCapture(video_path)
    if not capture.isOpened():
        print('Could not open', video_path)
        return -1

    W = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    H =  int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

    skip_to_frame = 180

    capture.set(cv.CAP_PROP_POS_FRAMES, skip_to_frame-1)   # skip to good starting frame
    is_success, start_frame = capture.read()

    initial_box = (50, 250, 100, 100)

    tracker = dt.MyTracker(start_frame, initial_box)
    fail_tracker = dt.MyTracker(start_frame)


    output_vid = cv.VideoWriter()
    FPS = 30
    output_vid.open('videos/output_video_v6.avi',
##                    'output_video.mp4',   # for web-shareable
                    int(capture.get(cv.CAP_PROP_FOURCC)),
##                    cv.VideoWriter_fourcc(*'h264'),   # for web-shareable mp4s 
                    FPS,
                    (W, H),
                    True)
    if not output_vid.isOpened():
        print('could not open output video for write')
        return -1
    
    output_fail_vid = cv.VideoWriter()
    output_fail_vid.open('videos/output_fails_v2.avi',
##                    'output_video.mp4',   # for web-shareable
                    int(capture.get(cv.CAP_PROP_FOURCC)),
##                    cv.VideoWriter_fourcc(*'h264'),   # for web-shareable mp4s 
                    FPS,
                    (W, H),
                    True)
    if not output_fail_vid.isOpened():
        print('could not open output video for write')
        return -1

    fails = []
    framenum = skip_to_frame
    
##    while True:
    while framenum < 3500:
        is_success, frame = capture.read()
        if not is_success:
            print('no more frames')
            break

        is_success, box = tracker.update(frame)
        x0, y0, x1, y1 = box
        img2 = None
        if not is_success:
            print('tracking failed')
            fail_tracker.update(frame)

            img2 = cv.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 3)
##            cv.imwrite('failed_detections/failure{}.png'.format(framenum), img2)
            
            fails.append(framenum)

            br, cntr_len, plte, _, _ = tracker.stats_of(frame, box)
            #print('brightness:', br, 'longest contour:', cntr_len, 'palette:', plte)
##            output_fail_vid.write(img2)
        else:
            region = dt.pixels_at(frame, box)
            cntrs = dt.contours_of(region)
            img2 = np.zeros((H, W, 3), np.uint8)
            sm_blank = np.zeros((y1-y0, x1-x0, 3), np.uint8)
            sm_blank = cv.drawContours(sm_blank, cntrs, -1, (255,255,255), 1)
            img2[y0:y1, x0:x1] = sm_blank
            
            img2 = cv.rectangle(frame, (x0, y0), (x1, y1), (128, 128, 128), 3)

            output_vid.write(img2)

        cv.imshow('window', img2)

        keyboard = cv.waitKey(1)
        if keyboard == 'q' or keyboard == 27:
            break
        framenum += 1

    output_vid.release()
    output_fail_vid.release()
    capture.release()


    # analyze results        
    vs.display_end_stats(fail_tracker)
    print(len(fails))
##    vs.display_end_stats(tracker)

def cover_character():
    video_path = 'videos/video_whitebg.avi'
    capture = cv.VideoCapture(video_path)

    color_video_path = 'videos/video_fullcolor.avi'
    color_capture = cv.VideoCapture(color_video_path)

    skip = 32
    color_capture.set(cv.CAP_PROP_POS_FRAMES, skip)


    W = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    H =  int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))


    output_vid = cv.VideoWriter()
    FPS = 60
    output_vid.open('videos/output_bg.avi',
                    int(capture.get(cv.CAP_PROP_FOURCC)),
                    FPS,
                    (W, H),
                    True)
    framenum = 0
    
    while True:
##        capture.set(cv.CAP_PROP_POS_FRAMES, framenum)
##        color_capture.set(cv.CAP_PROP_POS_FRAMES, framenum+skip)
        
        success1, bg_frame = capture.read()
        success2, color_frame = color_capture.read()
        if not (success1 and success2):
            print('no more frames')
            break

        cv.imshow('white', bg_frame)

        bool_mask = dt.precise_mask(bg_frame)
        color_frame[bool_mask] = 0

        cv.imshow('color', color_frame)

        
        print(framenum)
        
        keyboard = cv.waitKey()
        if keyboard == 'q' or keyboard == 27:
            break
        framenum += 1

    output_vid.release()
    capture.release()
    color_capture.release()

def keypoint_tracking():

    video_path = 'videos/2player_game.avi'
    video_path = 'videos/battlefield_camera.avi'
    capture = cv.VideoCapture(video_path)

    skip = 1400
    capture.set(cv.CAP_PROP_POS_FRAMES, skip)

    W = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    H =  int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

##    output_vid = cv.VideoWriter()
##    FPS = 30
##    output_vid.open('videos/output_bg.avi',
##                    int(capture.get(cv.CAP_PROP_FOURCC)),
##                    FPS,
##                    (W, H),
##                    True)

    learner = bg.MyBackgroundLearner()
    
    framenum = 0
    while True:
        success, frame = capture.read()
        if not success:
            print('no more frames')
            break

        kps = learner.keypoints(frame)
        cv.imshow('keypoints of frame', frame)
        print(framenum)

        keyboard = cv.waitKey()
        if keyboard == 'q' or keyboard == 27:
            break
        framenum += 1

if __name__ == '__main__':
    keypoint_tracking()
