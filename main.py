# Justin Wong
# melee vision

import cv2 as cv
import custom_detector as mydt
import matplotlib as plt
    
def find_the_thing(frame, last_box):
    last_box = [int(v) for v in last_box]
    x, y, w, h = last_box

    img_edges = cv.Canny(frame, 200, 400)
    cv.imshow('edges', img_edges)
    contours, hierarchy = cv.findContours(img_edges,
                                          cv.RETR_TREE,
                                          cv.CHAIN_APPROX_NONE)
    cv.drawContours(frame, contours, -1, (128, 128, 128), 3)
    cv.imshow('', frame)
    print(len(contours))
    print(contours[0])

def main():
    video_path = 'video_charsonly.avi'

    framenum = 0

    capture = cv.VideoCapture(video_path)

    if not capture.isOpened():
        print('Could not open', video_path)
        sys.exit(-1)

    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height =  int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    print(width, height, frame_count)

##    capture.set(cv.CAP_PROP_POS_FRAMES, 55);
    capture.set(cv.CAP_PROP_POS_FRAMES, 180);
    is_success, frame = capture.read()

    initial_box = (50, 250, 100, 100)
    
##    tracker = cv.TrackerCSRT_create()
##    tracker.init(frame, initial_box)

    tracker = mydt.MyTracker(frame, initial_box)

## output vid
    output_vid = cv.VideoWriter()
    FPS = 30
    output_vid.open('output_video.avi',
                    int(capture.get(cv.CAP_PROP_FOURCC)),
                    FPS,
                    (width, height),
                    True)
    if not output_vid.isOpened():
        print('could not open output video for write')
        return -1


##    cover_mask = (20, 380, 290, 90)

    last_box = None
    
    while True:
##    for _ in range(2):
        is_success, frame = capture.read()
        if not is_success:
            print('no more frames')
            break

        is_success, box = tracker.update(frame)
        if not is_success:
            print('tracking failed')

        x0, y0, x1, y1 = box
        img2 = cv.rectangle(frame, (x0, y0), (x1, y1), (128, 128, 128), 3)
        cv.imshow('window', img2)

        framenum += 1
        last_box = box

        keyboard = cv.waitKey(1)
        if keyboard == 'q' or keyboard == 27:
            break        

##        is_success, box = tracker.update(frame)
##        if is_success:
##            pass
##        else:
##            print('tracking failed', framenum)
####            break
##            
##            my_box = mydt.find_it(frame)   # find roi again
##            x0, y0, x1, y1 = my_box
##            box = (x0, y0, x1-x0, y1-y0)
##            tracker = cv.TrackerCSRT_create()
##            tracker.init(frame, box)
##            
##
##        x, y, w, h = [int(v) for v in box]
##        img2 = cv.rectangle(frame, (x, y), (x+w, y+h), (128, 128, 128), 3)
##
##        cv.imshow('window title', img2)
            
    scores = tracker._brightnesses
    boxes = tracker._boxes
    print(len(boxes))
    print(len(scores))
    scores.sort(reverse=True)
    print('top 10: ', scores[:10])
    print('bottom 10:', scores[-10:])
    print('avg score', sum(scores) / len(scores))    

if __name__ == '__main__':
    main()
