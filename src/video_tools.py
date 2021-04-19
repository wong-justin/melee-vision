'''Editing video and getting images'''

from image_tools import *
import cv2
import numpy as np

def num_frames(filepath):
    video = cv2.VideoCapture(filepath)
    # return video.get(cv2.CAP_PROP_FRAME_COUNT)    # broken because codecs; I think dolphin doesn't encode the frame count
    success, _ = video.read()
    frame_count = 0
    while success:
        success, _ = video.read()
        frame_count += 1
    return frame_count

def trim(frame_start, frame_end, filepath_in, filepath_out):
    video_in  = cv2.VideoCapture(filepath_in)
    video_out = create_similar_writer(video_in, filepath_out)

    # frame_num = frame_start
    # video_in.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)  # opencv bug, need to do alternative (just .read() a bunch I guess)
    #
    # success, img = video_in.read()
    # while success and frame_num < frame_end:
    #     video_out.write(img)
    #
    #     success, img = video_in.read()
    #     frame_num += 1
    #
    # video_in.release()

    ### slower but working method
    i = 0
    for frame in iter_frames(filepath_in):
        if i >= frame_end:
            break
        if i >= frame_start:
            video_out.write(frame)

        i += 1

    video_out.release()

def create_similar_writer(video_capture, filename):
    # returns VideoWriter with same size, fps, codec as given VideoCapture
    codec, width, height, fps = [video_capture.get(prop)
                                 for prop in (cv2.CAP_PROP_FOURCC,
                                              cv2.CAP_PROP_FRAME_WIDTH,
                                              cv2.CAP_PROP_FRAME_HEIGHT,
                                              cv2.CAP_PROP_FPS)]
    # https://stackoverflow.com/questions/61659346/how-to-get-4-character-codec-code-for-videocapture-object-in-opencv
    # h = int(828601953.0)
    # fourcc = chr(h&0xff) + chr((h>>8)&0xff) + chr((h>>16)&0xff) + chr((h>>24)&0xff)
    api_pref = cv2.CAP_FFMPEG   # opencv python VideoWriter constructor forces 6 args, even tho I don't care about api_pref and extra params
    return cv2.VideoWriter(filename, api_pref, int(codec), fps, (int(width), int(height)))#, params=[])

def grab_frame(n, filepath):
    # returns nth image frame in given video
    video = cv2.VideoCapture(filepath)

    success, img = video.read()
    frame_num = 0
    while success and frame_num < n:
        success, img = video.read()
        frame_num += 1
    return img

def iter_frames(filepath):
    # returns generator for image frames in given video
    video = cv2.VideoCapture(filepath)
    success, img = video.read()
    while success:
        yield img
        success, img = video.read()
    video.release()

def range_of_frames(filepath, _range):
    # returns frames indexed by range
    frame_nums = iter_ending_in_none(_range)
    next_num = next(frame_nums)
    for i, img in enumerate(iter_frames(filepath)):

        if next_num is None:  # no more frames to check for
            break

        if i == next_num:
            yield img
            next_num = next(frame_nums)

def iter_ending_in_none(iterable):
    # prevents error on next of last in iterable
    # check for None as last item
    for item in iter(iterable):
        yield item
    yield None

def filter_playback(filepath, filter_fn, title='', interval_ms=int(1000 * 1/120)):
    # shows video playback in new window, applying filter func to each image frame. Esc to close
    for frame in iter_frames(filepath):
        cv2.imshow(title, filter_fn(frame))
        if cv2.waitKey(interval_ms) & 0xFF == 27: # default timing is close to realtime fps I think, but idk why
            break
    cv2.destroyAllWindows()

def write_frames(filepath, frames):
    example_reader  = cv2.VideoCapture('../test/Game_20210408T225110.avi')
    writer = create_similar_writer(example_reader, filepath)

    for img in frames:
        writer.write(img)
    writer.release()

def manually_generate_mask(img, fp='./mask.npy'):
    # utility tool in new window; click and drag to select pixels. Esc to close.
    img_background = upscale_min_to_max_res(img) # rendered to user
    mask = np.zeros(MIN_RES, bool)               # hidden data being modified

    def on_mouse_activated(mouse_x, mouse_y):
        # update data
        x, y = downscale_pt_max_to_min_res((mouse_x, mouse_y))
        mask[x][y] = 1
        # update user
        top_left, bot_right = upscale_pt_min_to_max_res((x, y))
        cv2.rectangle(img_background, top_left, bot_right, (0,0,255), -1)

    def mouse_callback(event, mouse_x, mouse_y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_callback.mouse_down = True
            on_mouse_activated(mouse_x, mouse_y)
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_callback.mouse_down = False
        elif mouse_callback.mouse_down and event == cv2.EVENT_MOUSEMOVE:
            on_mouse_activated(mouse_x, mouse_y)
    mouse_callback.mouse_down = False

    window_name = 'click and drag pixels to mask'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, img_background)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    mask = mask.T
    np.save(fp, mask)
    return mask
