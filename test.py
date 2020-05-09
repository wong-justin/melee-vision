# Justin Wong
# tests for melee vision
import cv2 as cv
import custom_detector as dt
from timeit import default_timer as timer

def test_custom_detector():
    sample_img = cv.imread('initialframe.png')
    print(sample_img)
    if sample_img is None:
        print("Couldn't read image")
        return -1
    dt.find_it_v2(sample_img)

def test_int_conversion():
    def slashes():
        return 16 // 4
    def int_():
        return int(16 / 4)
    iters = 1000000
    time(slashes, n=iters)
    time(int_, n=iters)

def test_obj_detection():
    
    img = cv.imread('someframe.png')
    tracker = dt.MyTracker(img)
    iters = 1000

    time(tracker.find_it_v2, img, n=iters)
    time(tracker.find_it, img, n=iters)

def test_brightness():
    img = cv.imread('someframe.png')
    tracker = dt.MyTracker(img)
    results = tracker.find_it_v2(img)
    box = results[0]

    pixels = dt.pixels_at(img, box)
    cv.imshow('', pixels)
    brightness = dt.brightness_of(pixels)

    white_box = (0, 0, 50, 50)
    px_white = dt.pixels_at(img, white_box)
    br_white = dt.brightness_of(px_white)

    br_whole = dt.brightness_of(img)
    
    print(brightness, br_white, br_whole)

def test_if_branches():
    def no_else():
        if True:
            return 1
        else:
            return 0
    def use_else():
        if False:
            return 0
        else:
            return 0

    iters = 10000000
    time(use_else, n=iters)
    time(no_else, n=iters)

def time(func, *args, n=100000):
    start = timer()
    for _ in range(n):
        func(*args)
    end = timer()
    print(end - start)
    return end - start

if __name__ == '__main__':
    test_obj_detection()
