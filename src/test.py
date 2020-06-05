# Justin Wong
# tests for melee vision
import cv2 as cv
import custom_detector as dt
from timeit import default_timer as timer
import background as bg
import numpy as np
import matplotlib.pyplot as plt

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

def test_contours():

    img = cv.imread('yoshis.png')
    img = cv.imread('battlefield.jpg')
    img = cv.imread('dreamland.jpg')
    img = bg.resize(img, 0.8)
    H, W, channels = img.shape

##    img = bg.grayscale(img)
##    img = bg.invert(img)


    img = bg.grayscale(img)
    cv.imshow('g1', img)

    mask = np.ones((H, W), dtype=np.bool_)
    np.putmask(img, mask, (img * 1.2) // 1)
    cv.imshow('g2', img)

    
    img = cv.bilateralFilter(img, 7, 50, 50)
    img = bg.edges(img)

    cv.imshow('edges', img)

    lines = bg.find_horiz_lines(img)
##    lines = bg.find_horiz_lines(lines)
##    lines = img

    cv.imshow('lines', lines)

    ctrs = bg.contours(lines)
    groups = {}
    group_height_size = 30
    for c in ctrs:
        x, y, w, h = cv.boundingRect(c)
        y_lvl = y // group_height_size
##        y_lvl = (y + h / 2) // group_height_size
        if y_lvl not in groups:
            groups[y_lvl] = []
        groups[y_lvl].append(c)

    min_group_size = 1
    groups = {k:v for k,v in groups.items()
              if len(v) > min_group_size}

    print('num contour groups:', len(groups))
    print(groups.keys())

    blank = np.zeros((H, W, 3), dtype=np.uint8)
    last_img = blank
    for ctr_group in groups.values():
        print(len(ctr_group))
        last_img = cv.drawContours(last_img, ctr_group, -1, rnd_color())
        
    cv.imshow('contour groups', last_img)

def test_keypoints():

    img = cv.imread('yoshis.png')
    
    learner = bg.MyBackgroundLearner()
    keypts = learner.keypts(img, 'orb')
    print(keypts)
    img2 = cv.drawKeypoints(img, keypts, None, (0,0,255), flags=0)
    cv.imshow('keypoints', img2)

    k2 = learner.keypts(img, 'fast')
    img3 = cv.drawKeypoints(img, k2, None, (0,0,255), flags=0)
    cv.imshow('k2', img3)

def test_line_detection():
    img = cv.imread('battlefield.jpg')
##    img = bg.resize(img, 0.5)    
##    cv.imshow('initial', img)

    learner = bg.MyBackgroundLearner()

    lines = bg.find_horiz_lines(img)
    lines = bg.find_horiz_lines(lines)
    
##    cv.imshow('horiz lines', lines)

    keypts = learner.keypts(lines, 'orb')
    img3 = cv.drawKeypoints(lines, keypts, -1, (0,0,255), flags=0)
    cv.imshow('lines keypoints', img3)

        
    img4 = bg.vert_lines(lines)
    pt_filter = np.array((
        [-1, -1, -1],
        [-1,  1, -1],
        [-1, -1, -1]
    ), dtype=np.uint8)
    img4 = cv.filter2D(img4, -1, pt_filter)
    img4 = bg.find_corners(img4)
    cv.imshow('lines corners', img4)

    ctrs = bg.contours(lines)
    lines_in_color = cv.cvtColor(lines, cv.COLOR_GRAY2BGR)
    img5 = cv.drawContours(lines_in_color, ctrs, -1, (0,0,255), 1)
    cv.imshow('lines contours', img5)

def test_lines_v2():
    img = cv.imread('yoshis.png')
    img = cv.imread('battlefield.jpg')
##    img = cv.imread('dreamland.jpg')
##    img = cv.imread('pokemon.jpg')
##    img = cv.imread('fod.jpg')
    img = cv.imread('battlefield_shifted.jpg')
    
    img_0 = bg.resize(img)
    img = img_0
    H, W, _ = img_0.shape
    blank = np.zeros((H, W), np.uint8)

##    img = bg.grayscale(img)
    img = cv.bilateralFilter(img, 7, 50, 50)
    img = bg.grayscale(img)

    img = cv.Canny(img, 100, 200)
    ctrs, hierarchy = cv.findContours(img,
                                      cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)    
    scores, ctrs = bg.filter_contours(ctrs)

    img = cv.drawContours(blank.copy(), ctrs, -1, 255, 1)
    
    cv.imshow('contours 1', img)

    img = bg.horiz_lines(img, 5)
    expanded = bg.expand(img, 4)

##    cv.imshow('horiz expanded', expanded)

    ctrs, hierarchy = cv.findContours(expanded,
                                      cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)
    scores, ctrs = bg.filter_contours(ctrs)
        
    img = cv.drawContours(blank.copy(), ctrs, -1, 255, 1)
    early_mask = cv.drawContours(blank.copy(), ctrs, -1, 255, cv.FILLED)
##    cv.imshow('early mask', early_mask)

    cv.imshow('contours 2', img)

##    img = bg.horiz_lines(img, 5)
##    expanded = bg.expand(img, 3)
##    ctrs, hierarchy = cv.findContours(img,
##                                      cv.RETR_EXTERNAL,
##                                      cv.CHAIN_APPROX_SIMPLE)
##    scores, ctrs = bg.filter_contours(ctrs)        
##    img = cv.drawContours(blank.copy(), ctrs, -1, 255, 1)
##    cv.imshow('contours 3', img)
    

    corners = bg.vert_lines(img)
    corners = bg.expand(corners, 3)

##    cv.imshow('corners', corners)
    
    img = cv.Canny(img, 100, 200)
    ctrs, hierarchy = cv.findContours(corners,
                                      cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)

##    blank = np.zeros((H, W), np.uint8)
    fat_corners = cv.drawContours(blank.copy(), ctrs, -1, 255, 1)
    cv.imshow('fat corners', fat_corners)

    mask = blank.copy()
    custom_kps = []
    for c in ctrs:
        x, y, w, h = cv.boundingRect(c)
        px = x + w/2
        py = y + h/2
        radius = max( 10, (w+h)/2 )
##        cv.rectangle(mask, (x,y), (x+w,y+h), 255, -1)
        cv.circle(mask,
                  (int(px), int(py)),
                  int(radius),
                  255,
                  -1)
        custom_kps.append(cv.KeyPoint(
            x=px,
            y=px,
            _size=radius))
    cv.imshow('points mask', mask)
    
        
##    detector = cv.ORB_create()
##    kps, descs = cv.Feature2D.compute(img, custom_kps)
##    detector = cv.ORB.create()
##    detector = cv.xfeatures2d.StarDetector_create()
##    kps, descs = detector.compute(img, custom_kps)
##    kps, descs = cv.xfeatures2d.DAISY.compute(img, cusom_kps)
##    kps, descs = cv.detectAndCompute(img, None, useProvidedKeypoints=True)

    masked = blank.copy()
    masked[mask == 255] = img_0[mask == 255]
    masked = bg.resize(masked, 2)
    cv.imshow('mask filled', masked)
    

    detector = cv.ORB.create()
    kps, descs = detector.detectAndCompute(img_0, early_mask)

    img = cv.drawKeypoints(img_0, kps, img, (0,0,255))
    cv.imshow('keypoints on original', img)
##
##    print(len(ctrs), len(kps))
    
    


##    lines = cv.HoughLines(img,1,np.pi/180,80)
##    print(len(lines))
##
##    img = np.zeros((H, W), dtype=np.uint8)
##    for line in lines:
##        rho = line[0][0]
##        theta = line[0][1]
##        a = np.cos(theta)
##        b = np.sin(theta)
##        x0 = a*rho
##        y0 = b*rho
##        x1 = int(x0 + 1000*(-b))
##        y1 = int(y0 + 1000*(a))
##        x2 = int(x0 - 1000*(-b))
##        y2 = int(y0 - 1000*(a))
##
##        cv.line(img,(x1,y1),(x2,y2),255,1)
##
##    cv.imshow('hough lines', img)
##

def test_brief_descriptor():

    video_path = 'videos/video_fullcolor.avi'
    capture = cv.VideoCapture(video_path)

    skip = 800
    capture.set(cv.CAP_PROP_POS_FRAMES, skip)
    success, frame = capture.read()

    custom_kps = bg.custom_keypoints(frame)


    # decent descriptor but still excludes some important kps
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    kps, des = brief.compute(frame, custom_kps)
    print(len(custom_kps), len(kps))

    blank = frame.copy()
    blank = bg.resize(blank, 0.5)
    cv.drawKeypoints(blank, kps, blank, (0, 0, 255))
    cv.imshow('new computed keypoints', blank)

    gap = 1
    capture.set(cv.CAP_PROP_POS_FRAMES, skip + gap)

    success, frame2 = capture.read()
    kps2 = bg.custom_keypoints(frame)
    kps2, des2 = brief.compute(frame2, kps2)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)

    matches = bf.match(des, des2)

    matches = sorted(matches, key = lambda x : x.distance)
    print(len(matches))

    frame = bg.resize(frame)
    frame2 = bg.resize(frame2)

    result = cv.drawMatches(frame, kps, frame2, kps2, matches, frame2, flags=2)
    cv.imshow('matches', result)

def test_custom_descriptors():
    video_path = 'videos/video_fullcolor.avi'
    capture = cv.VideoCapture(video_path)

    skip = 800
    capture.set(cv.CAP_PROP_POS_FRAMES, skip)
    success, frame = capture.read()
    cv.imshow('frame', frame)

    custom_kps = bg.custom_keypoints(frame)

##    frame = bg.resize(frame)

    patch_radius_norm = 24
    
    H, W = frame.shape[:2]
    num_tests = 64
    test_pairs = gaussian_distr(patch_radius_norm, 2 * num_tests)

    descs = []
    for kp in custom_kps:
        descs.append(binary_descriptor(kp, frame, patch_radius_norm, test_pairs))

##    kp_rows = {}
##    for kp in custom_kps:
##        row = int( (kp.pt[1] // 5) ) * 5
##        if row not in kp_rows:
##            kp_rows[row] = []
##        kp_rows[row].append(kp)
##    for k, v in kp_rows.items():
##        print(k, len(v))
    return
    
def binary_descriptor(kp, frame, patch_size, test_pairs):
    
    patch_radius_norm = patch_size
    
    # descriptor feature possibilities:
    # dominant color
    # average color
    # gradient direction
    # binary brief descriptors seems nice and fast
    x, y = kp.pt
    x, y = int(round(x)), int(round(y))
    
    # options for trying to include edge cases:
    #   0 reject edge cases (i dont wanna)
    #   1 detect if edgy and return smaller radius
    #       less pixels
    #   2 detect if edgy and shift bounding box
    #       more pixels, no longer centered but maybe
    #       that's good for large kps (who represent merged small kps)
    #       and are probably not centered anyways
    #   3 get testing pairs anyways,
    #   check each if illegal (eg negative) and replace/return 0 for testing
    #       lots of checking, least performant
    #       returning 0 may skew results too much in a binary test

def test_match_descriptors():

    video_path = 'videos/battlefield_camera.avi'
    capture = cv.VideoCapture(video_path)

    skip = 1750
    capture.set(cv.CAP_PROP_POS_FRAMES, skip)
    success, frame2 = capture.read()
    gap = 1
    capture.set(cv.CAP_PROP_POS_FRAMES, skip+gap)
    success, frame1 = capture.read()
    H, W = frame1.shape[:2]


    print('starting now')

    kps1 = bg.custom_keypoints(frame1)
    kps2 = bg.custom_keypoints(frame2)

##    des1 = [binary_descriptor(kp,
##                       frame1,
##                       patch_radius,
##                       test_pairs) for kp in kps1]
##    des2 = [binary_descriptor(kp,
##                       frame2,
##                       patch_radius,
##                       test_pairs) for kp in kps2]
    learner = bg.MyBackgroundLearner()
    des1 = learner.binary_descriptors(kps1, frame1)
    des2 = learner.binary_descriptors(kps2, frame2)

    print(len(des1))
    print(len(des2))

    matches_result = []    # des2 matches for des1
    for i in range(len(des1)):
        d1 = des1[i]
        dists = []
        for j in range(len(des2)):
            d2 = des2[j]
            dist = hamming_dist(d1, d2)
            k = 0
            while k < len(dists):
                if dist < dists[k][1]:
                    break
                k += 1
            dists.insert(k, (j, dist))
        matches_result.append(dists)

    kp_match_pairs = []
    for i, kp in enumerate(kps1):
        ordered_matches = matches_result[i]
        best_match = ordered_matches[0]
        dist = best_match[1]
        index_of_match = best_match[0]
        kp_match = kps2[index_of_match]
        
        dist2 = ordered_matches[1][1]
        ratio = dist / dist2    # lowes ratio
        kp_match_pairs.append((kp, kp_match, ratio))

    kp_match_pairs.sort(key=lambda x: x[2])
##    kp_match_pairs = kp_match_pairs[:20]

    
    # display
    ratios = []
    xdiffs = []
    ydiffs = []
    combined = np.hstack((frame1, frame2))
    for i, (kp1, kp2, ratio) in enumerate(kp_match_pairs):
        ratios.append(ratio)
        x0, y0 = kp1.pt
        x1, y1 = kp2.pt
        dy = (y1 - y0)
        dx = (x1 - x0)
        m = slope(dy, dx)

        if abs(dx) > 50 or abs(dy) > 50:
            del kp_match_pairs[i]
            continue

        ydiffs.append(dy)
        xdiffs.append(dx)
        
        pt1 = (int(x0), int(y0))
        pt2 = (int(x1) + W, int(y1))
        color = rnd_color()
        combined = cv.circle(combined, pt1, 5, color)
        combined = cv.circle(combined, pt2, 5, color)
        combined = cv.line(combined, pt1, pt2, color, 1)
        print(round(ratio, 2), dx, dy, m)
        cv.imshow('matches', combined)
        cv.waitKey(1)
    cv.imshow('matches', combined)
    print(sum(xdiffs) / len(xdiffs), sum(ydiffs) / len(ydiffs), slope(sum(ydiffs), sum(xdiffs)))

    pts1 = np.array([kp1.pt for (kp1, kp2, ratio) in kp_match_pairs[:10]])
    pts2 = np.array([kp2.pt for (kp1, kp2, ratio) in kp_match_pairs[:10]])
    
    h, status = cv.findHomography(pts1, pts2)
    print(status, h)

    im_out = cv.warpPerspective(frame1, h, (1000, 1000))
    cv.imshow('warped', im_out)
    return
    
    
    
    ax1 = plt.subplot(211)
    plt.hist(ydiffs)
    ax2 = plt.subplot(212)
    plt.hist(xdiffs)
    plt.show()

def test_match_refining():
    frame1, frame2 = sample_frames(start=4960, gap=3)
    worker = bg.MyBackgroundLearner()
    kps1 = worker.keypoints(frame1)
    kps2 = worker.keypoints(frame2)
    des1 = worker.color_descriptors(kps1, frame1)
    des2 = worker.color_descriptors(kps2, frame2)

    all_match_sets = bg.match(des1, des2, key=bg.compare_palettes)
    refined_match_results = bg.reduce_repeated_matches(all_match_sets)

    print(all_match_sets[0])

    print([bg.top_index_of(match_set) for match_set in refined_match_results])
    print([bg.top_index_of(match_set) for match_set in all_match_sets])
    print(len(refined_match_results))
    print(len(all_match_sets))

def sample_frames(start=4690, gap=3, video='battlefield_camera.avi'):
    capture = cv.VideoCapture('videos/battlefield_camera.avi')
    capture.set(cv.CAP_PROP_POS_FRAMES, start)
    success, frame2 = capture.read()
    capture.set(cv.CAP_PROP_POS_FRAMES, start + gap)
    success, frame1 = capture.read()
    return frame1, frame2

def slope(dy, dx):
    return dy / dx if not dx == 0 else 1000    # arbitrary big represent vertical slope

def test_gaussian_distr():
    radius = 24
    num_pairs = 512
    pairs = bg.patch_pairs(radius, num_pairs)

    
    print(pairs)

    fig, ax = plt.subplots()
    for (pt1, pt2) in pairs:
        x0, y0 = pt1
        x1, y1 = pt2
        plt.plot((x0, x1), (y0, y1), 'bo-')
    plt.xlim(0, 2*radius)
    plt.ylim(0, 2*radius)
    ax.set_aspect(1)
    plt.show()

def test_preprocess_vs_postprocess():
    img = cv.imread('battlefield.jpg')

    learner = bg.MyBackgroundLearner()
    kps = learner.keypoints(img)
    des1 = learner.binary_descriptors(kps, img)
    des2 = learner.binary_descriptors_preprocess(kps, img)
    print(len(kps))

    # only slight difference in pre blur vs post blur
##    for d1, d2 in zip(des1, des2):
##        if not d1 == d2:
##            dist = hamming_dist(d1, d2)
##            print(dist)

    n = 100
    
    print('postprocess')    # a little faster but not much
    time(learner.binary_descriptors, kps, img, n=n)
    print('preprocess')
    time(learner.binary_descriptors_preprocess, kps, img, n=n)    
    

def test_process_patches_vs_process_whole():

    num_patches = 60
    
    img = cv.imread('battlefield.jpg')
    H, W = img.shape[:2]
    pts = [(int( np.random.random() * (W-10) ),
            int( np.random.random() * (H-10) )) for _ in range(num_patches)]

    def process(_img):
        _img = bg.grayscale(_img)
        _img = bg.blur(_img)
        _img = bg.expand(_img)
        return _img

    def get_patch(img, pt):
        return img[pt[1]:pt[1]+10,
                   pt[0]:pt[0]+10]
    
    def process_patches():
        processed_patches = []
        for pt in pts:
            patch = get_patch(img, pt)
            patch = process(patch)
            processed_patches.append(patch)

    def process_whole():
        processed_img = process(img)
        processed_patches = []
        for pt in pts:
            patch = get_patch(processed_img, pt)
            processed_patches.append(patch)

    n = 1000
    print('get patches then process')
    time(process_patches, n=n)
    print('process img then get patches')
    time(process_whole, n=n)
            

def test_mins_vs_ifs():
    W = 604
    H = 580
    x_edge = 601
    y_edge = 1

    x_safe = 300
    y_safe = 200

    def mins(x, y, patch_size=5):
        x_dist = min(x, W - x - 1)
        y_dist = min(y, H - y - 1)
        dist_from_edge = min(x_dist, y_dist)
        patch_size = min(patch_size, dist_from_edge)
        return patch_size

    def ifs(x, y, patch_size=5):
        x_dist = min(x, W - x - 1)
        if x_dist < patch_size:
            patch_size = x_dist
        y_dist = min(y, H - y - 1)
        if y_dist < patch_size:
            patch_size = y_dist
        return patch_size

    def mins_rand():
        x = int( W * np.random.random() )
        y = int( H * np.random.random() )
        mins(x, y)

    def ifs_rand():
        x = int( W * np.random.random() )
        y = int( H * np.random.random() )
        ifs(x, y)

    n = 1000000
    print('mins method')
    time(ifs, x_edge, y_edge, n=n)
    time(ifs, x_safe, y_safe, n=n)
    time(ifs_rand, n=n)

    print('ifs method')
    time(mins, x_edge, y_edge, n=n)
    time(mins, x_safe, y_safe, n=n)
    time(mins_rand, n=n)
    
    
def test_iss_vs_nots():
    def iss():
        return True is True
    def nots():
        return not True is False

    n = 10000000
    time(iss, n=n)
    time(nots, n=n)

def test_subtr_vs_ifs():    
    def sub():
        x = 255 * round(np.random.random())
        r1 = 255 - x

    def ifs():
        x = 255 * round(np.random.random())
        r1 = 255 if x==0 else 0

    iters = 1000000
    time(ifs, n=iters)
    time(sub, n=iters)

def time(func, *args, n=100000):
    start = timer()
    for _ in range(n):
        func(*args)
    end = timer()
    print(end - start)
    return end - start

def rnd_color():
    return [np.random.random() * 255 for _ in range(3)]

if __name__ == '__main__':
    test_gaussian_distr()
    
