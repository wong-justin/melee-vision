# Justin Wong
# stitch background images together from different angles

import cv2 as cv
import numpy as np

class BackgroundTracker():
    def __init__(self, default_background=None, patch_size=48):
        self.background = default_background
##        self.H, self.W = default_background.shape[:2]
        self.blur_filter = blur_kernel(blur_size=5)
        self.patch_radius = patch_size // 2
        self.patch_test_pairs = patch_pairs(self.patch_radius,
                                            num_pairs=512)

    def update(self, frame):
        # given new frame of background from different angle
        pass

    def keypoints(self, frame, method='custom', num_pts=80):
        if method == 'custom':
            return custom_keypoints(frame)
        elif method == 'shi_tomasi':
            return shi_tomasi_corners(frame, num_pts)
        else:
            return -1

    def binary_descriptors(self, kps, img):
        H, W = img.shape[:2]
        binary_strs = []
        for kp in kps:
            x, y = kp.pt
            x, y = int(round(x)), int(round(y))
            # shrinks if edgy
            curr_patch_radius = validate_if_edge(x, y, W, H, self.patch_radius)
            # extract patch
            x0 = x-curr_patch_radius
            x1 = x+curr_patch_radius
            y0 = y-curr_patch_radius
            y1 = y+curr_patch_radius
            patch = img[y0:y1,
                        x0:x1]
            # scale to normalized size
            ratio_to_full = self.patch_radius / curr_patch_radius
            patch = resize(patch, ratio_to_full)
            # box filter blur to approximate better gaussian blur
            patch = cv.filter2D(patch, -1, self.blur_filter)
            patch = cv.filter2D(patch, -1, self.blur_filter)  # more reps closer to gaussian
            # convert to intensity values
            patch = grayscale(patch)
            # binary test
            bit_str = b''
            for pt1, pt2 in self.patch_test_pairs:  # premade gaussian pt distribution
                if patch[pt1] > patch[pt2]:         # gets (y, x) but doesn't matter, still random
                    bit_str += b'1'
                else:
                    bit_str += b'0'            
            binary_strs.append(bit_str)
        return binary_strs

    def color_descriptors(self, kps, img):
        
        img = reduce(img)
        
        H, W = img.shape[:2]
        palettes = []
        for kp in kps:
            x, y = kp.pt
            x, y = int(round(x)), int(round(y))
            # shrinks if edgy
            curr_patch_radius = validate_if_edge(x, y, W, H, self.patch_radius)
            # extract patch
            x0 = x-curr_patch_radius
            x1 = x+curr_patch_radius
            y0 = y-curr_patch_radius
            y1 = y+curr_patch_radius
            patch = img[y0:y1,
                        x0:x1]
            
            palette = {}
            for row in patch:
                for px in row:
                    px = tuple(px)
                    if px not in palette:
                        palette[px] = 1
                    else:
                        palette[px] += 1
            palettes.append(palette)
        return palettes

#### MATCHING
    
def match(des1, des2, key):
    # des2 matches for des1
    
    all_match_sets = []    
    for i in range(len(des1)):
        d1 = des1[i]
        ordered_matches = []
        for j in range(len(des2)):
            d2 = des2[j]
            score = key(d1, d2)
            k = 0
            while k < len(ordered_matches):
                if score > ordered_matches[k]['score']:
                    break
                k += 1
            match = {'index': j, 'score': score}
            ordered_matches.insert(k, match)
        all_match_sets.append(ordered_matches)
    return all_match_sets

def reduce_repeated_matches(match_sets):
    '''recursive method passes thru match sets once each time to
    knock non-best duplicate matches back to their next best match.

    doesnt seem guaranteed to terminate... but probably since it deletes matches'''

    leading_indexes = [top_index_of(match_set) if len(match_set) > 0 else None
                       for match_set in match_sets]
##    print(leading_indexes)
        
    repeats = repeated(match_sets, key=top_index_of)
    
##    print(len(repeats))
##    for match_set_group in repeats:
##        for match_set in  match_set_group:
##            print(match_set[0])
    if repeats == []:
        return match_sets

        
    for competing_match_sets in repeats:
        repeated_index = competing_match_sets[0][0]['index']
        occurrences = [i for i, x in enumerate(leading_indexes)
                       if x == repeated_index]

        set_to_keep = max(competing_match_sets,
                          key=top_score_of)
        index_to_keep = competing_match_sets.index(set_to_keep)
        occurrences.pop(index_to_keep)

        for set_to_neuter in occurrences:
            match_sets[set_to_neuter].pop(0)
        
    return reduce_repeated_matches(match_sets)

def repeated(iterable, key):
    seen = {}
    for item in iterable:
        if not item:
            continue
        
        attr = key(item)
        if attr not in seen:
            seen[attr] = [item]
        else:
            seen[attr].append(item)

    repeats = []
    for k, vals in seen.items():
        if len(vals) > 1:
            repeats.append(vals)
    return repeats
def merge_sublists(list_of_sublists):
    return [item for sublist in list_of_sublists
            for item in sublist]
def top_index_of(ordered_matches):
    return ordered_matches[0]['index']
def top_score_of(ordered_matches):
    return ordered_matches[0]['score']

####

def compare_palettes(p1, p2):
    # normalize palettes' values during comparison
    p1_tot = sum(p1.values())
    p2_tot = sum(p2.values())
    
    shared = 0
    for color in p1.keys():
        if color in p2:
            shared += min(p1[color] / p1_tot, 
                          p2[color] / p2_tot)
    return 1 - shared   # invert so lower is better, just like hamming dist

def hamming_dist(byte_str1, byte_str2):
    differences = 0
    for bit1, bit2 in zip(byte_str1, byte_str2):
        differences += bit1 ^ bit2  # python can only bitwise operate on ints from byte str,
    return differences              # not bits from bit string sadly

#### KEYPOINTS

def custom_keypoints(img):
    # downsampling, perhaps replace with cv.pyrDown()
    rescale_factor = 2
    img_0 = resize(img, 1 / rescale_factor)
    img = img_0
    H, W, _ = img_0.shape
    blank = np.zeros((H, W), np.uint8)

    img = cv.bilateralFilter(img, 7, 50, 50)
    img = grayscale(img)
    
    #
    img = cv.Canny(img, 100, 200)
    ctrs, hierarchy = cv.findContours(img,
                                      cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)    
    scores, ctrs = filter_contours(ctrs)

    img = cv.drawContours(blank.copy(), ctrs, -1, 255, 1)
    
    img = horiz_lines(img, 5)
    expanded = expand(img, 4)

    ctrs, hierarchy = cv.findContours(expanded,
                                      cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)
    scores, ctrs = filter_contours(ctrs)
        
    img = cv.drawContours(blank.copy(), ctrs, -1, 255, 1)
    early_mask = cv.drawContours(blank.copy(), ctrs, -1, 255, cv.FILLED)    

    corners = vert_lines(img)
    corners = expand(corners, 3)
    
    img = cv.Canny(img, 100, 200)
    ctrs, hierarchy = cv.findContours(corners,
                                      cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)

    fat_corners = cv.drawContours(blank.copy(), ctrs, -1, 255, 1)

    mask = blank.copy()
    custom_kps = []
    for c in ctrs:
        x, y, w, h = cv.boundingRect(c)
        _x = x + w/2
        _y = y + h/2
##        radius = min( 10, (w+h)/2 )
##        radius = max(5, radius)
        radius = max( 5, (w+h)/2 )
        cv.circle(mask,
                  (int(_x), int(_y)),
                  int(radius),
                  255,
                  -1)
        custom_kps.append(cv.KeyPoint(
            x=_x * rescale_factor,          # back to original dimensions
            y=_y * rescale_factor,
            _size=radius))
##    cv.imshow('custom keypoints mask', mask)

    masked = img_0.copy()
    masked[mask == 0] = 0
    masked = resize(masked, 2)
##    cv.imshow('original mask', masked)
    
    return custom_kps

def shi_tomasi_corners(img, num_pts=80):
    gray = grayscale(img)
    corners = cv.goodFeaturesToTrack(gray, num_pts, 0.01, 10)
    corners = np.int0(corners)
    kps = []
    for i in corners:
        x,y = i.ravel()
        kps.append(cv.KeyPoint(
            x=x,
            y=y,
            _size=0))
    return kps

def validate_if_edge(x, y, W, H, patch_size):
    x_dist = min(x, W - x)
    if x_dist < patch_size:
        patch_size = x_dist
    y_dist = min(y, H - y)
    if y_dist < patch_size:
        patch_size = y_dist
    return patch_size
    
def horiz_lines(img, length=3):
    resize_factor = 1
    small = resize(img, 1 / resize_factor)
    gray = grayscale(small)

    def row(n):
        return np.ones(n, dtype=np.int8)

    def long_filter(n):
        return np.array((
            -1 * row(n),
             1 * row(n),
            -1 * row(n)
        ), dtype=np.int8)

    filtered_img = cv.filter2D(gray, -1, long_filter(length))

    big = resize(filtered_img, resize_factor)
    return big    

def find_corners(img):
    img = grayscale(img)
    corners = cv.cornerHarris(img,2,3,0.04)
    corners[corners > 0.01 * corners.max()] = (255)#[0,0,255]
    return corners

def find_corners_v2(img):
    H, W = img.shape
    tl = 1 * np.array((
        [-1, -1, -1],
        [-1,  1,  1],
        [-1,  1, -1]
    ), dtype=np.int8)
    tl = 1 * np.array((
        [-1, -1, -1, -1],
        [-1,  1,  1,  1],
        [-1,  1, -1, -1],
        [-1,  1, -1, -1]
    ), dtype=np.int8)
    tr = np.flip(tl, axis=1)
    bl = np.flip(tl, axis=0)
    br = np.flip(tr, axis=1)

    composite = np.zeros((H, W), dtype=np.uint8)
    for corner_filter in (tl, tr, bl, br):
        composite += cv.filter2D(img, -1, corner_filter)
    return composite

def find_corners_v3(img):
    H, W = img.shape
    corner_filter_L = 1 * np.array((
        [-1, -1, -1],
        [-1, -1,  1],
        [-1, -1, -1]
    ), dtype=np.int8)

    corner_filter_L = 1 * np.array((
        [-1, -1, -1, -1],
        [-1, -1,  1,  1],
        [-1, -1,  1, -1],
        [-1, -1,  1, -1]
    ), dtype=np.int8)
    
    
    corner_filter_R = np.flip(corner_filter_L, axis=1)
    composite = np.zeros((H, W), dtype=np.uint8)
    for corner_filter in (corner_filter_L, corner_filter_R):
        composite += cv.filter2D(img, -1, corner_filter)
    return composite    
    
def vert_lines(img):
    vert_filter = 1 * np.array((
        [-1,  1, -1],
        [-1,  1, -1],
        [-1,  1, -1]
    ), dtype=np.int8)
    img = cv.filter2D(img, -1, vert_filter)
    return img    
    
def resize(img, factor=0.5):
    return cv.resize(img, (0,0),
                     fx=factor,
                     fy=factor)
def grayscale(img):
    if len(img.shape) == 2: # does not have multiple channels
        return img
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def blur_kernel(blur_size=5):
    return 1/(blur_size ** 2) * np.ones((blur_size, blur_size))

def blur(img, factor=2):
    blur_filter = np.ones((factor, factor), np.int8) / (factor ** 2)
    return cv.filter2D(img, -1, blur_filter)

def edges(img):
    return cv.Canny(img, 300, 400)

def expand(img, factor=2):
##    expand_filter = -1 * np.ones((factor+2,factor+2), np.uint8)
##    expand_filter[1:1+factor, 1:1+factor] = 1
##    print(expand_filter)
    expand_filter = np.ones((factor, factor), np.uint8)
    return cv.filter2D(img, -1, expand_filter)

def contours(img):
    edges = cv.Canny(img, 200, 300)
    contours, hierarchy = cv.findContours(edges,
                                          cv.RETR_EXTERNAL ,
                                          cv.CHAIN_APPROX_SIMPLE)
    return contours

def filter_contours(ctrs):
    ctrs_and_scores = []
    for c in ctrs:
        x, y, w, h = cv.boundingRect(c)
        ctrs_and_scores.append((c, w))
    ctrs_and_scores.sort(reverse=True, key=lambda x: x[1])
           
    
    max_w = ctrs_and_scores[0][1]
    w_thresh = 0.1 * max_w
##    for i, (c, w) in enumerate(ctrs_and_scores):
##        if w >= w_thresh:
##            widths.append(w)
##            ctrs.append(c)
##        else:
##            break
    i_max = min(20, len(ctrs))
    widths = []
    ctrs = []
    for i in range(i_max):
        c, w = ctrs_and_scores[i]
        if w >= w_thresh:
            widths.append(w)
            ctrs.append(c)
            
    return widths, ctrs

def filter_contours_v2(ctrs):
    ctrs_and_rects = []
    for c in ctrs:
        x, y, w, h = cv.boundingRect(c)
        ctrs_and_rects.append((c, w, h))

    ctrs_and_rects.sort(reverse=True, key=lambda x: x[1])
    max_width = ctrs_and_rects[0][1]

    width_thresh = 0.1 * max_width
##    width_thresh = 0
    ctrs = []
    for i, (c, w, h) in enumerate(ctrs_and_rects):
        if w >= width_thresh:
            ctrs.append((c, w/h, i)) #ctr, w/h ratio, width_rank
        else:
            break

    ctrs.sort(reverse=True, key=lambda x: x[1])
    for j, (c, r, i) in enumerate(ctrs):
        ctrs[j] = (c, i, j) # ctr, width_rank + ratio_rank

    ctrs.sort(key=lambda x:x[1] + x[2])
    scores = [i+j for (c,i,j) in ctrs]
    print([(i, j) for (c,i,j) in ctrs])
##    ctrs = [c for (c, i, j) in ctrs[:10]]
    ctrs = [c for (c,i,j) in ctrs]
    return scores, ctrs    

def invert(img):
    H, W = img.shape
    mask = np.ones((H, W), np.bool_)
    np.putmask(img, mask, 255 - img)
    return img

def reduce(img, colors=64):
    img = (img // colors) * colors
    return img

def new_pt(mean, std_dev, lower, upper):
    x, y = np.random.normal(mean, std_dev, 2)
    x, y = int(round(x)), int(round(y))
    if (x >= lower and x < upper and
        y >= lower and y < upper):
        return x, y
    else:
        return new_pt(mean, std_dev, lower, upper)

def patch_pairs(radius, num_pairs):
    # only needs to be called once to get static pt pairs for all future tests
    # normal(gaussian) distr clusters in circular center of patch
    # num_points should be less than num pixels in patch, (2*radius)**2
    lower = 0
    upper = 2*radius
    mean = radius
    std_dev = radius / 3    # distr within +-3 std deviations
                            # 0.3% chance it will be outside of radius
    pairs = []
    while len(pairs) < num_pairs:
        pt1 = new_pt(mean, std_dev, lower, upper)
        pt2 = new_pt(mean, std_dev, lower, upper)
        while pt1 == pt2:
            pt2 = new_pt(mean, std_dev, lower, upper)

        if ((pt1, pt2) not in pairs and
            (pt2, pt1) not in pairs):
            pairs.append((pt1, pt2))

    return pairs
