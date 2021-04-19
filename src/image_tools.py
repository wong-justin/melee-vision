'''Image filters and computer vision functions'''

import cv2
import numpy as np

### --- resizing

MIN_RES = (73,60)
MAX_RES = (584,480) # original ssbm dims
SCALE_RES = 8       # ratio max / min

def upscale_min_to_max_res(img):
    # 73x60 -> 584x480
    return cv2.resize(img, MAX_RES, interpolation=cv2.INTER_NEAREST)

def downscale_max_to_min_res(img):
    # 584x480 -> 73x60
    return cv2.resize(img, MIN_RES)

def resize_wrapper(filter_fn):
    # downsize, apply filter_fn(img), then upsize
    # used for convenience helper as interface bt pixel data and visualization
    # eg. filter_playback(video, resize_wrapper(my_filter))
    def fn(img):
        small = downscale_max_to_min_res(img)
        filtered = filter_fn(small)
        return upscale_min_to_max_res(filtered)
    return fn

def resize_by(img, factor=0.5):
    # generic resizing, mainly for downsizing
    return cv2.resize(img, (0,0), fx=factor, fy=factor)

def upscale_exact(img, factor=2):
    # for showing pixel data bigger and not blurred
    return cv2.resize(img, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)

def downscale_pt_max_to_min_res(pt):
    # pt in max_res. returns corresponding pixel in min_res
    x, y = pt
    return (int(x/SCALE_RES), int(y/SCALE_RES))

def upscale_pt_min_to_max_res(pt):
    # pixel in min_res. returns corresponding bounding region in max_res: (top_left, bot_right)
    x, y = pt
    pt1 = (x * SCALE_RES,   # top left
           y * SCALE_RES)
    pt2 = ((x + 1) * SCALE_RES, # bot right
           (y + 1) * SCALE_RES)
    return pt1, pt2

### --- filters

def apply_mask(img, mask):
    # mask is shape of img but 1 channel of bool
    # masked pixels turn black
    copy = img.copy()
    copy[mask,:] = 0
    return copy

def apply_mask_blur(img, mask):
    # masked pixels are blurred and half blackened
    copy = img.copy()
    dark_blurred = ((1/2) * blur(copy, 4)).astype(np.uint8)
    return np.where(to_3channels(mask), dark_blurred, copy)

def canny_edge(img):
    # standard canny edge detection
    return cv2.Canny(img, 100, 200)

def sobel_edge(img):
    # sobel edge detection, copied from tutorial
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    img2 = cv2.GaussianBlur(img, (3, 3), 0)
    img3 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(img3, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img3, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

def merge(img1, img2):
    return cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

# def merge_multiple(*imgs):
#     n = len(imgs)
#     shape = imgs[0].shape
#     combined = np.zeros(shape)
#     for img in imgs:
#         combined += (1/n) * img
#     return combined.astype(np.uint8)

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# def bgr_to_grayscale(img):
#     # maybe this specific naming would be better if dealing with differently channeled images
#     pass

def reduce_colors(img, ncolors=64):
    img = (img // ncolors) * ncolors
    return img

def gradients(img, axis=0):
    # axis 0 for vert gradients, 1 for horiz. returns gray.
    gray = to_grayscale(img)
    h, w = gray.shape

    # offset rows (axis 0) or cols (axis 1) to perform fast numpy subtraction
    # subtraction with uint8 can produce up to -255 (int16)
    post_pad = pad_zeros(gray, end=True, nrows=2, axis=axis).astype(np.int16)
    pre_pad = pad_zeros(gray, end=False, nrows=2, axis=axis).astype(np.int16)

    # 1 bad row at beginning and end (edge cases produced from offset)
    diffs = (pre_pad - post_pad)[1:h+1,:] if axis == 0 else (pre_pad - post_pad)[:,1:w+1]   # how to generalize to any axis?
    return normalize_to_uint8(diffs)

    # old slow method:
    # copy = np.zeros((H, W), dtype=np.int16)  # vals can go negative from pxA-pxB, up to -255
    #     # slow native for loops:
    #     for x in range(1, W-1):
    #         for y in range(1, H-1):
    # #             left  = img[y, x-1]
    # #             right = img[y, x+1]
    #
    #             top = img[y-1, x]
    #             bot = img[y+1, x]
    #
    # #             copy[y,x] = int(right) - int(left)
    #             copy[y,x] = int(top) - int(bot)
    # return normalize_to_uint8(copy)

def blur(img, size=2):
    return blur_using_kernel(img, blur_kernel(size))

### --- keypoints

def shi_tomasi_corners(img, num_pts):
    gray = to_grayscale(img)
    corners = cv2.goodFeaturesToTrack(gray, num_pts, 0.01, 10)
    corners = np.int0(corners)
    kps = []
    for i in corners:
        x,y = i.ravel()
        kps.append(cv2.KeyPoint(
            x=x,
            y=y,
            _size=0))
    return kps

def find_keypoints(img, method='shi_tomasi'):
    if method == 'shi_tomasi':
        num_pts = 20    # hardcode just for now
        gray = to_grayscale(img)
        corners = cv2.goodFeaturesToTrack(gray, num_pts, 0.01, 10)
        corners = np.int0(corners)

        return [tuple( c.ravel() ) for c in corners]
        # for c in corners:
        #     yield tuple(c.ravel())   # x, y
            # x,y = c.ravel()
            # yield (int(x), int(y))
    # elif method == 'custom_1':
    #     pass

def draw_pts(img, pts):
    # red marker at positions
    copy = img.copy()
    for pt in pts:
        cv2.circle(copy, pt, 1, (0,0,255), -1)
    return copy

def pts_on_black(pts, shape):
    # white pixel on black background
    black = np.zeros(shape[:2], np.uint8)
    for (x,y) in pts:
        black[y][x] = 255
    return black

### --- patches

PATCH_RADIUS = 4  # around a keypoint
PATCH_DIAMETER = 2 * PATCH_RADIUS

def square_centered_at(pt, radius):
    # returns topleft, botright bounding pts
    x,y = pt
    return (x-radius, y-radius), (x+radius, y+radius)

def neuter_out_of_bounds(rect, bounds):
    (xmin, ymin), (xmax, ymax) = bounds
    (x0, y0), (x1, y1) = rect
#     if x0 < xmin:
#         x0 = xmin
#     if y0 < ymin:
#         y0 = ymin
#     and so forth could be faster?
    x0 = max(x0, xmin)
    y0 = max(y0, ymin)
    x1 = min(x1, xmax)
    y1 = min(y1, ymax)
    return (x0, y0), (x1, y1)

def pad_neutered_patch(patch):
    # if not full square size bc boundary conditions
    h, w = patch.shape[:2]
    if h == PATCH_DIAMETER and w == PATCH_DIAMETER:
    # if h == w == PATCH_DIAMETER:
        return patch
    # center the incomplete patch on full sized black background
    full = np.zeros((PATCH_DIAMETER, PATCH_DIAMETER, 3), np.uint8)
    x0 = (PATCH_DIAMETER - w) // 2
    y0 = (PATCH_DIAMETER - h) // 2
    full[y0:y0+h,x0:x0+w] = patch
    return full

def patch_at(img, pt):
    h, w = img.shape[:2]
    (x0,y0), (x1,y1) = neuter_out_of_bounds(
        square_centered_at(pt, PATCH_RADIUS),
        bounds=((0,0), (w,h))
    )
    return pad_neutered_patch( img[y0:y1,x0:x1] )

def mask_patches_around_pts(img, pts):
    # only show patches around points; black elsewhere
    h, w = img.shape[:2]
    copy = np.zeros_like(img)
    for pt in pts:
        (x0,y0), (x1,y1) = neuter_out_of_bounds(
            square_centered_at(pt, PATCH_RADIUS),
            bounds=((0,0), (w,h))
        )
        copy[y0:y1,x0:x1] = img[y0:y1,x0:x1]
    return copy

def find_patches_at_keypoints(img):
    pts = find_keypoints(img)
    return [patch_at(img, pt) for pt in pts]

def patches_around_pts_on_black(pts, shape):
    h,w = shape[:2]
    black = np.zeros((h,w), np.uint8)
    for pt in pts:
        (x0,y0), (x1,y1) = neuter_out_of_bounds(
            square_centered_at(pt, PATCH_RADIUS),
            bounds=((0,0), (w,h))
        )
        black[y0:y1,x0:x1] = 255
    return black

### --- helpers / misc

def blur_using_kernel(img, kernel):
    return cv2.filter2D(img, -1, kernel)

def blur_kernel(size):
    return 1/(size*size) * np.ones((size, size))

def normalize_to_uint8(gray):
    _min = gray.min()
    span = gray.max() - _min

    return ((gray - _min) * (1/span) * 255).astype(np.uint8)

def pad_zeros(arr, end=True, nrows=1, axis=0):
    # insert nrows of zeros at beginning or end of axis
    new_shape = tuple_assign(arr.shape, axis, nrows)
    zeros = np.zeros(new_shape)
    return np.concatenate((
        (arr, zeros) if end else (zeros, arr)
    ), axis)

def tuple_assign(tup, i, val):
    lis = list(tup)
    lis[i] = val
    return tuple(lis)

def threshold(gray, percentile=1):
    # values in top percentile -> 255, else 0
    h, w = gray.shape
    count = h * w
    vals = gray.flatten()
    percentile_index = int( (percentile/100) * count )
    thresh_index = np.argpartition(vals, -percentile_index)[-percentile_index]
    thresh_val = vals[thresh_index]
#     return np.where(gray > thresh_val, gray, 0)
    return np.where(gray > thresh_val, np.full(gray.shape, 255, np.uint8), 0)

def normalize_gradients(gray):
    # black/white features on gray -> white features on black
    # 128 -> 0; 0,255 -> 255
    diffs = (abs(gray.astype(np.int16) - 128))
#     return (255 - normalize_to_uint8(diffs))
    return normalize_to_uint8(diffs)

### --- channels

def to_3channels(gray):
    return np.stack((gray,)*3, axis=-1)

def to_ycrcb(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)

def split_channels(img):
    # return [c.T for c in masked.T]
    return cv2.split(masked)
