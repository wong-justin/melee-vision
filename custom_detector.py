# Justin Wong
# custom detector to find character in image
import cv2 as cv
import numpy as np
import colorsys
import constants as const
import time

class MyTracker():

    def __init__(self, frame_init, box_init=None):
        details = frame_init.shape
        self.H, self.W = details[0], details[1]
        self.grid = nested_quads([(0, 0, self.W, self.H)], 3) #for find_it_v1
        self._regions = []
        self._brightnesses = []
        self._contour_lengths = []
        self._color_palettes = []

    def update(self, new_frame):
        box = self.find_it_v2(new_frame)
        (brightness,
         contour_length,
         palette,
         color_similarity,
         dist_moved) = self.stats_of(new_frame, box)

        self._regions.append(box)
        self._brightnesses.append(brightness)
        self._contour_lengths.append(contour_length)
        self._color_palettes.append(palette)
        
        a = color_similarity > const.COLOR_SIM_THRESH
        b = brightness < const.BRIGHTNESS_THRESH
        c = contour_length > const.CONTOUR_THRESH
        d = dist_moved < const.DISTANCE_THRESH
        is_success = sum((a, b, c, d)) >= 3

        return is_success, box
    
    def find_it(self, frame):
        '''
        frame is an instance of opencv.Mat
        '''
        # iter thru increasingly small sections of screen to find most pixel activity

##        self.grid.sort(reverse=True, key=lambda rect: brightness_of(pixels_at(frame, rect)))
        self.grid.sort(key=lambda rect: brightness_of(pixels_at(frame, rect)))
        bbox = self.grid[0]
        
        return bbox
    
    def find_it_v2(self, frame):
        box = search_nested_quads((0, 0, self.W, self.H),
                                  lambda q: brightness_of(pixels_at(frame, q)),
                                  3)
        box = list(box)
        # x0,y0,x1,y1
        # 0, 1, 2, 3

        T = (1, -const.BORDER_WIDTH)
        R = (2, const.BORDER_WIDTH)
        B = (3, const.BORDER_WIDTH)
        L = (0, -const.BORDER_WIDTH)
        directions = [T, R, B, L]
        while True:  
            changed = False
            for i, (pos, increase) in enumerate(directions):
                border = box.copy()
                opp_pos = (pos + 2) % 4
                border[opp_pos] = border[pos]
                border[pos] += increase
                success, border = self.rectify(*border)
                
                if success and (
                    brightness_of(pixels_at(frame, border)) < const.BRIGHTNESS_THRESH):
                    box[pos] = border[pos]
                    changed = True
                else:
                    directions.pop(i)
                    
            if not changed:
                break
        return box


    def find_it_v3(self, frame):
        contours = contours_of(frame)
        longest_contour = max(contours, key=lambda x: len(x)) if (
            contours) else None

        x,y,w,h = cv.boundingRect(longest_contour)
        
        return (x, y, x+w, y+h)

    def stats_of(self, frame, box):
        region = pixels_at(frame, box)
        
        brightness = brightness_of(region)
        palette = color_palette_of(region)
        contours = contours_of(region)
        
        longest_contour = len(max(contours, key=lambda x: len(x))) if (
            contours) else 0
        
        dist_moved = dist(box, self._regions[-1]) if (
            not len(self._regions) == 0) else 0
        color_similarity = comp_palettes(palette, self._color_palettes[-1]) if (
            not len(self._color_palettes) == 0) else 0

        return (brightness,
                longest_contour,
                palette,
                color_similarity,
                dist_moved)
    
    def rectify(self, x0, y0, x1, y1):
        '''
        validate edge cases.
        case 1: edges all within frame -> true
        case 2: only outward edge(s) of given rect exceeds frame
                (eg top of rect above top of frame) -> true
        case 3: inward edges of given rect exceeds frame
                (eg bottom of rect above top of frame) -> false, won't fix
        '''
        is_valid = True
        if x0 >= self.W or x1 <= 0 or y0 >= self.H or y1 <= 0:
            return False, (x0, y0, x1, y1)
        if x0 < 0:
            x0 = 0
        if x1 > self.W:
            x1 = self.W
        if y0 < 0:
            y0 = 0
        if y1 > self.H:
            y1 = self.H

        return True, (x0, y0, x1, y1)

def predict_next_pos():
    pass
##
##    x, y = (axis for axis in np.transpose(last_pts))
##
##    coeffs = np.polyfit(x, y, degree)    

def pixels_at(img, rect):
    x0, y0, x1, y1 = rect
    return img[y0:y1, x0:x1]
    
def brightness_of(region):
    avg = cv.mean(region)[0]
##    print(avg)
    return avg

def contours_of(region):
    edges = cv.Canny(region, 200, 300)
    contours, hierarchy = cv.findContours(edges,
                                          cv.RETR_TREE,
                                          cv.CHAIN_APPROX_SIMPLE)
    return contours

def color_palette_of(region):
    details = region.shape
    rows, cols = details[0], details[1]
    region = reduce_colors(region)
    palette = {}
    for y in range(0, rows, const.EVERY_NTH_PX):
        for x in range(0, cols, const.EVERY_NTH_PX):
            px_color = region[y, x]
            px_color = tuple(px_color)[::-1]    #bgr to rgb
            if px_color not in palette:
                palette[px_color] = 0
            palette[px_color] += 1
    return palette

def comp_palettes(p1, p2):
    num_bins = const.NUM_COLORS // 4
    p1 = palette_to_bins(p1, num_bins)
    p2 = palette_to_bins(p2, num_bins)
    p1_tot = sum(p1)
    p2_tot = sum(p2)
    if p1_tot == 0 or p2_tot == 0:
        return 0
    
    # normalization for different sizes
    same = 0
    total = p1_tot
    if not p1_tot == p2_tot:  
        p1 = [v / p1_tot for v in p1]
        p2 = [v / p2_tot for v in p2]
        total = 1
            
    same = 0
    for i in range(num_bins):
        same += min(p1[i], p2[i])
    ratio_same = same / total
    return ratio_same
    
def palette_to_bins(plte, num_bins):

    bins = [0]*num_bins
    for color, freq in plte.items():
        hue = colorsys.rgb_to_hsv(*color)[0]
        bin_num = int(hue * 16)
        bins[bin_num] += freq
    return bins

def reduce_colors(img):
    n = const.NUM_COLORS
    reduced = (img // n) * n
    return reduced

def search_nested_quads(rect, sort_key, depth):
    if depth == 0:
        return rect
    else:
        quad = min(quadrants(*rect), key=sort_key)
        return search_nested_quads(quad, sort_key, depth-1)

def nested_quads(rect_arr, depth=1):
    if depth == 0:
        return rect_arr
    else:
        result = []
        for rect in rect_arr:
            result.extend(quadrants(*rect))
        return nested_quads(result, depth-1)
        
def quadrants(x0, y0, x2, y2):
    y1 = int( (y0 + y2) / 2 )
    x1 = int( (x0 + x2) / 2 )
    return [
        (x0, y0, x1, y1),
        (x1, y0, x2, y1),
        (x0, y1, x1, y2),
        (x1, y1, x2, y2)
    ]

def dist(box1, box2):
    pt1 = (box1[0], box1[1])
    pt2 = (box2[0], box2[1])
    xdiff = pt2[0] - pt1[0]
    ydiff = pt2[1] - pt1[1]
    return np.sqrt(xdiff ** 2 + ydiff ** 2)

def test_it():
    
    pass

if __name__ == '__main__':
    test_it()
