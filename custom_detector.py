# Justin Wong
# custom detector to find character in image
import cv2 as cv
import numpy as np

class MyTracker():

    def __init__(self, reference_frame=None, reference_box=None):
        details = reference_frame.shape
        self.H, self.W = details[0], details[1]
        self.grid = nested_quads([(0, 0, self.W, self.H)], 3)
        self._boxes = []
        self._brightnesses = []
        self._contour_counts = []

    def update(self, new_frame):
        bounding_box, brightness = self.find_it(new_frame)
        self._boxes.append(bounding_box)
        self._brightnesses.append(brightness)
        is_success = True
        return is_success, bounding_box
    
    def find_it(self, frame):
        '''
        frame is an instance of opencv.Mat
        '''
        # iter thru increasingly small sections of screen to find most pixel activity

        self.grid.sort(reverse=True, key=lambda rect: brightness_of(frame, rect))
        return self.grid[0], brightness_of(frame, self.grid[0])
    

def last_pos_change():
    history_len = 2
    last_pts = [(rect[0], rect[1]) for rect in self.boxes[-history_len:]]
    x0, y0 = last_pts[0]
    x1, y1 = last_pts[1]
    pos_diff_vector = (x1 - x0, y1 - y0)
    return pos_diff_vector
##
##    x, y = (axis for axis in np.transpose(last_pts))
##
##    coeffs = np.polyfit(x, y, degree)
    
def brightness_of(frame, rect):
    x0, y0, x1, y1 = rect
    region = frame[y0:y1, x0:x1]
    avg = cv.mean(region)[0]
##    print(avg)
    return avg

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

def test_it():
    sample_img = cv.imread('initialframe.png')
    _, box = find_it(sample_img)
    print(box)

if __name__ == '__main__':
    test_it()
