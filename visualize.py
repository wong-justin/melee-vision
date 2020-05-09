# Justin Wong
# visualize results of melee vision using matplotlib charts
import matplotlib.pyplot as plt

# pass in a custom_detector.MyTracker because it has compiled stats

def display_end_stats(tracker):

    plt.subplot(221)
    total_palettes = combine_palettes(tracker._color_palettes)
    plt.pie(total_palettes.values(),
            colors=scaled_rgbs(total_palettes.keys()))
    plt.title('Average colors of detections')

    plt.subplot(222)
    contour_lengths = tracker._contour_lengths
    plt.hist(contour_lengths)
    plt.title('Longest contour length of detections')

    plt.subplot(224)
    brightnesses = tracker._brightnesses
    plt.hist(brightnesses)
    plt.title('Brightness of detections')

    plt.subplot(223)
    boxes = tracker._regions
    x = [v[0] for v in boxes]
    y = [v[1] for v in boxes]
    plt.plot(x, y)
    plt.title('Movement of detections')

    plt.show()
    


def display_palette_diffs(tracker):
    total_palettes = combine_palettes(tracker._color_palettes)
    single_palette = tracker._color_palettes[-60]
    
    shared_ratio = mydt.comp_palettes(total_palettes, single_palette)
    print(shared_ratio)

    _, (p1, p2) = plt.subplots(1, 2, subplot_kw={'aspect':'equal'})
    p1.pie(total_palettes.values(),
           colors=scaled_rgbs(total_palettes.keys()))
    p1.set_title('Average colors of detections')
    
    p2.pie(single_palette.values(),
           colors=scaled_rgbs(single_palette.keys()))
    p2.set_title('Colors from a single detection')
    
    plt.show()
    
def combine_palettes(palettes):
    all_colors = {}
    for p in palettes:
        for c, freq in p.items():
            if c not in all_colors:
                all_colors[c] = 0
            all_colors[c] += freq
            
    return all_colors

def scaled_rgbs(rgbs):
    '''0-255 to 0-1'''
    return [tuple(v / 255 for v in rgb) for rgb in rgbs]

def display_movement_diffs(tracker):
    pts = [(r[0], r[1]) for r in tracker._regions]
    scores = [(0, 0)]
    last_pt = pts[0]
    for i in range(1, len(pts)):
        pt = pts[i]
        xdiff = pt[0] - last_pt[0]
        ydiff = pt[1] - last_pt[1]
        dist = np.sqrt(xdiff ** 2 + ydiff ** 2)
        scores.append((xdiff, ydiff))
        last_pt = pt

    diffs = []
    for i in range(len(scores)-1, 0, -1):
        pt_actual = scores[i]
        pt_pred = scores[i-1]
        xdiff = pt_actual[0] - pt_pred[0]
        ydiff = pt_actual[1] - pt_pred[1]
        dist = np.sqrt(xdiff ** 2 + ydiff ** 2)
        diffs.append(dist)

    diffs.append(0)
    scores = diffs[::-1]
        
    print(len(scores))

    mmin = int(min(scores)) - 1
    mmax = int(max(scores)) + 1

    bins = range(mmin, mmax+2, 3)

    plt.hist(scores, bins)
    plt.xlabel('dist moved')
    plt.ylabel('# frame occurences')
    plt.title('difference from predicted movement')
    plt.grid(True)
    plt.show()  
    
