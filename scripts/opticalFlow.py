### optical_flow.py
#
# Optical Flow related functions
# Author: Anoop Naravaram
# Editor: Allen Wang
#
###
import numpy as np
import cv2

from scipy import ndimage
GRID_COLS = 30
GRID_ROWS = 20

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS |
                 cv2.TERM_CRITERIA_COUNT, 10, 0.03))

class OpticalFlowInfo():
    """ Object containing optical flow info """
    def __init__(self, frame):
        self.frame = frame
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.grid_points = np.vstack(
            np.moveaxis(
                np.array(
                    np.meshgrid(
                        np.linspace(0, self.gray.shape[1], GRID_COLS),
                        np.linspace(0, self.gray.shape[0], GRID_ROWS)
                    )
                ),
                0,
                -1
            )
        )[:, np.newaxis].astype(np.float32)

def detectApproacher(ofi1, ofi2):
    p0 = ofi1.grid_points

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(
            ofi1.gray, ofi2.gray, p0, None, **lk_params)

    p_flow = p1 - p0
    p_flow[st == 0] = np.nan
    field = p_flow.reshape(GRID_ROWS, GRID_COLS, 2)

    field[:,:,0] = ndimage.filters.gaussian_filter(field[:,:,0],1)
    field[:,:,1] = ndimage.filters.gaussian_filter(field[:,:,1],1)

    dfieldx_dx = np.gradient(field[:,:,0])[1]
    dfieldy_dy = np.gradient(field[:,:,1])[0]
    div_field = dfieldx_dx + dfieldy_dy

#    print np.nan_to_num(div_field).sum()
    
    if (np.nan_to_num(div_field).sum() > 50):
        return True
    return False
