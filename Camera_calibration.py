

from helper_functions import *

basepath = os.getcwd()
datapath = basepath + '/data/Calibration_Imgs/'

PatternSize = (9,6)
SquareSize = 1.0
chessboard_corners = []
image_points = []
world_points = []
correspondences = []

## Populate World points.
worldp = np.zeros((PatternSize[1]*PatternSize[0], 3), dtype=np.float64)
worldp[:, :2] = np.indices(PatternSize).T.reshape(-1,2) 
worldp *=SquareSize


## Get point correspondences for all the images
for imagefile in os.listdir(datapath):
    I = cv2.imread(datapath + imagefile)
    ret, corners = cv2.findChessboardCorners(I, patternSize = PatternSize) 
    if ret:
        corners.reshape(-1, 2)
        if corners.shape[0] == worldp.shape[0]:
            image_points.append(corners)
            world_points.append(worldp[:,:-1])
            correspondences.append([corners.astype(np.int),worldp[:, :-1].astype(np.int)])
        else:
            print('Error detecting corners in this image',imagefile)

## Normalization

normalized_correspondences = getNormalizedCorrespondences(correspondences)

## Compute Homography
H = []
for corresp in normalized_correspondences:
    H.append(compute_Homography(corresp))

  ## Further Refine the Homogrphy Estimates

H_refined = []
for i in range(len(H)):
    H_optimized = refine_homography(H[i], normalized_correspondences, corresp)
    H_refined.append(H_optimized)

## Extract intrensic parameters from Homography Matrix

A = get_intrinsic_parameters(H_refined)
