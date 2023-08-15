'''
install cma: https://pypi.org/project/cma/
cma docs: https://cma-es.github.io/apidocs-pycma/


'''
import os
from time import time
import numpy as np, matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)

# set constants
FOV_h = np.deg2rad(56)
FOV_v = np.deg2rad(46)
HEIGHT = 3. #meters (measured: 295 cm)
X_LEN, Y_LEN = 5., 7.5 #meters
CAM_SIZE = 0.3 # upper limit measured is about 30 cm
#plan: if distance between any 2 cameras (E vertex) is < 2*CAM_SIZE: return np.NaN

def coefficients_from_points(p1:np.ndarray, p2=None, p3=None, verbose=False) -> np.ndarray:
    '''
    Given three points, returns the coefficients a, b, c, d of the plane equation of the form ax+by+cz+d=0.
    '''
    if p2 is None or p3 is None:
        assert type(p1) == np.ndarray and p1.shape == (3, 3)
        pq, pr = p1[1, :] - p1[0, :], p1[2, :] - p1[0, :]
        p1 = p1[0, :]
    else:
        assert len(p1) == 3 and len(p2) == 3 and len(p3) == 3
        pq, pr = p2 - p1, p3 - p1
    n = np.cross(pq, pr)
    if verbose: print(f"plane equation: {n[0]:.2f}*x + {n[1]:.2f}*y + {n[2]:.2f}*z + {-n[0]*p1[0] - n[1]*p1[1] - n[2]*p1[2]:.2f} = 0")
    return np.array([n[0], n[1], n[2], -n[0]*p1[0] - n[1]*p1[1] - n[2]*p1[2]])
def algeb_coef2mat_coef(coef:np.ndarray) -> np.ndarray:
    """
    From the ax+by+cz+d=0 and theta=(a,b,c,d) calculate theta_hat, shuch that A@theta_hat=z; A is of the form [[x0,y0,1], [x1,y1,1], ...]
    """
    assert len(coef) == 4, 'Algebraic theta must have 4 coefficients'
    return np.array([-coef[0]/coef[2], -coef[1]/coef[2], -coef[3]/coef[2]])
def plane_from_points(p1, p2, p3):
    pq, pr = p2 - p1, p3 - p1
    n = np.cross(pq, pr)

    # n[0]*(xx-p1[0]) + n[1]*(yy-p1[1]) + n[2]*(z-p1[2]) = 0
    # n[0]*xx - n[0]*p1[0] + n[1]*yy - n[1]*p1[1] + n[2]*z - n[2]*p1[2] = 0
    # print(f"plane equation: {n[0]:.2f}*x + {n[1]:.2f}*y + {n[2]:.2f}*z = {n[0]*p1[0] + n[1]*p1[1] + n[2]*p1[2]:.2f}")

    # set boundaries
    X = np.vstack((p1, p2, p3))
    x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
    y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])

    xx, yy = np.meshgrid(np.linspace((0.8*x_min>0 + 1.25*x_min<0)*x_min, (0.8*x_max<0 + 1.25*x_max>0)*x_max, 8), np.linspace((0.8*y_min>0 + 1.25*y_min<0)*y_min, (0.8*y_max<0 + 1.25*y_max>0)*y_max, 8))
    
    zz = (n[0]*p1[0] + n[1]*p1[1] + n[2]*p1[2] - n[0]*xx - n[1]*yy) / n[2]
    return xx, yy, zz

def plot_seen_points(camera_params:np.ndarray, axes_limits=None, rho=4.0, ax=None) -> None:
    assert len(camera_params)%4==0
    cameras = []
    for i in range(0, len(camera_params), 4):# build cameras
        cameras.append(Camera((camera_params[i], camera_params[i+1]), camera_params[i+2], camera_params[i+3]))
    n_cams = len(cameras)

    if axes_limits:# default axes_limits defined by X_LEN and Y_LEN
        if len(axes_limits) == 4:
            x_min, x_max, y_min, y_max = axes_limits
            z_min, z_max = 0, HEIGHT
            assert x_min<x_max and y_min<y_max
        elif len(axes_limits) == 6:
            x_min, x_max, y_min, y_max, z_min, z_max = axes_limits
            assert x_min<x_max and y_min<y_max and z_min<z_max
        else:
            raise ValueError("axes_limits must be a tuple with length 4 or 6")
    else:
        x_min, x_max = -X_LEN/2, X_LEN/2
        y_min, y_max = -Y_LEN/2, Y_LEN/2
        z_min, z_max = 0, HEIGHT

    xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, int(rho*(x_max-x_min))),
                             np.linspace(y_min, y_max, int(rho*(y_max-y_min))),
                             np.linspace(z_min, z_max, int(rho*(z_max-z_min))))
    n_points = np.prod(xx.shape)
    test_point_set = np.stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)), axis=1)

    mask_mat = np.zeros((n_cams, n_points), dtype=bool)
    for cam_idx, cam in enumerate(cameras):
        mask_mat[cam_idx, :] = cam.peekaboo2(test_point_set)
    mask_seen_by_three = np.zeros(n_points, dtype=bool)
    for idx in range(n_points):
        mask_seen_by_three[idx] = np.sum(mask_mat[:, idx])>=3 #mask_seen_by_three is a n_points long vector of bools, where a True value in the ith position means it was seen by at least 3 cameras

    assert ax is not None, "ax must be given if the seen points are to be plotted"
    ax.scatter(test_point_set[mask_seen_by_three, 0], test_point_set[mask_seen_by_three, 1], test_point_set[mask_seen_by_three, 2], color='black', alpha=0.2)
    # ax.scatter(test_point_set[~mask_seen_by_three, 0], test_point_set[~mask_seen_by_three, 1], test_point_set[~mask_seen_by_three, 2], color='yellow', alpha=0.1) # too messy
    return

def fitness1(camera_params:np.ndarray, axes_limits:tuple, rho=5.0, weighing_dict=None, symmetry=None, minimize=False, verbose=False) -> float:
    """
    KNOWN PROBLEM: VISUALISING NOT-SEEN POINTS CAN BECOME A MESS
    One possible fitness function for measuring the quality of the placement of the cameras. This was relatively easy to implement, however the measured space is of constant size (so maximum space needs to be manually checked for any number of cameras AND if the space is too large, there is a lot of unnecessary calculations) and convexity of the seen space is not taken into account.
    NOTE: fitness should return np.NaN when cameras are too close

    camera_params: 1 dim list describing the cameras. Must be divisible by the number of parameters necessary for one camera
    axes_limits: a tuple of length 6 or 8: (x_min, x_max, y_min, y_max, z_min, z_max) or (x_min, x_max, y_min, y_max)
    rho: number of points in one direction within a unit of distance
    minimize: bool, generally only true when passing the function to CMA-ES trainer/solver
    """
    assert len(camera_params)%4==0
    assert len(axes_limits)==6, "axes_limits must be a tuple with length 6"
    assert weighing_dict is not None, "weighing dict parameter must be given"
    if symmetry: camera_params = symmetric_params2regular_params(camera_params, symmetry)
    x_min, x_max, y_min, y_max, z_min, z_max = axes_limits
    cameras = []
    for i in range(0, len(camera_params), 4):
        # build cameras
        cameras.append(Camera((camera_params[i+0], camera_params[i+1]), camera_params[i+2], camera_params[i+3]))
    n_cams = len(cameras)
    if weighing_dict['stay_within_range']>=0:
        x_min, x_max, y_min, y_max = (1+weighing_dict['stay_within_range'])*np.array(axes_limits[:-2])
        assert np.allclose(np.zeros(2), np.array((x_min+x_max, y_min+y_max))), "The 'stay_within_range' funcitonality assumes that the origin of the x-y plane is at (0, 0)"
        for cam in cameras:
            if not (np.logical_and(x_min<cam.vertices_m[:, 0], cam.vertices_m[:, 0]<x_max).all() and np.logical_and(y_min<cam.vertices_m[:, 1], cam.vertices_m[:, 1]<y_max).all()):
                return np.NaN
        x_min, x_max, y_min, y_max, z_min, z_max = axes_limits
    assert x_min<x_max and y_min<y_max and z_min<z_max, "minimum values must be strictly smaller than maximum values"
    if minimize:# if training (minimize=True), enforce that cameras are not to be too close to one another
        for i in range(n_cams-1):
            for j in range(i+1, n_cams):
                if np.linalg.norm(cameras[i].vertices_d['E']-cameras[j].vertices_d['E']) < 2*CAM_SIZE:
                    if verbose: print("solution rejected: cameras too close")
                    return np.NaN
    # generate test-points
    xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, int(rho*(x_max-x_min))),
                             np.linspace(y_min, y_max, int(rho*(y_max-y_min))),
                             np.linspace(z_min, z_max, int(rho*(z_max-z_min))))
    n_points = np.prod(xx.shape)
    test_point_set = np.stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)), axis=1)
    # every row of mask_mat corresponds to a camera and every column to a test-point if i_th camera sees j_th point then mask_mat[i, j]==True
    mask_mat = np.zeros((n_cams, n_points), dtype=bool)
    for cam_idx, cam in enumerate(cameras):
        assert type(cam)==Camera, "All element of 'cameras' must be of type 'Camera'"
        mask_mat[cam_idx, :] = cam.peekaboo2(test_point_set)
    mask_seen_by_three = np.zeros(n_points, dtype=bool)
    for idx in range(n_points):
        mask_seen_by_three[idx] = np.sum(mask_mat[:, idx])>=3 #mask_seen_by_three is a n_points long vector of bools, where a True value in the ith position means it was seen by at least 3 cameras

    result = 0
    max_result = 0
    # weigh each point according to how far it is from the origin
    # a: how strongly this weighing should be considered within the point-by-point basis weights; b: how strongly should proximity to origin be favored
    if type(weighing_dict['distance_from_origin']) == float:
        if weighing_dict['distance_from_origin']>0:
            a = weighing_dict['distance_from_origin']
            b = 1
            weights = 1/(1 + b*np.linalg.norm(test_point_set, axis=1))
            result += a * np.sum(weights*mask_seen_by_three)/np.sum(weights) 
            max_result += a
    elif type(weighing_dict['distance_from_origin']) == tuple:
        if np.all(np.array(weighing_dict['distance_from_origin'])>0):
            assert len(weighing_dict['distance_from_origin'])==2, "If the distance form origin parameter is a tuple, it must be of length 2."
            a, b = weighing_dict['distance_from_origin']
            weights = 1/(1 + b*np.linalg.norm(test_point_set, axis=1))
            result += a * np.sum(weights*mask_seen_by_three)/np.sum(weights) 
            max_result += a
    else: raise Exception("Distance form origin parameter must be a float or a tuple of length 2.")

    # couldn't figure out a way to normalise spread, therefore this functionality will break the 0-1 range of the fitness funciton. To somehow make it still scale by the space, I divide by the x and y size of the space
    if weighing_dict['spread']>0:
        cam_positions = np.zeros((n_cams, 2))
        for i in range(n_cams):# build cameras
            cam_positions[i, 0] = camera_params[4*i]
            cam_positions[i, 1] = camera_params[4*i+1]
        centroid = np.mean(cam_positions, axis=0)
        distances = np.linalg.norm(np.stack((cam_positions[:, 0]-centroid[0], cam_positions[:, 1]-centroid[1])), axis=0)
        max_distance_sum = (x_max-x_min)*(y_max-y_min) #not really the maximum spread, but this value should at least scale with the space
        result += weighing_dict['spread'] * np.sum(distances)/max_distance_sum
        max_result += weighing_dict['spread']

    if weighing_dict['soft_convexity'] > 0 or weighing_dict['hard_convexity'] > 0:
        test_points_seen = test_point_set[mask_seen_by_three]
        try:
            hull = ConvexHull(test_points_seen)
        except:
            # print('Convex hull could not be computed')
            # HULL_FAIL = HULL_FAIL + 1
            return np.NaN
        triangulation = Delaunay(test_points_seen[hull.vertices])
        mask_inside_hull = triangulation.find_simplex(test_point_set) >= 0 #mask for points inside the hull
        n_points_in_hull_not_seen = np.sum(np.logical_and(mask_inside_hull, ~mask_seen_by_three))
        n_seen = np.sum(mask_seen_by_three)
        convexity_ratio = n_seen/(n_seen+n_points_in_hull_not_seen)
        assert 0 <= convexity_ratio <= 1, "Something is not right: convexity ratio out of range"
        # soft convexity
        if weighing_dict['soft_convexity'] > 0:
            result += weighing_dict['soft_convexity']*convexity_ratio
            max_result += weighing_dict['soft_convexity']
        # hard convexity
        if weighing_dict['hard_convexity'] > 0:
            assert weighing_dict['hard_convexity'] < 1, "hard_convexity parameter must be strictly less than 1.0 (ideally below 0.95)"
            if convexity_ratio < weighing_dict['hard_convexity']: return np.NaN #reject every solution that is below the hard limit

    if verbose: print(f'The cameras see {100*np.sum(mask_seen_by_three)/n_points:.1f}% of points')

    # every weghing parameter is turned off
    if max_result == 0:
        result = np.sum(mask_seen_by_three)/n_points
        max_result = 1
    
    return 1 - result/max_result if minimize else result/max_result

def fitness2(cameras, axes_limit_init:tuple, threshold=0.2, lambd=0.1, rho=10.0) -> tuple:
    """
    KNOWN PROBLEM: GOES JUST A LITTLE BIT BELOW THE THRESHOLD
    Another possible fitness function. It measures the fitness of a solution by finding the maximum space where the ratio of the seen space is above the threshold
    cameras: same as fitness1
    axes_limit_init: initial guesstimation for smallest space with seen points >= threshold
    threshold: at least this portion of the space must be visible by 3 cameras at the same time
    lambd: constant stepsize for incrementing the space
    rho: density of the test particles
    """
    assert lambd>0
    assert type(axes_limit_init)==tuple and len(axes_limit_init)==6
    x_min, x_max, y_min, y_max, z_min, z_max = axes_limit_init
    ftnss = fitness1(cameras, (x_min, x_max, y_min, y_max, z_min, z_max), rho, plot_seen_points=True, ax=ax)
    assert ftnss >= threshold, f"Threshold is {threshold}, while initial fitness is only {ftnss}"
    # move space in the direction there fitness stays highest (supposes convex shaped visibility)
    fit_dict = {'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0, 'z_max': 0}

    while fitness1(cameras, (x_min, x_max, y_min, y_max, z_min, z_max), rho)>threshold:
        fit_dict['x_min'] = fitness1(cameras, (x_min - lambd*(x_max-x_min), x_max, y_min, y_max, z_min, z_max), rho)
        fit_dict['x_max'] = fitness1(cameras, (x_min, x_max+lambd*(x_max-x_min), y_min, y_max, z_min, z_max), rho)
        fit_dict['y_min'] = fitness1(cameras, (x_min, x_max, y_min - lambd*(y_max-y_min), y_max, z_min, z_max), rho)
        fit_dict['y_max'] = fitness1(cameras, (x_min, x_max, y_min, y_max+lambd*(y_max-y_min), z_min, z_max), rho)
        fit_dict['z_max'] = fitness1(cameras, (x_min, x_max, y_min, y_max, z_min, z_max+lambd*(z_max-z_min)), rho)
        # lazy max search
        key, val = '', 0
        for k in fit_dict:
            if fit_dict[k] > val:
                val = fit_dict[k]
                key = k
        if   key == 'x_min': x_min -= lambd*(x_max-x_min)
        elif key == 'x_max': x_max += lambd*(x_max-x_min)
        elif key == 'y_min': y_min -= lambd*(y_max-y_min)
        elif key == 'y_max': y_max += lambd*(y_max-y_min)
        elif key == 'z_min': z_min -= lambd*(z_max-z_min)
        elif key == 'z_max': z_max += lambd*(z_max-z_min)
        else: raise Exception("Something is not right with the keys")
    return x_min, x_max, y_min, y_max, z_min, z_max

def symmetric_params2regular_params(camera_params:np.ndarray, symmetry:str) -> np.ndarray:
    if symmetry=='square':
        result = np.zeros(4*len(camera_params))
        for i in range(0, len(camera_params), 4):
            result[4*i+0:4*i+4] = camera_params[i:i+4] * np.array((1, 1, 1, 1), dtype=int)
            result[4*i+4:4*i+8] = camera_params[i:i+4] * np.array((1, -1, 1, -1), dtype=int)
            result[4*i+8:4*i+12] = camera_params[i:i+4] * np.array((-1, 1, -1, 1), dtype=int)
            result[4*i+12:4*i+16] = camera_params[i:i+4] * np.array((-1, -1, -1, -1), dtype=int)
        return result
    elif symmetry=='circle':
        result = np.zeros(2*len(camera_params))
        for i in range(0, len(camera_params), 4):
            result[2*i+0:2*i+4] = camera_params[i:i+4] * np.array((1, 1, 1, 1), dtype=int)
            result[2*i+4:2*i+8] = camera_params[i:i+4] * np.array((-1, -1, -1, -1), dtype=int)
        return result
    else:
        raise Exception(f"Incorrect symmetry parameter: {symmetry}")


class Camera:
    '''
    Class describing a camera, positioned arbitrarily with a predifined height.

    Attributes:
        coords (tuple): (x, y) coordinates of the camera
        pitch (float): pitch angle of the camera in degrees
        yaw (float): yaw angle of the camera in degrees
    '''
    def __init__(self, coords:tuple, pitch:float, yaw:float) -> None:
        assert type(coords)==tuple and len(coords)==2, "Wrong format for coords"
        self.pitch = pitch
        self.yaw = yaw
        # calculate and store vertices in a dictionary
        self.vertices_d = {}
        self.vertices_d['E'] = np.array([coords[0], coords[1], HEIGHT])
        self.vertices_d['A'] = np.array((self.vertices_d['E'][0] - HEIGHT*np.tan(FOV_h/2 - self.pitch), self.vertices_d['E'][1] - HEIGHT*np.tan(FOV_v/2 - self.yaw), 0))
        self.vertices_d['B'] = np.array((self.vertices_d['E'][0] + HEIGHT*np.tan(FOV_h/2 + self.pitch), self.vertices_d['E'][1] - HEIGHT*np.tan(FOV_v/2 - self.yaw), 0))
        self.vertices_d['C'] = np.array((self.vertices_d['E'][0] - HEIGHT*np.tan(FOV_h/2 - self.pitch), self.vertices_d['E'][1] + HEIGHT*np.tan(FOV_v/2 + self.yaw), 0))
        self.vertices_d['D'] = np.array((self.vertices_d['E'][0] + HEIGHT*np.tan(FOV_h/2 + self.pitch), self.vertices_d['E'][1] + HEIGHT*np.tan(FOV_v/2 + self.yaw), 0))
        # store vertices in a matrix
        self.vertices_m = np.zeros((5, 3))
        for idx, key in enumerate(self.vertices_d):
            self.vertices_m[idx, :] = self.vertices_d[key]
        # calculate and store planes in a dictionary
        self.planes_d = {}
        self.planes_d['ABE'] = coefficients_from_points(self.vertices_d['A'], self.vertices_d['B'], self.vertices_d['E'])
        self.planes_d['BDE'] = coefficients_from_points(self.vertices_d['B'], self.vertices_d['D'], self.vertices_d['E'])
        self.planes_d['CDE'] = coefficients_from_points(self.vertices_d['C'], self.vertices_d['D'], self.vertices_d['E'])
        self.planes_d['ACE'] = coefficients_from_points(self.vertices_d['A'], self.vertices_d['C'], self.vertices_d['E'])
        self.planes_d['ABC'] = coefficients_from_points(self.vertices_d['A'], self.vertices_d['B'], self.vertices_d['C'])

    def plot_vertices(self, ax, color='blue', alpha=1.0) -> None:
        ax.scatter(self.vertices_m[:, 0], self.vertices_m[:, 1], self.vertices_m[:, 2], color=color, alpha=alpha)
        for idx, key in enumerate(self.vertices_d):
            ax.text(self.vertices_m[idx, 0], self.vertices_m[idx, 1], self.vertices_m[idx, 2], key)

    def plot_plane(self, plane:str, ax, alpha=0.2) -> None:
        assert type(plane)==str and len(plane)==3, "Wrong format for plane"
        assert 'E' in plane, "Plane must contain the camera"
        xx, yy, zz = plane_from_points(self.vertices_d[plane[0]], self.vertices_d[plane[1]], self.vertices_d[plane[2]])
        ax.plot_surface(xx, yy, zz, alpha=alpha)
        return

    def peekaboo1(self, points:np.ndarray) -> np.ndarray:
        '''
        Given a set of points (n, 3), returns a boolean array (n, ) indicating whether the points are visible from the camera.

        Attributes:
            points (np.ndarray): array of points to be checked

        Returns:
            np.ndarray: boolean array indicating whether the points are visible from the camera
        '''
        assert len(points.shape)==2 and points.shape[1]==3, 'Badly shaped input'
            # extract extreme values
        x_min, x_max, y_min, y_max = np.min(self.vertices_m[:, 0]), np.max(self.vertices_m[:, 0]), np.min(self.vertices_m[:, 1]), np.max(self.vertices_m[:, 1])
            # construct new extreme values as a funciton of z
        z = points[:, 2]
        x_min_z = (self.vertices_d['E'][0] - x_min)/HEIGHT * z + x_min
        x_max_z = (self.vertices_d['E'][0] - x_max)/HEIGHT * z + x_max
        y_min_z = (self.vertices_d['E'][1] - y_min)/HEIGHT * z + y_min
        y_max_z = (self.vertices_d['E'][1] - y_max)/HEIGHT * z + y_max

        return np.logical_and(np.logical_and(x_min_z < points[:, 0], points[:, 0] < x_max_z), np.logical_and(y_min_z < points[:, 1], points[:, 1] < y_max_z))
    
    def peekaboo2(self, points:np.ndarray) -> np.ndarray:
        '''
        Given a set of points (n, 3), returns a boolean (n, ) array indicating whether the points are visible from the camera.

        Attributes:
            points (np.ndarray): array of points to be checked

        Returns:
            np.ndarray: boolean array indicating whether the points are visible from the camera
        '''
        assert len(points.shape)==2 and points.shape[1]==3, f'Badly shaped input ({points.shape} instead of (n, 3))'
        theta = self.planes_d['ABE']
        result = theta[0]*points[:, 0] + theta[1]*points[:, 1] + theta[2]*points[:, 2] + theta[3] < 0
        theta = self.planes_d['BDE']
        result = np.logical_and(result, theta[0]*points[:, 0] + theta[1]*points[:, 1] + theta[2]*points[:, 2] + theta[3] < 0)
        theta = self.planes_d['CDE']
        result = np.logical_and(result, theta[0]*points[:, 0] + theta[1]*points[:, 1] + theta[2]*points[:, 2] + theta[3] > 0)
        theta = self.planes_d['ACE']
        result = np.logical_and(result, theta[0]*points[:, 0] + theta[1]*points[:, 1] + theta[2]*points[:, 2] + theta[3] > 0)
        return np.logical_and(result, points[:, 2]>=0)

def generate_random_cameras(n:int, x_range:tuple, y_range:tuple) -> np.ndarray:
    params = np.random.rand(n, 4)
    params[:, 0] = (x_range[1]-x_range[0])*params[:, 0]+x_range[0] #scaling in x direction
    params[:, 1] = (y_range[1]-y_range[0])*params[:, 1]+y_range[0] #scaling in y direction
    params[:, 2:] = np.pi*params[:, 2:]-np.pi/2 #scaling the angles
    return params.reshape(-1)

# not used, there are examples for how to save and load multiple arrays
def random_search_starting_pos(n_iter:int, n_cameras:int, axes_limits:tuple, weighing_dict:dict, symmetric=False, read_start=False, verbose=False) -> None :
    np.random.seed(int(time()))
    x_range, y_range = (axes_limits[0], axes_limits[1]), (axes_limits[2], axes_limits[3])
    if read_start:
        _ = np.load('init_mat.npz')
        cams_init_mat = np.hstack((_['fitness'].reshape((-1, 1)), _['mat']))
        if verbose: print(f"Starting with fintesses {_['fitness']}")
    else:
        cams_init_mat = np.zeros((5, 1+4*n_cameras))
        for i in range(5):
            cams_init_mat[i, 1:] = generate_random_cameras(n_cameras, x_range, y_range, symmetric)
            cams_init_mat[i, 0] = fitness1(cams_init_mat[i, 1:], axes_limits, 4, weighing_dict)
        cams_init_mat = cams_init_mat[np.argsort(cams_init_mat[:, 0]), :]

    for _ in range(n_iter):
        cams_vec = generate_random_cameras(n_cameras, x_range, y_range, symmetric)
        fitness = fitness1(cams_vec, axes_limits, 4, weighing_dict)
        if fitness>cams_init_mat[0, 0]:
            cams_init_mat[0, 0] = fitness
            cams_init_mat[0, 1:] = cams_vec
            cams_init_mat = cams_init_mat[np.argsort(cams_init_mat[:, 0]), :]
    _ = len(weighing_dict)
    weight_keys = np.zeros(_, dtype=str)
    weight_vals = np.zeros(_)
    for idx, key in enumerate(weighing_dict):
        weight_keys[idx] = key
        weight_vals[idx] = weighing_dict[key]
    np.savez_compressed('init_mat.npz', mat=cams_init_mat[:, 1:], fitness=cams_init_mat[:, 0], weight_keys=weight_keys, weight_vals=weight_vals)
    if verbose: print(f"Reached fitnesses {cams_init_mat[:, 0]}")
    return
# not used, same
def load_starting_pos(return_fitness=False, return_weights=False):
    '''Return val: (matrix of 5 best random result (5 by 4*n_cameras), fitness(len=5; optional), weights_dictionary(optional))'''
    container = np.load('init_mat.npz')
    result = [container['mat']]
    if return_fitness: result.append(container['fitness'])
    if return_weights:
        weights = {}
        for idx, key in enumerate(container['weight_keys']):
            weights[key] = container['weight_vals'][idx]
        result.append(weights)
    return result

def go_through_results(path, symmetry=None) -> None:
    w = {'distance_from_origin': 0., 'stay_within_range': 0., 'spread': 0., 'soft_convexity': 0., 'hard_convexity': 0.}
    keys_ordered = ['distance_from_origin', 'stay_within_range', 'spread', 'soft_convexity', 'hard_convexity']
    with os.scandir(path) as entries:
        l = [entry.name for entry in entries if entry.is_file()]
    for idx, filename in enumerate(l):
        # construct weights from filename
        n = filename[:-4]
        f_l = [float(i) for i in n.split(sep='_')]
        for key, val in zip(keys_ordered, f_l):
            w[key] = val
        # calculate fitness
        vec = symmetric_params2regular_params(np.load(path+filename), symmetry) if symmetry else np.load(path+filename)
        print(f'Score: {fitness1(vec, ALL_RANGE, 4, w, verbose=True):.2e}')
        # plot seen points
        fig = plt.figure(f"{1+idx}")
        ax = fig.add_subplot(projection='3d')
        plot_seen_points(vec, ALL_RANGE, 4, ax)
        # show cameras
        cameras = []
        for i in range(0, len(vec), 4):
            cameras.append(Camera((vec[i], vec[i+1]), vec[i+2], vec[i+3]))
        for c in cameras:
            c.plot_vertices(ax, color='blue')
        ax.set_xlim3d(X_RANGE[0], X_RANGE[1])
        ax.set_ylim3d(Y_RANGE[0], Y_RANGE[1])
        ax.set_zlim3d(0, HEIGHT)
        plt.show()
    return

cam = Camera((-0.5, -0.8), np.deg2rad(1.77), np.deg2rad(11.8))
cameras = [Camera((-0.5, -0.8), np.deg2rad(21.77), np.deg2rad(11.8)),
           Camera((-0.2, -0.4), np.deg2rad(12.77), np.deg2rad(1.8)),
           Camera((0.5, 0.8), np.deg2rad(-21.77), np.deg2rad(-37.8))]

manual_vec = symmetric_params2regular_params(np.array((-2., -2.8, np.deg2rad(21.77), np.deg2rad(34.8))), 'square')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# plot axes
if True:
    scaling = 10
    ax.plot([-scaling, scaling], [0, 0], [0, 0], color='red')
    ax.plot([0, 0], [-scaling, scaling], [0, 0], color='green')
    ax.plot([0, 0], [0, 0], [-scaling, scaling], color='blue')
    ax.xaxis.label.set_color('red')
    ax.yaxis.label.set_color('green')
    ax.zaxis.label.set_color('blue')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    ax.set_xlim3d(-X_LEN/2, X_LEN/2)
    ax.set_ylim3d(-Y_LEN/2, Y_LEN/2)
    ax.set_zlim3d(-0.2, HEIGHT)
    ax.set_title('Pyramid')
# plot the pyramid
if False:
    ax.scatter(cam.vertices_m[:, 0], cam.vertices_m[:, 1], cam.vertices_m[:, 2], color='orange')
    for idx, key in enumerate(cam.vertices_d):
        ax.text(cam.vertices_m[idx, 0], cam.vertices_m[idx, 1], cam.vertices_m[idx, 2], key)
# plot point cloud around the pyramid
if False:
    N = 500
    point_cloud_m = np.hstack((
        np.random.rand(N, 1)*abs(cam.vertices_d['A'][0]-cam.vertices_d['B'][0]) + min(cam.vertices_d['A'][0], cam.vertices_d['B'][0]), np.random.rand(N, 1)*abs(cam.vertices_d['A'][1]-cam.vertices_d['C'][1]) + min(cam.vertices_d['A'][1], cam.vertices_d['C'][1]), np.random.rand(N, 1)*abs(cam.vertices_d['A'][2]-cam.vertices_d['E'][2]) + min(cam.vertices_d['A'][2], cam.vertices_d['E'][2])))
    mask = cam.peekaboo2(point_cloud_m)
    ax.scatter(point_cloud_m[:, 0][mask], point_cloud_m[:, 1][mask], point_cloud_m[:, 2][mask], color='black', alpha=0.3)
# test fitness1 function
if False:
    print(f"Fintess: {100*fitness1(cameras, axes_limits=(-0.4, 0.4, -0.75, -0.25, 0.0, 0.4), plot_seen_points=True, ax=ax):.3f}%")
    colors = ['red', 'green', 'blue']
    for idx, c in enumerate(cameras):
        c.plot_vertices(ax, colors[idx])
# test fitness2 function
if False:
    axes_limit_initial_guess = (-0.4, 0.4, -0.75, -0.25, 0.0, 0.4)
    new_space = fitness2(cameras, axes_limit_initial_guess)
    print(fitness1(cameras, new_space))

'''Implement CMA-ES'''
import cma
X_RANGE = (-X_LEN/2, X_LEN/2)
Y_RANGE = (-Y_LEN/2, Y_LEN/2)
Z_RANGE = (0, HEIGHT)
ALL_RANGE = X_RANGE+Y_RANGE+Z_RANGE

N_CAMERAS = 7
_ = [0, None, 'circle', 'square']
i = 3
SYMMETRY = _[i]
if SYMMETRY: N_CAMERAS = N_CAMERAS // (2 if SYMMETRY=='circle' else 4)
print(f"n_cams = {N_CAMERAS}")
RHO = 4
WEIGHTS = {'distance_from_origin': float(-i+1), 'stay_within_range': -0.8, 'spread': -1.0, 'soft_convexity': -1., 'hard_convexity': -0.7}
STR_OUT = "results/" +\
          "{:.{}e}".format(WEIGHTS['distance_from_origin'], 3) + "_" +\
          "{:.{}e}".format(WEIGHTS['stay_within_range'], 3) + "_" +\
          "{:.{}e}".format(WEIGHTS['spread'], 3) + "_" +\
          "{:.{}e}".format(WEIGHTS['soft_convexity'], 3) + "_" +\
          "{:.{}e}".format(WEIGHTS['hard_convexity'], 3) + ".npy"

# create a numpy vector of Camera objects
camera_vector = generate_random_cameras(N_CAMERAS, X_RANGE, Y_RANGE)
# camera_vector = np.load('init_mat.npy')[3, 1:]
# fitness_symmetric(camera_vector, ALL_RANGE, 4, WEIGHTS, verbose=True)

sigma = 0.5 * 1/4*(X_RANGE[1]-X_RANGE[0]) #"``sigma0`` should be about 1/4th of the search domain width"
# KEEP EXPERIMENTING WITH THE SIGMA PARAMETER
args = (ALL_RANGE, RHO, WEIGHTS, SYMMETRY, True, False)

# generate a result
# xbest, es = cma.fmin2(fitness1, x0=camera_vector, sigma0=sigma, args=args)
# np.save(STR_OUT, xbest)
# exit("Done!")

# automatically check results
# go_through_results('results/', SYMMETRY)
# exit()

# manually check results
xbest = symmetric_params2regular_params(np.load(f'results/-{i}.000e+00_-8.000e-01_-1.000e+00_-1.000e+00_-7.000e-01.npy'), SYMMETRY) if SYMMETRY else np.load('results/-1.000e+00_-8.000e-01_-1.000e+00_-1.000e+00_-7.000e-01.npy')
# xbest = np.load(f'results/manual/no_weights_7_cams_no_symmetry_90percent.npy')
# print('n_cameras: ', len(xbest)//4)

print(f'Score: {fitness1(xbest, ALL_RANGE, RHO, WEIGHTS, verbose=True, minimize=True):.2e}')

# plot_seen_points(xbest, ALL_RANGE, 4, ax)


#visualize cameras:
cameras = []
for i in range(0, len(xbest), 4):
    cameras.append(Camera((xbest[i], xbest[i+1]), xbest[i+2], xbest[i+3]))
for c in cameras:
    c.plot_vertices(ax, color='blue')


plt.show()
print("Done!")