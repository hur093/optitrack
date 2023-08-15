'''
install cma: https://pypi.org/project/cma/
cma docs: https://cma-es.github.io/apidocs-pycma/


'''
import os
from time import time
from typing import Tuple
import numpy as np, matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
# import cma
np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)

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
        self.vertices_d['A'] = np.array((self.vertices_d['E'][0] - HEIGHT*np.tan(FOV_H/2 - self.pitch), self.vertices_d['E'][1] - HEIGHT*np.tan(FOV_V/2 - self.yaw), 0))
        self.vertices_d['B'] = np.array((self.vertices_d['E'][0] + HEIGHT*np.tan(FOV_H/2 + self.pitch), self.vertices_d['E'][1] - HEIGHT*np.tan(FOV_V/2 - self.yaw), 0))
        self.vertices_d['C'] = np.array((self.vertices_d['E'][0] - HEIGHT*np.tan(FOV_H/2 - self.pitch), self.vertices_d['E'][1] + HEIGHT*np.tan(FOV_V/2 + self.yaw), 0))
        self.vertices_d['D'] = np.array((self.vertices_d['E'][0] + HEIGHT*np.tan(FOV_H/2 + self.pitch), self.vertices_d['E'][1] + HEIGHT*np.tan(FOV_V/2 + self.yaw), 0))
        # store vertices in a matrix
        self.vertices_m = np.zeros((5, 3))
        for idx, key in enumerate(self.vertices_d):
            self.vertices_m[idx, :] = self.vertices_d[key]
        # calculate and store planes in a dictionary
        self.planes_d = {}
        self.planes_d['ABE'] = self._coefficients_from_points(self.vertices_d['A'], self.vertices_d['B'], self.vertices_d['E'])
        self.planes_d['BDE'] = self._coefficients_from_points(self.vertices_d['B'], self.vertices_d['D'], self.vertices_d['E'])
        self.planes_d['CDE'] = self._coefficients_from_points(self.vertices_d['C'], self.vertices_d['D'], self.vertices_d['E'])
        self.planes_d['ACE'] = self._coefficients_from_points(self.vertices_d['A'], self.vertices_d['C'], self.vertices_d['E'])
        self.planes_d['ABC'] = self._coefficients_from_points(self.vertices_d['A'], self.vertices_d['B'], self.vertices_d['C'])
    def _coefficients_from_points(self, p1:np.ndarray, p2=None, p3=None, verbose=False) -> np.ndarray:
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
    def _plane_from_points(p1, p2, p3):
        pq, pr = p2 - p1, p3 - p1
        n = np.cross(pq, pr)

        # set boundaries
        X = np.vstack((p1, p2, p3))
        x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
        y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])

        xx, yy = np.meshgrid(np.linspace((0.8*x_min>0 + 1.25*x_min<0)*x_min, (0.8*x_max<0 + 1.25*x_max>0)*x_max, 8), np.linspace((0.8*y_min>0 + 1.25*y_min<0)*y_min, (0.8*y_max<0 + 1.25*y_max>0)*y_max, 8))
        
        zz = (n[0]*p1[0] + n[1]*p1[1] + n[2]*p1[2] - n[0]*xx - n[1]*yy) / n[2]
        return xx, yy, zz
    
    def plot_vertices(self, ax, color='blue', alpha=1.0) -> None:
        ax.scatter(self.vertices_m[:, 0], self.vertices_m[:, 1], self.vertices_m[:, 2], color=color, alpha=alpha)
        for idx, key in enumerate(self.vertices_d):
            ax.text(self.vertices_m[idx, 0], self.vertices_m[idx, 1], self.vertices_m[idx, 2], key)

    def plot_plane(self, plane:str, ax, alpha=0.2) -> None:
        assert type(plane)==str and len(plane)==3, "Wrong format for plane"
        assert 'E' in plane, "Plane must contain the camera"
        xx, yy, zz = self._plane_from_points(self.vertices_d[plane[0]], self.vertices_d[plane[1]], self.vertices_d[plane[2]])
        ax.plot_surface(xx, yy, zz, alpha=alpha)
        return
    
    def peekaboo(self, points:np.ndarray) -> np.ndarray:
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


class CameraOptimizer:
    def __init__(self, fov:Tuple[float], dims:Tuple[float], n_cameras:int, cam_r:int=0, verbose:bool=False) -> None:
        assert len(fov)==2, "fov must be a tuple of length 2: (FOV_horizontal, FOV_vertical), given in radiand"
        assert len(dims)==3, "dims must be a tuple of length 3: (length, width, height), given in meters"
        assert cam_r >= 0, "cam_r must be a positive integer"
        self.FOV_H, self.FOV_V = fov
        self.X_RANGE, self.Y_RANGE = (-dims[0]/2, dims[0]/2), (-dims[1]/2, dims[1]/2)
        self.Z_RANGE = (0, dims[2])
        self.ALL_RANGE = self.X_RANGE+self.Y_RANGE+self.Z_RANGE
        self.CAMERA_RADIUS = cam_r
        self.verbose = verbose
        self.N_CAMERAS = n_cameras
        self.cameras = None
        self.set_random_cameras()
        return
    def set_random_cameras(self) -> None:
        params = np.random.rand(self.N_CAMERAS, 4)
        params[:, 0] = (self.X_RANGE[1]-self.X_RANGE[0])*params[:, 0]+self.X_RANGE[0] #scaling in x direction
        params[:, 1] = (self.Y_RANGE[1]-self.Y_RANGE[0])*params[:, 1]+self.Y_RANGE[0] #scaling in y direction
        params[:, 2:] = np.pi*params[:, 2:]-np.pi/2 #scaling the angles
        self.cameras = params.reshape(-1)
        return
    
    def fitness(self, rho=4.0, verbose=False) -> float:
        """
        KNOWN PROBLEM: VISUALISING NOT-SEEN POINTS CAN BECOME A MESS
        One possible fitness function for measuring the quality of the placement of the cameras. This was relatively easy to implement, however the measured space is of constant size (so maximum space needs to be manually checked for any number of cameras AND if the space is too large, there is a lot of unnecessary calculations) and convexity of the seen space is not taken into account.
        NOTE: fitness should return np.NaN when cameras are too close

        self.cameras: 1 dim list describing the cameras. Must be divisible by the number of parameters necessary for one camera
        axes_limits: a tuple of length 6 or 8: (x_min, x_max, y_min, y_max, z_min, z_max) or (x_min, x_max, y_min, y_max)
        rho: number of points in one direction within a unit of distance
        minimize: bool, generally only true when passing the function to CMA-ES trainer/solver
        """
        assert len(self.cameras)%4==0
        assert len(self.ALL_RANGE)==6, "self.ALL_RANGE must be a tuple with length 6"
        x_min, x_max, y_min, y_max, z_min, z_max = self.ALL_RANGE
        cameras = []
        for i in range(0, len(self.cameras), 4):# build cameras
            cameras.append(Camera((self.cameras[i+0], self.cameras[i+1]), self.cameras[i+2], self.cameras[i+3]))
        n_cams = len(cameras)
        if self.CAMERA_RADIUS>0:# enforce that cameras are not to be too close to one another
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
            mask_mat[cam_idx, :] = cam.peekaboo(test_point_set)
        mask_seen_by_three = np.zeros(n_points, dtype=bool)
        for idx in range(n_points):
            mask_seen_by_three[idx] = np.sum(mask_mat[:, idx])>=3 #mask_seen_by_three is a n_points long vector of bools, where a True value in the ith position means it was seen by at least 3 cameras
        result = np.sum(mask_seen_by_three)/n_points
        if verbose: print(f'The cameras see {100*result:.1f}% of points')
        return 1 - result

FOV_H = np.deg2rad(56)
FOV_V = np.deg2rad(46)
HEIGHT = 3. #meters (measured: 295 cm)
X_LEN, Y_LEN = 5., 7.5 #meters
CAM_SIZE = 0.3 # upper limit measured is about 30 cm

optim = CameraOptimizer((FOV_H, FOV_V), (X_LEN, Y_LEN, HEIGHT), 3, CAM_SIZE)
optim.set_random_cameras()
optim.fitness(verbose=True)