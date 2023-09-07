'''
Optimize the placement of cameras for a motion-capture camera system.

install cma: https://pypi.org/project/cma/
cma docs: https://cma-es.github.io/apidocs-pycma/
'''
import os
from time import time
from typing import Tuple
import numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, Delaunay
import cma

np.set_printoptions(precision=3, suppress=True)

EPS = 0.015

def symmetric_params2regular_params(camera_params:np.ndarray, symmetry:str) -> np.ndarray:
    """
    Apply symmetry to the cameras given by camera_params.

    Parameters:
    ----------
        camera_params : np.ndarray
            array of cameras
        symmetry : str
            type of symmetry to be applied. Possible values: 'square', 'point'

    Returns:
    -------
        np.ndarray
            array of cameras with symmetry applied
    """
    if symmetry=='square':
        result = np.zeros(4*len(camera_params))
        for i in range(0, len(camera_params), 4):
            result[4*i+0:4*i+4] = camera_params[i:i+4] * np.array((1, 1, 1, 1), dtype=int)
            result[4*i+4:4*i+8] = camera_params[i:i+4] * np.array((1, -1, 1, -1), dtype=int)
            result[4*i+8:4*i+12] = camera_params[i:i+4] * np.array((-1, 1, -1, 1), dtype=int)
            result[4*i+12:4*i+16] = camera_params[i:i+4] * np.array((-1, -1, -1, -1), dtype=int)
        return result
    elif symmetry=='point':
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
    ----------
        coords : Tuple[float], length=3
            x, y, z coordinates of the camera
        tilt : Tuple[float], length=2
            pitch and yaw angles of the camera given in radians
        fov : Tuple[float], length=2
            horizontal and vertical field of view of the camera given in radians
    '''
    def __init__(self, coords:Tuple[float], tilt:Tuple[float], fov:Tuple[float]) -> None:
        assert type(coords)==tuple and len(coords)==3, "Wrong format for coords"
        assert type(tilt)==tuple and len(tilt)==2, "Wrong format for tilt"
        assert type(fov)==tuple and len(fov)==2, "Wrong format for fov"
        self.fov_h, self.fov_v = fov[0], fov[1]
        self.pitch, self.yaw = tilt[0], tilt[1]
        _ = (1-EPS) * np.pi
        assert 2*np.abs(self.pitch) < _-self.fov_h, "Pitch out of range"
        assert 2*np.abs(self.yaw) < _-self.fov_v, "Yaw out of range"
        # calculate and store vertices in a dictionary
        self.vertices_d = {}
        self.vertices_d['E'] = np.array([coords[0], coords[1], coords[2]])
        self.vertices_d['A'] = np.array((self.vertices_d['E'][0] - coords[2]*np.tan(self.fov_h/2 - self.pitch),
                                         self.vertices_d['E'][1] - coords[2]*np.tan(self.fov_v/2 - self.yaw), 0))
        self.vertices_d['B'] = np.array((self.vertices_d['E'][0] + coords[2]*np.tan(self.fov_h/2 + self.pitch),
                                         self.vertices_d['E'][1] - coords[2]*np.tan(self.fov_v/2 - self.yaw), 0))
        self.vertices_d['C'] = np.array((self.vertices_d['E'][0] - coords[2]*np.tan(self.fov_h/2 - self.pitch),
                                         self.vertices_d['E'][1] + coords[2]*np.tan(self.fov_v/2 + self.yaw), 0))
        self.vertices_d['D'] = np.array((self.vertices_d['E'][0] + coords[2]*np.tan(self.fov_h/2 + self.pitch),
                                         self.vertices_d['E'][1] + coords[2]*np.tan(self.fov_v/2 + self.yaw), 0))
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
    def _plane_from_points(self, p1, p2, p3):
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
        '''
        Plots the vertices of the camera in 3D space.

        Parameters:
        ----------
            ax : matplotlib.axes._subplots.Axes3DSubplot
                axes to plot on
            color : str, optional
                color of the vertices, makes it easier to distinguish between multiple plotted cameras. By default 'blue'
            alpha : float, optional
                transparency of the vertices, by default 1.0
        
        Returns:
        -------
            None
        '''
        ax.scatter(self.vertices_m[:, 0], self.vertices_m[:, 1], self.vertices_m[:, 2], color=color, alpha=alpha)
        for idx, key in enumerate(self.vertices_d):
            ax.text(self.vertices_m[idx, 0], self.vertices_m[idx, 1], self.vertices_m[idx, 2], key)
        # plot triangles showing the direction of the camera
        # collect vectors for the triangles
        triangle_sidelength = 0.7
        a_vec = self.vertices_d['A'] - self.vertices_d['E']
        a_vec = a_vec / np.linalg.norm(a_vec)
        b_vec = self.vertices_d['B'] - self.vertices_d['E']
        b_vec = b_vec / np.linalg.norm(b_vec)
        c_vec = self.vertices_d['C'] - self.vertices_d['E']
        c_vec = c_vec / np.linalg.norm(c_vec)
        d_vec = self.vertices_d['D'] - self.vertices_d['E']
        d_vec = d_vec / np.linalg.norm(d_vec)
        # plot the triangles
        verts = np.zeros((4, 3, 3))
        for idx in range(4): verts[idx, 0, :] = self.vertices_d['E']
        verts[0, 1, :] = self.vertices_d['E'] + triangle_sidelength*a_vec
        verts[0, 2, :] = self.vertices_d['E'] + triangle_sidelength*b_vec
        verts[1, 1, :] = self.vertices_d['E'] + triangle_sidelength*a_vec
        verts[1, 2, :] = self.vertices_d['E'] + triangle_sidelength*c_vec
        verts[2, 1, :] = self.vertices_d['E'] + triangle_sidelength*d_vec
        verts[2, 2, :] = self.vertices_d['E'] + triangle_sidelength*b_vec
        verts[3, 1, :] = self.vertices_d['E'] + triangle_sidelength*d_vec
        verts[3, 2, :] = self.vertices_d['E'] + triangle_sidelength*c_vec
        ax.add_collection3d(Poly3DCollection(verts, alpha=0.5, linewidths=0.5, edgecolors='k'))
        return

    def _plot_plane(self, edges:str, ax, alpha=0.2) -> None:
        assert type(edges)==str and len(edges)==3, "Wrong format for edges"
        assert 'E' in edges, "Plane must contain the camera"
        xx, yy, zz = self._edges_from_points(self.vertices_d[edges[0]], self.vertices_d[edges[1]], self.vertices_d[edges[2]])
        ax.plot_surface(xx, yy, zz, alpha=alpha)
        return
    
    def plot_faces(self, ax, alpha=0.2) -> None:
        '''
        The vertices of a camera make a pyramid. This function plots the faces of the pyramid.

        Parameters:
        ----------
            ax : matplotlib.axes._subplots.Axes3DSubplot
                axes to plot on
            alpha : float, optional
                transparency of the faces, by default 0.2
        
        Returns:
        -------
            None
        '''
        verts = np.zeros((4, 3, 3))
        for idx in range(4): verts[idx, 0, :] = self.vertices_d['E']
        verts[0, 1, :] = self.vertices_d['A']
        verts[0, 2, :] = self.vertices_d['B']
        verts[1, 1, :] = self.vertices_d['B']
        verts[1, 2, :] = self.vertices_d['D']
        verts[2, 1, :] = self.vertices_d['C']
        verts[2, 2, :] = self.vertices_d['D']
        verts[3, 1, :] = self.vertices_d['A']
        verts[3, 2, :] = self.vertices_d['C']
        ax.add_collection3d(Poly3DCollection(verts, alpha=alpha, linewidths=0.5, edgecolors='k'))
        return
    
    def peekaboo(self, points:np.ndarray) -> np.ndarray:
        '''
        Given a set of points (of shape (n, 3)), returns a boolean (of shape (n, )) array indicating whether the points are visible from the camera.

        Parameters:
        ----------
            points : np.ndarray
                array of points to be checked

        Returns:
        -------
            np.ndarray
                boolean array indicating whether the points are visible from the camera
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
    '''
    Class for optimizing the placement of cameras in a given space.

    Attributes:
    ----------
        fov : Tuple[float], length=2
            horizontal and vertical field of view of the cameras given in radians
        dims : Tuple[float], length=3
            length, width and height of the space given in meters
        n_cameras : int
            number of cameras to be placed
        cam_r : float
            radius of the cameras given in meters
        verbose : bool
            whether to print out information during the optimization process (CURRENTLY DOESN'T DO MUCH)
    '''
    def __init__(self, fov:Tuple[float], dims:Tuple[float], n_cameras:int, cam_r:float=0., verbose:bool=False) -> None:
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
        # THE SIGMA PARAMETER IS CURRENTLY 'SET IN STONE', BUT KEEP ON EXPERIMENTING WITH ITS VALE!
            # general observation: larger sigma vales than the current one leads to less runtime and less accurate results
        # self.sigma0 = 0.25 * np.array([dims[0], dims[1], np.pi, np.pi])
        # self.sigma0 = 0.25 * np.max([dims[0], dims[1], np.pi])
        self.rho = None
        self.cameras = None
        return
    
    def help(self) -> None:
        '''
        Currently does nothing
        '''
        pass

    def set_random_cameras(self) -> None:
        '''
        Give a random initial placement of the cameras. The cameras are placed randomly, but the pitch and yaw angles are limited in a way that the cameras don't see anything above themselves.

        Parameters:
        ----------
            None

        Returns:
        -------
            None
        '''
        self.rho = None
        params = np.random.rand(self.N_CAMERAS, 4)
        params[:, 0] = (self.X_RANGE[1]-self.X_RANGE[0])*params[:, 0]+self.X_RANGE[0] #scaling in x direction
        params[:, 1] = (self.Y_RANGE[1]-self.Y_RANGE[0])*params[:, 1]+self.Y_RANGE[0] #scaling in y direction
        _ = (1-EPS) * np.pi
        angle_range_h =  (0.5*(self.FOV_H-_), 0.5*(_-self.FOV_H))
        angle_range_v =  (0.5*(self.FOV_V-_), 0.5*(_-self.FOV_V))
        params[:, 2] = (angle_range_h[1]-angle_range_h[0])*params[:, 2]+angle_range_h[0] #scaling the pitch angles
        params[:, 3] = (angle_range_v[1]-angle_range_v[0])*params[:, 3]+angle_range_v[0] #scaling the yaw angles
        self.cameras = params.reshape(-1)
        return
    
    def fitness(self, cameras_arr=None, rho=4.0, verbose=False) -> float:
        """
        One possible fitness function for measuring the quality of the placement of the cameras. The fitness is the percentage of points that are seen by at least 3 cameras.
        NOTE: fitness should return np.NaN when cameras are too close

        Parameters:
        ----------
            cameras_arr : np.ndarray, optional
                array of camera parameters. If None (default), it will use the cameras stored in the object.
            rho : float, optional
                density of the grid of points to be checked. Higher values lead to larger runtimes, while too low values may lead to an inaccurate estimation of the continous real-world space. By default 4.0
            verbose : bool, optional
                print the percentage of points seen by at least 3 cameras, or the reason for rejecting the setup given by camera_arr. By default False

        Returns:
        -------
            float
                fitness value. The lower the better. (so actually works more like a loss function)
        """
        if cameras_arr is None: cameras_arr = self.cameras
        assert len(cameras_arr)%4==0
        assert len(self.ALL_RANGE)==6, "self.ALL_RANGE must be a tuple with length 6"
        x_min, x_max, y_min, y_max, z_min, z_max = self.ALL_RANGE
        camera_objects = []
        for i in range(0, len(cameras_arr), 4):# build cameras
            camera_objects.append(Camera((cameras_arr[i+0], cameras_arr[i+1], z_max), (cameras_arr[i+2], cameras_arr[i+3]), (self.FOV_H, self.FOV_V)))
        n_cams = len(camera_objects)
        if self.CAMERA_RADIUS>0:# enforce that cameras are not to be too close to one another
            for i in range(n_cams-1):
                for j in range(i+1, n_cams):
                    if np.linalg.norm(camera_objects[i].vertices_d['E']-camera_objects[j].vertices_d['E']) < 2*self.CAMERA_RADIUS:
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
        for cam_idx, cam in enumerate(camera_objects):
            assert type(cam)==Camera, "All element of 'cameras' must be of type 'Camera'"
            mask_mat[cam_idx, :] = cam.peekaboo(test_point_set)
        mask_seen_by_three = np.zeros(n_points, dtype=bool)
        for idx in range(n_points):
            mask_seen_by_three[idx] = np.sum(mask_mat[:, idx])>=3 #mask_seen_by_three is a n_points long vector of bools, where a True value in the ith position means it was seen by at least 3 cameras
        result = np.sum(mask_seen_by_three)/n_points
        if verbose: print(f'The cameras see {100*result:.1f}% of points')
        return 1 - result

    def train(self, x0=None, rho=4.0, fitness_upper_limit=0.9) -> None:
        '''
        Use cma library to optimize the placement of the cameras.

        Parameters:
        ----------
            x0 : np.ndarray, optional
                initial placement of the cameras. If None (default), it will use the cameras stored in the object.
            rho : float, optional
                density of the grid of points to be checked. Higher values lead to larger runtimes, while too low values may lead to an inaccurate estimation of the continous real-world space. By default 4.0
            fitness_upper_limit : float, optional
                upper limit for the fitness value. If the fitness value is above this limit, the optimization will be repeated. By default 0.9
                
        Returns:
        -------
            None
        '''
        self.rho = rho
        if x0 is None: x0 = self.cameras
        sigma = 0.5 * 1/4*(self.X_RANGE[1]-self.X_RANGE[0]) #"``sigma0`` should be about 1/4th of the search domain width"
        args = (rho, False)
        _ = (1-EPS) * np.pi
        bounds = [self.N_CAMERAS*[self.X_RANGE[0], self.Y_RANGE[0], 0.5*(self.FOV_H-_), 0.5*(self.FOV_V-_)],
                  self.N_CAMERAS*[self.X_RANGE[1], self.Y_RANGE[1],  0.5*(_-self.FOV_H),  0.5*(_-self.FOV_V)]]
        self.cameras, es = cma.fmin2(self.fitness, x0, sigma, {'bounds': bounds}, args=args)
        while self.fitness()>fitness_upper_limit or self.fitness() is np.NaN:
            self.cameras, es = cma.fmin2(self.fitness, x0, sigma, {'bounds': bounds}, args=args)
        return

    def save(self, path:str) -> None:
        '''
        Save the current state of the object to a .npz file.

        Parameters:
        ----------
            path : str
                path to the file to be saved

        Returns:
        -------
            None
        '''
        metadata = np.array([self.FOV_H, self.FOV_V, self.X_RANGE[0], self.X_RANGE[1], self.Y_RANGE[0], self.Y_RANGE[1], self.Z_RANGE[0], self.Z_RANGE[1], self.N_CAMERAS, self.CAMERA_RADIUS, self.rho if self.rho else np.NaN])
        np.savez_compressed(path, cameras=self.cameras, metadata=metadata)
        return
    
    def load(self, path:str) -> None:
        '''
        Load the state of the object from a .npz file. Will overwrite every current attribute.

        Parameters:
        ----------
            path : str
                path to the file to be loaded

        Returns:
        -------
            None

        Notes:
        -----
            The easiest way to get a blank optimizer to load into is the blank_optimizer function.
        '''
        data = np.load(path)
        self.FOV_H, self.FOV_V = data['metadata'][0:2]
        self.X_RANGE, self.Y_RANGE, self.Z_RANGE = tuple(data['metadata'][2:4]), tuple(data['metadata'][4:6]), tuple(data['metadata'][6:8])
        self.ALL_RANGE = self.X_RANGE+self.Y_RANGE+self.Z_RANGE
        self.N_CAMERAS = int(data['metadata'][8])
        self.CAMERA_RADIUS = data['metadata'][9]
        self.rho = data['metadata'][10]
        self.cameras = data['cameras']
        return

    def plot_cameras(self, ax) -> None:
        '''
        Plot the vertices of the cameras in 3D space.

        Parameters:
        ----------
            ax : matplotlib.axes._subplots.Axes3DSubplot
                axes to plot on

        Returns:
        -------
            None
        '''
        cameras = []
        for i in range(0, 4*self.N_CAMERAS, 4): cameras.append(Camera((self.cameras[i+0], self.cameras[i+1], self.Z_RANGE[1]), (self.cameras[i+2], self.cameras[i+3]), (self.FOV_H, self.FOV_V)))
        for c in cameras:
            c.plot_vertices(ax, color='blue')
        return
    
    def plot_seen_points(self, ax, rho=4.0, color='black') -> None:
        '''
        Plot the points that are seen by at least 3 cameras.

        Parameters:
        ----------
            ax : matplotlib.axes._subplots.Axes3DSubplot
                axes to plot on
            rho : float, optional
                density of the grid of points to be plotted. By default 4.0
            color : str, optional
                color of the points, by default 'black'

        Returns:
        -------
            None
        '''
        cameras = []
        for i in range(0, 4*self.N_CAMERAS, 4): cameras.append(Camera((self.cameras[i+0], self.cameras[i+1], self.Z_RANGE[1]), (self.cameras[i+2], self.cameras[i+3]), (self.FOV_H, self.FOV_V)))

        x_min, x_max = self.X_RANGE
        y_min, y_max = self.Y_RANGE
        z_min, z_max = self.Z_RANGE
        xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, int(rho*(x_max-x_min))),
                                np.linspace(y_min, y_max, int(rho*(y_max-y_min))),
                                np.linspace(z_min, z_max, int(rho*(z_max-z_min))))
        n_points = np.prod(xx.shape)
        test_point_set = np.stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)), axis=1)

        mask_mat = np.zeros((self.N_CAMERAS, n_points), dtype=bool)
        for cam_idx, cam in enumerate(cameras):
            mask_mat[cam_idx, :] = cam.peekaboo(test_point_set)
        mask_seen_by_three = np.zeros(n_points, dtype=bool)
        for idx in range(n_points):
            mask_seen_by_three[idx] = np.sum(mask_mat[:, idx])>=3 #mask_seen_by_three is a n_points long vector of bools, where a True value in the ith position means it was seen by at least 3 cameras

        ax.scatter(test_point_set[mask_seen_by_three, 0], test_point_set[mask_seen_by_three, 1], test_point_set[mask_seen_by_three, 2], color=color, alpha=0.2)
        # ax.scatter(test_point_set[~mask_seen_by_three, 0], test_point_set[~mask_seen_by_three, 1], test_point_set[~mask_seen_by_three, 2], color='yellow', alpha=0.1) # too messy
        return

class WeightedOptimizer(CameraOptimizer):
    '''
    Child class of the CameraOptimizer. It can finetune the placement of cameras based on a set of criteria.

    Attributes:
    ----------
        weights : dict
            dictionary with possible keys: 'distance_from_origin', 'stay_within_range', 'spread', 'soft_convexity', 'hard_convexity'
        fov : Tuple[float], length=2
            horizontal and vertical field of view of the cameras given in radians
        dims : Tuple[float], length=3
            length, width and height of the space given in meters
        n_cameras : int
            number of cameras to be placed
        cam_r : float
            radius of the cameras given in meters
        verbose : bool
            whether to print out information during the optimization process (CURRENTLY DOESN'T DO MUCH)

    Notes:
    -----
        The help() function gives a short description of what the weights do.
    '''
    def __init__(self, weights, fov:Tuple[float], dims:Tuple[float], n_cameras:int, cam_r:float=0., verbose:bool=False) -> None:
        assert type(weights)==dict, "weights must be a dictionary with possible keys: 'distance_from_origin', 'stay_within_range', 'spread', 'soft_convexity', 'hard_convexity'.\nForm more info, call WeightedOptimizer.help()"
        self.weights = {}
        self.weights['distance_from_origin'] = weights.get('distance_from_origin', -1.0)
        self.weights['stay_within_range'] = weights.get('stay_within_range', -1.0)
        self.weights['spread'] = weights.get('spread', -1.0)
        self.weights['soft_convexity'] = weights.get('soft_convexity', -1.0)
        self.weights['hard_convexity'] = weights.get('hard_convexity', -1.0)
        self.hull_fail_counter = 0
        super().__init__(fov, dims, n_cameras, cam_r, verbose)
        return
    
    def help(self):
        '''
        Give information on what the weights do.
        '''
        print("The WeightedOptimizer class is an optimizer that can finetune the placement of cameras based on a set of criteria. It takes in a dictionary called 'weights' with possible keys: 'distance_from_origin', 'stay_within_range', 'spread', 'soft_convexity', 'hard_convexity'.\n'distance_from_origin': Points close to the origin have higher weights, centering the visible volume.\n'stay_within_range': How strictly must the cameras's vision stay within the specified space. 0.0 is the strictest value, discarding every solution where not all cameras's vision are within the specified ranges. A value of 1.0 means that the allowed range is double that of the specified both in x and y directions and in general a value of 'f' means a '1+f' multiplier.\n'spread': This parameter entices the solution to keep the cameras as far from one another as possible, in case the solver gets stuck in grouping the cameras very close to one another. WARNING: this feature breaks the 0:1 range of the fitness function.\n'soft_convexity': Entices the solver to generate a convex solution without rejecting any solutions.\n'hard_convexity': Entices the solver to generate a convex solution by rejecting all solutions not deemed convex enough.\n")
        return

    def fitness(self, cameras_arr=None, rho=4.0, verbose=False) -> float:
        '''
        One possible fitness function for measuring the quality of the placement of the cameras. The fitness is determined by a number of factors, which can be tweaked by the weights attribute.
        NOTE: fitness should return np.NaN when cameras are too close

        Parameters:
        ----------
            cameras_arr : np.ndarray, optional
                array of camera parameters. If None (default), it will use the cameras stored in the object.
            rho : float, optional
                density of the grid of points to be checked. Higher values lead to larger runtimes, while too low values may lead to an inaccurate estimation of the continous real-world space. By default 4.0
            verbose : bool, optional
                print the percentage of points seen by at least 3 cameras, or the reason for rejecting the setup given by camera_arr. By default False

        Returns:
        -------
            float
                fitness value. The lower the better. (so actually works more like a loss function)
        '''
        if cameras_arr is None: cameras_arr = self.cameras
        assert len(cameras_arr)%4==0
        assert len(self.ALL_RANGE)==6, "self.ALL_RANGE must be a tuple with length 6"
        x_min, x_max, y_min, y_max, z_min, z_max = self.ALL_RANGE
        camera_objects = []
        for i in range(0, len(cameras_arr), 4):# build cameras
            camera_objects.append(Camera((cameras_arr[i+0], cameras_arr[i+1], z_max), (cameras_arr[i+2], cameras_arr[i+3]), (self.FOV_H, self.FOV_V)))
        n_cams = len(camera_objects)
        if self.CAMERA_RADIUS>0:# enforce that cameras are not to be too close to one another
            for i in range(n_cams-1):
                for j in range(i+1, n_cams):
                    if np.linalg.norm(camera_objects[i].vertices_d['E']-camera_objects[j].vertices_d['E']) < 2*self.CAMERA_RADIUS:
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
        for cam_idx, cam in enumerate(camera_objects):
            assert type(cam)==Camera, "All element of 'cameras' must be of type 'Camera'"
            mask_mat[cam_idx, :] = cam.peekaboo(test_point_set)
        mask_seen_by_three = np.zeros(n_points, dtype=bool)
        for idx in range(n_points):
            mask_seen_by_three[idx] = np.sum(mask_mat[:, idx])>=3 #mask_seen_by_three is a n_points long vector of bools, where a True value in the ith position means it was seen by at least 3 cameras

        result = 0
        max_result = 0
        # weigh each point according to how far it is from the origin
        # a: how strongly this weighing should be considered within the point-by-point basis weights; b: how strongly should proximity to origin be favored
        if type(self.weights['distance_from_origin']) == float:
            if self.weights['distance_from_origin']>0:
                a = self.weights['distance_from_origin']
                b = 1
                weights = 1/(1 + b*np.linalg.norm(test_point_set, axis=1))
                result += a * np.sum(weights*mask_seen_by_three)/np.sum(weights) 
                max_result += a
        elif type(self.weights['distance_from_origin']) == tuple:
            if np.all(np.array(self.weights['distance_from_origin'])>0):
                assert len(self.weights['distance_from_origin'])==2, "If the distance form origin parameter is a tuple, it must be of length 2."
                a, b = self.weights['distance_from_origin']
                weights = 1/(1 + b*np.linalg.norm(test_point_set, axis=1))
                result += a * np.sum(weights*mask_seen_by_three)/np.sum(weights) 
                max_result += a
        else: raise Exception("Distance form origin parameter must be a float or a tuple of length 2.")

        # couldn't figure out a way to normalise spread, therefore this functionality will break the 0-1 range of the fitness funciton. To somehow make it still scale by the space, I divide by the x and y size of the space
        if self.weights['spread']>0:
            cam_positions = np.zeros((n_cams, 2))
            for i in range(n_cams):# build cameras
                cam_positions[i, 0] = cameras_arr[4*i]
                cam_positions[i, 1] = cameras_arr[4*i+1]
            centroid = np.mean(cam_positions, axis=0)
            distances = np.linalg.norm(np.stack((cam_positions[:, 0]-centroid[0], cam_positions[:, 1]-centroid[1])), axis=0)
            max_distance_sum = (x_max-x_min)*(y_max-y_min) #not really the maximum spread, but this value should at least scale with the space
            result += self.weights['spread'] * np.sum(distances)/max_distance_sum
            max_result += self.weights['spread']

        if self.weights['soft_convexity'] > 0 or self.weights['hard_convexity'] > 0:
            test_points_seen = test_point_set[mask_seen_by_three]
            try:
                hull = ConvexHull(test_points_seen)
            except:
                if self.hull_fail_counter == 0: print("Convex hull could not be computed.")
                if self.verbose and self.hull_fail_counter%1000 == 1000-1: print(f'Convex hull could not be computed for the {1+self.hull_fail_counter}th time')
                self.hull_fail_counter += 1
                return np.NaN
            triangulation = Delaunay(test_points_seen[hull.vertices])
            mask_inside_hull = triangulation.find_simplex(test_point_set) >= 0 #mask for points inside the hull
            n_points_in_hull_not_seen = np.sum(np.logical_and(mask_inside_hull, ~mask_seen_by_three))
            n_seen = np.sum(mask_seen_by_three)
            convexity_ratio = n_seen/(n_seen+n_points_in_hull_not_seen)
            assert 0 <= convexity_ratio <= 1, "Something is not right: convexity ratio out of range"
            # soft convexity
            if self.weights['soft_convexity'] > 0:
                result += self.weights['soft_convexity']*convexity_ratio
                max_result += self.weights['soft_convexity']
            # hard convexity
            if self.weights['hard_convexity'] > 0:
                assert self.weights['hard_convexity'] < 1, "hard_convexity parameter must be strictly less than 1.0 (ideally below 0.95)"
                if convexity_ratio < self.weights['hard_convexity']: return np.NaN #reject every solution that is below the hard limit

        if verbose: print(f'The cameras see {100*np.sum(mask_seen_by_three)/n_points:.1f}% of points')

        # every weghing parameter is turned off
        if max_result == 0:
            result = np.sum(mask_seen_by_three)/n_points
            max_result = 1
        
        return 1 - result/max_result

    def save(self, path:str) -> None:
        dfo_strength, dfo_curve = self.weights['distance_from_origin'] if type(self.weights['distance_from_origin'])==tuple else (self.weights['distance_from_origin'], 1.0)
        weights = np.array([dfo_strength, dfo_curve, self.weights['stay_within_range'], self.weights['spread'], self.weights['soft_convexity'], self.weights['hard_convexity']])
        metadata = np.array([self.FOV_H, self.FOV_V, self.X_RANGE[0], self.X_RANGE[1], self.Y_RANGE[0], self.Y_RANGE[1], self.Z_RANGE[0], self.Z_RANGE[1], self.N_CAMERAS, self.CAMERA_RADIUS, self.rho if self.rho else np.NaN, self.hull_fail_counter])
        np.savez_compressed(path, weights=weights, cameras=self.cameras, metadata=metadata)
        return

    def load(self, path:str) -> None:
        data = np.load(path)

        self.weights = {}
        self.weights['distance_from_origin'] = tuple(data['weights'][0:2])
        self.weights['stay_within_range'] = data['weights'][2]
        self.weights['spread'] = data['weights'][3]
        self.weights['soft_convexity'] = data['weights'][4]
        self.weights['hard_convexity'] = data['weights'][5]

        self.FOV_H, self.FOV_V = data['metadata'][0:2]
        self.X_RANGE, self.Y_RANGE, self.Z_RANGE = tuple(data['metadata'][2:4]), tuple(data['metadata'][4:6]), tuple(data['metadata'][6:8])
        self.ALL_RANGE = self.X_RANGE+self.Y_RANGE+self.Z_RANGE
        self.N_CAMERAS = int(data['metadata'][8])
        self.CAMERA_RADIUS = data['metadata'][9]
        self.rho = data['metadata'][10]
        self.hull_fail_counter = data['metadata'][11]

        self.cameras = data['cameras']
        return

class SymmetricOptimizer(CameraOptimizer):
    '''
    Child class of the CameraOptimizer, which places cameras in a symmetrical manner to reduce dimensionality.

    Attributes:
    ----------
        symmetry : str
            possible values: 'point', 'square'. The former uses point symmetry, halving the degree of freedom, while the latter uses the x and y axes to reduce the degree of freedom by a factor of 4.
        fov : Tuple[float], length=2
            horizontal and vertical field of view of the cameras given in radians
        dims : Tuple[float], length=3
            length, width and height of the space given in meters
        n_cameras : int
            number of cameras to be placed
        cam_r : float
            radius of the cameras given in meters
        verbose : bool
            whether to print out information during the optimization process (CURRENTLY DOESN'T DO MUCH)
    '''
    def __init__(self, symmetry, fov:Tuple[float], dims:Tuple[float], n_cameras:int, cam_r:float=0., verbose:bool=False) -> None:
        # symmetry
        assert type(symmetry)==str, "symmetry parameter must be a string"
        assert symmetry=='point' or symmetry=='square', "Possible values for symmetry parameter are 'point' and 'square'"
        self.symmetry = symmetry
        # n_cameras
        if symmetry=='point':
            if n_cameras%2 != 0:
                print(f"Number of cameras must be divisible by 2. Changing number of cameras to {n_cameras - n_cameras%2}")
            n_free_cams = n_cameras // 2
            n = n_free_cams*2
        if symmetry=='square':
            if n_cameras%4 != 0:
                print(f"Number of cameras must be divisible by 4. Changing number of cameras to {n_cameras - n_cameras%4}")
            n_free_cams = n_cameras // 4
            n = n_free_cams*4
        if n_free_cams==0: raise Exception("Number of cameras is too low for this symmetry parameter.")
        super(SymmetricOptimizer, self).__init__(fov, dims, n, cam_r, verbose)
        self.n_cameras_sym = n_free_cams
        # cameras
        # self.cameras initialised by super's init
        self.cameras_sym = None
        return
    
    def _update(self):
        '''
        Derive cameras from cameras_sym.
        '''
        assert len(self.cameras_sym) == 4*self.n_cameras_sym, "Bad size for symmetric cameras."
        self.cameras = symmetric_params2regular_params(self.cameras_sym, self.symmetry)
        return

    def set_random_cameras(self) -> None:
        super().set_random_cameras()
        self.cameras_sym = self.cameras[:4*self.n_cameras_sym]
        self._update()
        while self.fitness() is np.NaN:
            super().set_random_cameras()
            self.cameras_sym = self.cameras[:4*self.n_cameras_sym]
            self._update()
        return

    def train(self, rho=4.0, fitness_upper_limit=0.9) -> None:
        while self.fitness()>fitness_upper_limit or self.fitness() is np.NaN:
            super().train(self.cameras_sym, rho, fitness_upper_limit=1e3)
            self.cameras_sym = self.cameras
            self._update()
            print(self.fitness())
        return

    def save(self, path:str) -> None:
        metadata = np.array([self.FOV_H, self.FOV_V, self.X_RANGE[0], self.X_RANGE[1], self.Y_RANGE[0], self.Y_RANGE[1], self.Z_RANGE[0], self.Z_RANGE[1], self.N_CAMERAS, self.CAMERA_RADIUS, self.rho if self.rho else np.NaN])
        np.savez_compressed(path, symmetry=self.symmetry, cameras=self.cameras, metadata=metadata)
        return
    
    def load(self, path:str) -> None:
        data = np.load(path)
        self.symmetry = data['symmetry']
        self.FOV_H, self.FOV_V = data['metadata'][0:2]
        self.X_RANGE, self.Y_RANGE, self.Z_RANGE = tuple(data['metadata'][2:4]), tuple(data['metadata'][4:6]), tuple(data['metadata'][6:8])
        self.ALL_RANGE = self.X_RANGE+self.Y_RANGE+self.Z_RANGE
        self.N_CAMERAS = int(data['metadata'][8])
        self.CAMERA_RADIUS = data['metadata'][9]
        self.rho = data['metadata'][10]
        self.cameras = data['cameras']

        self.n_cameras_sym = self.N_CAMERAS // (2 if self.symmetry=='point' else 4)
        self.cameras_sym = self.cameras[np.array(self.n_cameras_sym*[4*[1]+4*[0]], dtype=bool).reshape(-1)] if self.symmetry=='point' else self.cameras[np.array(self.n_cameras_sym*[4*[1]+12*[0]], dtype=bool).reshape(-1)]
        assert np.array_equal(self.cameras, symmetric_params2regular_params(self.cameras_sym, self.symmetry)), "Error during loading"
        return

class WeightedSymmetricOptimizer(WeightedOptimizer):
    '''
    Child class of the WeightedOptimizer, which places cameras in a symmetrical manner to reduce dimensionality.

    Attributes:
    ----------
        weights : dict
            dictionary with possible keys: 'distance_from_origin', 'stay_within_range', 'spread', 'soft_convexity', 'hard_convexity'
        symmetry : str
            possible values: 'point', 'square'. The former uses point symmetry, halving the degree of freedom, while the latter uses the x and y axes to reduce the degree of freedom by a factor of 4.
        fov : Tuple[float], length=2
            horizontal and vertical field of view of the cameras given in radians
        dims : Tuple[float], length=3
            length, width and height of the space given in meters
        n_cameras : int
            number of cameras to be placed
        cam_r : float
            radius of the cameras given in meters
        verbose : bool
            whether to print out information during the optimization process (CURRENTLY DOESN'T DO MUCH)

    Notes:
    -----
        The help() function gives a short description of what the weights do.
    '''
    def __init__(self, weights, symmetry, fov:Tuple[float], dims:Tuple[float], n_cameras:int, cam_r:float=0., verbose:bool=False) -> None:
        # symmetry
        assert type(symmetry)==str, "symmetry parameter must be a string"
        assert symmetry=='point' or symmetry=='square', "Possible values for symmetry parameter are 'point' and 'square'"
        self.symmetry = symmetry
        # n_cameras
        if symmetry=='point':
            if n_cameras%2 != 0:
                print(f"Number of cameras must be divisible by 2. Changing number of cameras to {n_cameras - n_cameras%2}")
            n_free_cams = n_cameras // 2
            n = n_free_cams*2
        if symmetry=='square':
            if n_cameras%4 != 0:
                print(f"Number of cameras must be divisible by 4. Changing number of cameras to {n_cameras - n_cameras%4}")
            n_free_cams = n_cameras // 4
            n = n_free_cams*4
        if n_free_cams==0: raise Exception("Number of cameras is too low for this symmetry parameter.")
        super().__init__(weights, fov, dims, n, cam_r, verbose)
        self.n_cameras_sym = n_free_cams
        # cameras
        # self.cameras initialised by super's init
        self.cameras_sym = None
        return
    
    def _update(self):
        assert len(self.cameras_sym) == 4*self.n_cameras_sym, "Bad size for symmetric cameras."
        self.cameras = symmetric_params2regular_params(self.cameras_sym, self.symmetry)
        return
    
    def set_random_cameras(self) -> None:
        super().set_random_cameras()
        self.cameras_sym = self.cameras[:4*self.n_cameras_sym]
        self._update()
        while self.fitness() is np.NaN:
            super().set_random_cameras()
            self.cameras_sym = self.cameras[:4*self.n_cameras_sym]
            self._update()
        return
    
    def train(self, rho=4.0, fitness_upper_limit=0.9) -> None:
        print('here1')
        while self.fitness()>fitness_upper_limit or self.fitness() is np.NaN:
            print('here2')
            super().train(self.cameras_sym, rho, fitness_upper_limit=1e3)
            self.cameras_sym = self.cameras
            self._update()
        print('here3')
        return

    def save(self, path:str) -> None:
        dfo_strength, dfo_curve = self.weights['distance_from_origin'] if type(self.weights['distance_from_origin'])==tuple else (self.weights['distance_from_origin'], 1.0)
        weights = np.array([dfo_strength, dfo_curve, self.weights['stay_within_range'], self.weights['spread'], self.weights['soft_convexity'], self.weights['hard_convexity']])
        metadata = np.array([self.FOV_H, self.FOV_V, self.X_RANGE[0], self.X_RANGE[1], self.Y_RANGE[0], self.Y_RANGE[1], self.Z_RANGE[0], self.Z_RANGE[1], self.N_CAMERAS, self.CAMERA_RADIUS, self.rho if self.rho else np.NaN, self.hull_fail_counter])
        np.savez_compressed(path, weights=weights, symmetry=self.symmetry, cameras=self.cameras, metadata=metadata)
        return

    def load(self, path:str) -> None:
        data = np.load(path)

        self.symmetry = data['symmetry']

        self.weights = {}
        self.weights['distance_from_origin'] = tuple(data['weights'][0:2])
        self.weights['stay_within_range'] = data['weights'][2]
        self.weights['spread'] = data['weights'][3]
        self.weights['soft_convexity'] = data['weights'][4]
        self.weights['hard_convexity'] = data['weights'][5]

        self.FOV_H, self.FOV_V = data['metadata'][0:2]
        self.X_RANGE, self.Y_RANGE, self.Z_RANGE = tuple(data['metadata'][2:4]), tuple(data['metadata'][4:6]), tuple(data['metadata'][6:8])
        self.ALL_RANGE = self.X_RANGE+self.Y_RANGE+self.Z_RANGE
        self.N_CAMERAS = int(data['metadata'][8])
        self.CAMERA_RADIUS = data['metadata'][9]
        self.rho = data['metadata'][10]
        self.hull_fail_counter = data['metadata'][11]
        self.cameras = data['cameras']

        self.n_cameras_sym = self.N_CAMERAS // (2 if self.symmetry=='point' else 4)
        self.cameras_sym = self.cameras[np.array(self.n_cameras_sym*[4*[1]+4*[0]], dtype=bool).reshape(-1)] if self.symmetry=='point' else self.cameras[np.array(self.n_cameras_sym*[4*[1]+12*[0]], dtype=bool).reshape(-1)]
        assert np.array_equal(self.cameras, symmetric_params2regular_params(self.cameras_sym, self.symmetry)), "Error during loading"
        return


def blank_optimizer(optimizer:str='CameraOptimizer') -> CameraOptimizer:
    '''
    Get a blank optimizer object with the given type. Usefull for loading optimizers from file.

    Parameters:
    ----------
        optimizer : str, optional
            possible values: 'CameraOptimizer', 'WeightedOptimizer', 'SymmetricOptimizer', 'WeightedSymmetricOptimizer'. By default 'CameraOptimizer'

    Returns:
    -------
        CameraOptimizer
            blank optimizer object
    '''
    possible_inputs= ['CameraOptimizer', 'WeightedOptimizer', 'SymmetricOptimizer', 'WeightedSymmetricOptimizer']
    weights = {'distance_from_origin': -1., 'stay_within_range': -1., 'spread': -1., 'soft_convexity': -1., 'hard_convexity': -1.}
    if   optimizer==possible_inputs[0]:
        return CameraOptimizer((0, 0), (0, 0, 0), -2, 0., False)
    elif optimizer==possible_inputs[1]:
        return WeightedOptimizer(weights, (0, 0), (0, 0, 0), -2, 0., False)
    elif optimizer==possible_inputs[2]:
        return SymmetricOptimizer('point', (0, 0), (0, 0, 0), -2, 0., False)
    elif optimizer==possible_inputs[3]:
        return WeightedSymmetricOptimizer(weights, 'point', (0, 0), (0, 0, 0), -2, 0., False)
    else:
        raise ValueError(f'Bad value for optimizer.\nPossible inputs are {possible_inputs}.')

def plot_axes(ax, optim:CameraOptimizer) -> None:
    '''
    Set the dimensions of a figure to the space given by the optimizer and plot the (x, y, z) axes.

    Parameters:
    ----------
        ax : matplotlib.axes._subplots.Axes3DSubplot
            axes to plot on
        optim : CameraOptimizer
            optimizer object

    Returns:
    -------
        None
    '''
    ax.plot(optim.X_RANGE, [0, 0], [0, 0], color='red')
    ax.plot([0, 0], optim.Y_RANGE, [0, 0], color='green')
    ax.plot([0, 0], [0, 0], optim.Z_RANGE, color='blue')
    ax.xaxis.label.set_color('red')
    ax.yaxis.label.set_color('green')
    ax.zaxis.label.set_color('blue')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    xmin, xmax = optim.X_RANGE
    ax.set_xlim3d(xmin, xmax)
    ymin, ymax = optim.Y_RANGE
    ax.set_ylim3d(ymin, ymax)
    zmin, zmax = optim.Z_RANGE
    ax.set_zlim3d(zmin, zmax)
    return

