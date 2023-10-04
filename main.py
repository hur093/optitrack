import numpy as np, matplotlib.pyplot as plt
np.set_printoptions(precision=1, suppress=True)
import optitrackoptim as opt

FOV_H, FOV_V = np.deg2rad(56), np.deg2rad(46) # specific to the cameras used
X_LEN, Y_LEN, HEIGHT = 5., 7.5, 3. #meters
N_CAMERAS = 7
CAM_SIZE = 0.3   #radius of the cameras in meters; this is set so a solution won't have cameras overlapping

weights = {'distance_from_origin': 3.0, 'stay_within_range': -1.0, 'spread': -1.0, 'soft_convexity': 2.0, 'hard_convexity': -0.4} #for more info call opt.blank_optimizer('WeightedOptimizer').help()

'''Train'''
if False:
    optim = opt.CameraOptimizer((FOV_H, FOV_V), (X_LEN, Y_LEN, HEIGHT), N_CAMERAS, CAM_SIZE)
    # optim = opt.WeightedOptimizer(weights, (FOV_H, FOV_V), (X_LEN, Y_LEN, HEIGHT), N_CAMERAS, CAM_SIZE)
    # optim = opt.SymmetricOptimizer('point', (FOV_H, FOV_V), (X_LEN, Y_LEN, HEIGHT), N_CAMERAS, CAM_SIZE)
    # optim = opt.WeightedSymmetricOptimizer(weights, 'point', (FOV_H, FOV_V), (X_LEN, Y_LEN, HEIGHT), N_CAMERAS, CAM_SIZE)
    optim.set_random_cameras()
    optim.train(rho=3, fitness_upper_limit=0.5)
    optim.save('vanilla7.npz')
    print(f'Fitness value: {optim.fitness(verbose=True)}')
    exit()


'''Plot a single, manually set camera'''
plot_point_cloud = True
if True:
    fig_cam = plt.figure('Just a camera')
    ax_cam = fig_cam.add_subplot(projection='3d')

    mycam = opt.Camera((-1.5, 0.25, HEIGHT), (np.deg2rad(-45), np.deg2rad(45)), (FOV_H, FOV_V))
    mycam.plot_vertices(ax_cam, 'red')
    mycam.plot_faces(ax_cam)
    # opt.plot_axes(ax_cam, opt.CameraOptimizer((FOV_H, FOV_V), (X_LEN, Y_LEN, HEIGHT), N_CAMERAS, CAM_SIZE))
    ax_cam.set_aspect('equal')

    if plot_point_cloud:
        N = 2000
        point_cloud_m = opt.get_random_points((-2, 2), (-2, 4), (-0.6, 3), N)
        mask = mycam.peekaboo(point_cloud_m)
        ax_cam.scatter(point_cloud_m[:, 0][mask], point_cloud_m[:, 1][mask], point_cloud_m[:, 2][mask], color='black', alpha=0.3)
        ax_cam.set_xlabel('x-axis')
        ax_cam.set_ylabel('y-axis')
        ax_cam.set_zlabel('z-axis')

'''Load an optimizer from file and plot'''
if False:
    fig_opt = plt.figure('Loaded optimizer')
    ax_opt = fig_opt.add_subplot(projection='3d')

    optim = opt.blank_optimizer("CameraOptimizer")
    optim.load('vanilla7.npz')
    opt.plot_axes(ax_opt, optim)
    optim.plot_cameras(ax_opt)
    # optim.plot_seen_points(ax_opt)

plt.show()