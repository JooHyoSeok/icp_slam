"""
ref: https://github.com/ClayFlannigan/icp/blob/master/icp.py

try this later: https://github.com/agnivsen/icp/blob/master/basicICP.py
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

def best_fit_transform(current_scan_points, previous_scan_points):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points current_scan_points to previous_scan_points in m spatial dimensions
    Input:
      current_scan_points: Nxm numpy array of corresponding points (num, dimension)
      previous_scan_points: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps current_scan_points on to previous_scan_points
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert current_scan_points.shape == previous_scan_points.shape

    # get number of dimensions
    m = current_scan_points.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(current_scan_points, axis=0)
    centroid_B = np.mean(previous_scan_points, axis=0)
    AA = current_scan_points - centroid_A
    BB = previous_scan_points - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(current_points, previous_points):
    '''
    Find the nearest (Euclidean) neighbor in previous_points for each point in current_points
    Input:
        current_points: Nxm array of points
        previous_points: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: previous_points indices of the nearest neighbor
    '''

    assert current_points.shape == previous_points.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(previous_points)
    distances, indices = neigh.kneighbors(current_points, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(current_scan_points, previous_scan_points, prev_relative_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points current_scan_points on to points previous_scan_points
    Input:
        current_scan_points: Nxm numpy array of source mD points (5000, 3)
        previous_scan_points: Nxm numpy array of destination mD point (5000, 3)
        prev_relative_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps current_scan_points on to previous_scan_points
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert current_scan_points.shape == previous_scan_points.shape

    # get number of dimensions
    m = current_scan_points.shape[1]

    # make points homogeneous, copy them to maintain the originals
    current_points = np.ones((m+1,current_scan_points.shape[0]))
    previous_points = np.ones((m+1,previous_scan_points.shape[0]))

    current_points[:m,:] = np.copy(current_scan_points.T)
    previous_points[:m,:] = np.copy(previous_scan_points.T)

    # apply the initial pose estimation
    if prev_relative_pose is not None:
        current_points = np.dot(prev_relative_pose, current_points) # for fast convergence speed

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(current_points[:m,:].T, previous_points[:m,:].T)
        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(current_points[:m,:].T, previous_points[:m,indices].T)

        # update the current source
        current_points = np.dot(T, current_points)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    matched_points_with_prev = np.copy(current_points)

    ## current_scan_points.T  = (3, 5000) axis = 0 col, axis = 1 , row
    # print("original error : " , np.mean(np.sqrt(np.sum(np.square(current_scan_points.T - previous_scan_points.T), axis = 0))))
    # print("optimal error : " , np.mean(np.sqrt(np.sum(np.square(matched_points_with_prev[:m,:] - previous_scan_points.T), axis = 0))))
    # print("original error : " , np.mean(np.mean(np.square(current_scan_points.T - previous_scan_points.T), axis = 0)))
    # print("optimal error : " , np.mean(np.mean(np.square(matched_points_with_prev[:m,:] - previous_scan_points.T), axis = 0)))
    # center_of_mass_diff = np.mean(np.sqrt(np.square(np.mean(current_scan_points.T, axis= 1) - np.mean(previous_scan_points.T, axis= 1))))
    # print("mass of center diff : ", center_of_mass_diff )
    # optimal_center_of_mass_diff = np.mean(np.sqrt(np.square(np.mean(matched_points_with_prev[:m,:], axis= 1) - np.mean(previous_scan_points.T, axis= 1))))
    # print("optimal_center_of_mass_diff",optimal_center_of_mass_diff)
    # calculate final transformation
    T,_,translation = best_fit_transform(current_scan_points, matched_points_with_prev[:m,:].T)
    # print(np.mean(distances))
    # print(np.linalg.norm(T[:,-1].T[:3],axis=0))
    # print(np.linalg.norm(translation))
    return T, translation, i

