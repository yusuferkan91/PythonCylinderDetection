import open3d as o3d
import numpy as np
from skspatial.objects import Line


def fit(pcd_points):
    points = np.asarray(pcd_points.points)

    line_pcd, direction, max_y, min_y = None, None, None, None

    if points.shape[0] > 0:
        line_fit = Line.best_fit(points)
        point = line_fit.point
        direction = line_fit.direction

        max_y_ind = np.where(points[:, 1] == points[:, 1].max())[0]
        min_y_ind = np.where(points[:, 1] == points[:, 1].min())[0]
        max_y = points[max_y_ind][0]
        min_y = points[min_y_ind][0]
        n = np.linspace(max_y[1]+0.2, min_y[1]-0.2, 100)
        line = [(direction[0] * x + point[0], direction[1] * x + point[1], direction[2] * x + point[2]) for x in n]
        line_arr = np.array(line)
        line_arr = line_arr[(line_arr[:, 1] < max_y[1]) & (line_arr[:, 1] > min_y[1])]
        line_pcd = create_point_cloud(line_arr)
        line_pcd.paint_uniform_color([1, 0, 0])
    return line_pcd, direction, max_y, min_y


def get_angle(line1_normal, line2_normal, data, x_vs_z):
    unit_vector1 = line1_normal / np.linalg.norm(line1_normal)
    unit_vector2 = line2_normal / np.linalg.norm(line2_normal)

    dot_product = np.dot(unit_vector1, unit_vector2)

    angle = np.arccos(dot_product)
    angle = (np.pi/2) - angle
    return -abs(angle)
    # v1_u = unit_vector(line1_normal)
    # v2_u = unit_vector(line2_normal)
    # return -np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))-(np.pi/2)
    # u1_u2 = (line1_normal[0] * line2_normal[0]) + (line1_normal[1] * line2_normal[1]) + (
    #             line1_normal[2] * line2_normal[2])
    # sqr_u1 = np.sqrt(np.square(line1_normal).sum())
    # sqr_u2 = np.sqrt(np.square(line2_normal).sum())
    # alpha = u1_u2 / (sqr_u1 * sqr_u2)
    # angle = np.arccos(alpha)
    #
    # angle = (np.pi / 2) - angle
    #
    # return -angle


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def line_cloud(point_a, point_b):
    line, line_normal = get_line(point_a, point_b)
    plane, plane_normal = get_line(point_a, (0, 0, 1))
    angle = get_angle(line_normal, plane_normal)
    plane_pcd = create_point_cloud(plane)
    line_pcd = create_point_cloud(line)
    return line_pcd, angle, plane_pcd


def get_line(point_a, point_b):
    n = np.linspace(1.6, -0.3, 50)
    normal = (point_b[0] - point_a[0], point_b[1] - point_a[1], point_b[2] - point_a[2])
    line = [(normal[0] * x + point_a[0], normal[1] * x + point_a[1], normal[2] * x + point_a[2]) for x in n]
    return line, normal


def create_point_cloud(array):
    p_cloud = o3d.geometry.PointCloud()
    p_cloud.points = o3d.utility.Vector3dVector(array)
    return p_cloud


def select_point(points):
    points1 = points[:41, :]
    points2 = points[41:82, :]
    points3 = points[82:123, :]
    points4 = points[123:, :]
    selected1 = points1[np.random.choice(points1.shape[0], 1)][0]
    selected2 = points2[np.random.choice(points2.shape[0], 1)][0]
    selected3 = points3[np.random.choice(points3.shape[0], 1)][0]
    selected4 = points4[np.random.choice(points4.shape[0], 1)][0]
    list_points = [selected1, selected2, selected3, selected4]
    return list_points


def compute_error(source, target):
    dists = source.compute_point_cloud_distance(target)
    dists = np.asarray(dists)
    ind = np.where(dists > 0.003)[0]
    outlier_cloud = source.select_by_index(ind)
    src = np.asarray(source.points)
    outlier = np.asarray(outlier_cloud.points)
    error = (outlier.shape[0] * 100) / src.shape[0]
    return error

