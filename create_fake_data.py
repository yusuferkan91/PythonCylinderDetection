import open3d as o3d
import numpy as np


def create_vector(w, h, d):
    vec = o3d.geometry.AxisAlignedBoundingBox(np.array([0, 0, 0]), np.array([w, h, d]))
    return vec


def create_point_cloud(array):
    p_cloud = o3d.geometry.PointCloud()
    p_cloud.points = o3d.utility.Vector3dVector(array)
    return p_cloud


def get_max_min_points(bbox_point):
    x_max = bbox_point[np.where(bbox_point[:, 0] == bbox_point[:, 0].max())[0]][0][0]
    x_min = bbox_point[np.where(bbox_point[:, 0] == bbox_point[:, 0].min())[0]][0][0]
    y_max = bbox_point[np.where(bbox_point[:, 1] == bbox_point[:, 1].max())[0]][0][1]
    y_min = bbox_point[np.where(bbox_point[:, 1] == bbox_point[:, 1].min())[0]][0][1]
    z_max = bbox_point[np.where(bbox_point[:, 2] == bbox_point[:, 2].max())[0]][0][2]
    z_min = bbox_point[np.where(bbox_point[:, 2] == bbox_point[:, 2].min())[0]][0][2]
    return x_min, x_max, y_min, y_max, z_min, z_max


def relieving_pcd(pcd_points, bbox_point):
    x_min, x_max, y_min, y_max, z_min, z_max = get_max_min_points(bbox_point)
    return pcd_points[np.where((pcd_points[:, 0] > x_max) | (pcd_points[:, 0] < x_min) |
                               (pcd_points[:, 1] > y_max) | (pcd_points[:, 1] < y_min) |
                               (pcd_points[:, 2] > z_max) | (pcd_points[:, 2] < z_min))]

pcd = o3d.io.read_point_cloud("fake_cloud.pcd")
pcd.rotate(pcd.get_rotation_matrix_from_xyz((-(np.pi / 16), 0, 0)))
pcd_points = np.asarray(pcd.points)
bbox = pcd.get_oriented_bounding_box()
bbox_point = np.asarray(bbox.get_box_points())

x_min, x_max, y_min, y_max, z_min, z_max = get_max_min_points(bbox_point)

height = y_max - y_min
width = x_max - x_min
depth = z_max - z_min
base_center = "0.06121395_0.04890479_-0.82540863981"
base_radius = "0.157"

for i in range(10):
    h = np.random.choice(np.linspace(height/8, height/3, 10))
    w = np.random.choice(np.linspace(width/6, width/3, 10))

    translate_point = [(np.random.choice(np.linspace(x_min, (x_max-w), 10))), (np.random.choice(np.linspace(y_min, (y_max-h), 10))), z_min]

    vector_crop_bbox = create_vector(w, h, depth)
    vector_crop_bbox.translate(translate_point)

    inlier_pcd = pcd.crop(vector_crop_bbox)
    outlier_pcd = create_point_cloud(relieving_pcd(pcd_points, np.array(vector_crop_bbox.get_box_points())))

    inlier_points = np.asarray(inlier_pcd.points)
    crush_dist = np.random.choice(np.linspace(0.007, 0.015, 10))
    inlier_points = np.vstack((inlier_points[:, 0], inlier_points[:, 1], (inlier_points[:, 2] - crush_dist))).T
    outlier_points = np.asarray(outlier_pcd.points)

    z_angle = np.random.choice(np.linspace(-np.pi/20, np.pi/20, 10))
    x_angle = np.random.choice(np.linspace(-np.pi/20, np.pi/20, 10))

    mu = 0
    sigma = 0.0015
    points = np.vstack((inlier_points, outlier_points))
    print(points.shape)
    points += np.random.normal(mu, sigma, size=points.shape)

    new_pcd = create_point_cloud(points)
    new_pcd.rotate(pcd.get_rotation_matrix_from_xyz((0, 0, z_angle)))
    new_pcd.rotate(pcd.get_rotation_matrix_from_xyz((x_angle, 0, 0)))
    name = "fake_datas/3/z_angle_" + str(z_angle) + "_x_angle_" + str(x_angle) + "_R_" + base_radius + "_C_" + base_center + ".pcd"
    o3d.io.write_point_cloud(name, new_pcd)

# pc = o3d.io.read_point_cloud("fake_datas/z_angle_0.296705972839036_x_angle_0.2792526803190927_R_0.157_C_0.06121395_0.04890479_-0.82540863981.pcd")
# o3d.visualization.draw_geometries([pc])
