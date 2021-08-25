import copy

import detection_cylinder
import tools_cylinder as cylinder
import open3d as o3d
import numpy as np


def create_cylinder(center=(0.047, -0.843), radius=-0.155, n=np.linspace(0.314, -0.075, 50)):
    """
    :param center: Silindirin baslangic cemberinin merkezi
    :param radius: Cemberin yaricapi
    :param n: Y ekseni boyunca cember sayisi
    :return: Silindir Point cloud

    Default silindir degerleri gercek veriye yaklaşık değerler olustursun diye bu sekilde secilmistir"""
    return cylinder.get_circle(center[0], center[1], radius, n)


def crop_point_cloud(pcd, x_max=0.25, x_min=-0.1, y_max=0.34, y_min=0.01, z_max=-0.6, z_min=-0.77):
    """
    :param pcd: Point cloud
    :param x_max: X ekseninde maksimum deger
    :param x_min: X ekseninde minimum deger
    :param y_max: Y ekseninde maksimum deger
    :param y_min: Y ekseninde minimum deger
    :param z_max: Z ekseninde maksimum deger
    :param z_min: Z ekseninde minimum deger
    :return: Point cloud

    Point cloud'u (pcd) min max degerlerine gore kirpar ve tup yuzeyine benzer yuzey olusturur."""
    pcd1 = cylinder.prepare_pcd(pcd, x_max=x_max, x_min=x_min, y_max=y_max, y_min=y_min, z_max=z_max, z_min=z_min)
    return pcd1


def add_noise(pcd, noise_type, sigma=0.003, SNR=0.01, amount=0.1):
    """
    :param pcd: Point Cloud
    :param noise_type: Noise tipi
    :param sigma: Gauss parametresi. Gurultu siddeti
    :param SNR: Salt and pepper parametresi. Gurultu siddeti
    :param amount: Salt and pepper parametresi. Gurultu verinin yuzde kacina uygulanacak
    :return: Point cloud
    """
    point_cloud = None
    if noise_type == 'gauss':
        point_cloud = cylinder.apply_noise(pcd, sigma)
    if noise_type == 'salt_and_pepper':
        point_cloud = cylinder.salt_pepper(pcd, SNR, amount)
    return point_cloud


def add_crush(pcd):
    """
    :param pcd: Point cloud uzerine random vuruk/ ezik bölge ekler
    :return: Point cloud
    """
    pcd_points = np.asarray(pcd.points)
    bbox = pcd.get_oriented_bounding_box()
    bbox_point = np.asarray(bbox.get_box_points())

    x_min, x_max, y_min, y_max, z_min, z_max = get_max_min_points(bbox_point)

    height = y_max - y_min
    width = x_max - x_min
    depth = z_max - z_min
    h = np.random.choice(np.linspace(height/8, height/3, 10))
    w = np.random.choice(np.linspace(width/6, width/3, 10))

    translate_point = [(np.random.choice(np.linspace(x_min, (x_max-w), 10))), (np.random.choice(np.linspace(y_min, (y_max-h), 10))), z_min]

    vector_crop_bbox = create_bbox(w, h, depth)
    vector_crop_bbox.translate(translate_point)

    inlier_pcd = pcd.crop(vector_crop_bbox)
    outlier_pcd = cylinder.create_point_cloud(outside_of_bbox(pcd_points, np.array(vector_crop_bbox.get_box_points())))

    inlier_points = np.asarray(inlier_pcd.points)
    crush_dist = np.random.choice(np.linspace(0.007, 0.01, 10))
    inlier_points = np.vstack((inlier_points[:, 0], inlier_points[:, 1], (inlier_points[:, 2] - crush_dist))).T
    outlier_points = np.asarray(outlier_pcd.points)

    points = np.vstack((inlier_points, outlier_points))
    point_cloud = cylinder.create_point_cloud(points)
    return point_cloud


def get_max_min_points(bbox_point):
    """
    :param bbox_point: Bounding box
    :return: Bounding boxin mininmum maksimum degerlerini verir.
    """
    x_max = bbox_point[np.where(bbox_point[:, 0] == bbox_point[:, 0].max())[0]][0][0]
    x_min = bbox_point[np.where(bbox_point[:, 0] == bbox_point[:, 0].min())[0]][0][0]
    y_max = bbox_point[np.where(bbox_point[:, 1] == bbox_point[:, 1].max())[0]][0][1]
    y_min = bbox_point[np.where(bbox_point[:, 1] == bbox_point[:, 1].min())[0]][0][1]
    z_max = bbox_point[np.where(bbox_point[:, 2] == bbox_point[:, 2].max())[0]][0][2]
    z_min = bbox_point[np.where(bbox_point[:, 2] == bbox_point[:, 2].min())[0]][0][2]
    return x_min, x_max, y_min, y_max, z_min, z_max


def outside_of_bbox(pcd_points, bbox_point):
    """
    :param pcd_points: Point cloud
    :param bbox_point: Bounding box
    :return: Point cloud uzerinde bounding box disinda kalan alani verir.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = get_max_min_points(bbox_point)
    return pcd_points[np.where((pcd_points[:, 0] > x_max) | (pcd_points[:, 0] < x_min) |
                               (pcd_points[:, 1] > y_max) | (pcd_points[:, 1] < y_min) |
                               (pcd_points[:, 2] > z_max) | (pcd_points[:, 2] < z_min))]


def create_bbox(w, h, d):
    """
    :param w: Genislik
    :param h: Yukseklik
    :param d: Derinlik
    :return: Bounding box olusturur.
    """
    vec = o3d.geometry.AxisAlignedBoundingBox(np.array([0, 0, 0]), np.array([w, h, d]))
    return vec


def add_angle(pcd, alpha, beta, is_random=True):
    """
    :param pcd: Point cloud
    :param alpha: Z normali (0,0,1) ile point cloud arasindaki aciyi temsil eder. X ekseni
    "cevresinde" dondurulur.
    :param beta: X normali (1,0,0) ile point cloud arasindaki aciyi temsil eder. Z ekseni
    "cevresinde" dondurulur.
    :param is_random: Eger true ise alpha ve beta acisi ust sinir kabul edilir ve random
    deger seçilir. False is direk alpha-beta kadar dondurulur.
    :return: Point cloud
    """
    point_cloud = copy.deepcopy(pcd)
    if is_random:
        alpha_list = np.linspace(0, alpha, 10)
        beta_list = np.linspace(0, beta, 10)
        alpha = np.random.choice(alpha_list, 1)
        beta = np.random.choice(beta_list, 1)
    point_cloud.rotate(point_cloud.get_rotation_matrix_from_xyz((alpha, 0, 0)))
    point_cloud.rotate(point_cloud.get_rotation_matrix_from_xyz((0, 0, beta)))
    return point_cloud


def compute_distance_pcd(source, target):
    """
    :param source: Ezik bolgenin bulundugu point cloud
    :param target: Silindir point cloud
    :return: ezik bölgenin silindirden uzakligina gore 2 point cloud verir.
    """
    dist = source.compute_point_cloud_distance(target)
    dists = np.asarray(dist)
    ind = np.where(dists > 0.005)[0] # ezik siddeti 0.005ten buyukse secer
    outlier_cloud = source.select_by_index(ind)
    inlier_cloud = source.select_by_index(ind, invert=True)

    # cl1, ind1 = outlier_cloud.remove_radius_outlier(nb_points=16, radius=0.008)
    # outlier_cloud = outlier_cloud.select_by_index(ind1)
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0, 0, 1])
    target.paint_uniform_color([0.7, 0.7, 0.7])
    return inlier_cloud, outlier_cloud

"""Silindir olusturma"""
pcd, _ = create_cylinder()

"""Silindirin kirpilmasi"""
crop_pcd = crop_point_cloud(pcd)

"""random vuruk ezik bolge olusturur."""
crop_pcd = add_crush(crop_pcd)

"""Gurultu ekleme yontemleri"""
gauss_pcd = add_noise(crop_pcd, 'gauss', sigma=0.001)
salt_and_pepper_pcd = add_noise(crop_pcd, 'salt_and_pepper', SNR=0.01, amount=0.1)

"""Aci ekleme yontemleri"""
rotate_gauss_pcd = add_angle(gauss_pcd, np.pi/16, np.pi/20, is_random=True)
rotate_salt_and_pepper_pcd = add_angle(salt_and_pepper_pcd, np.pi/16, np.pi/20, is_random=True)

"""Silindir detect islemleri"""
salt_and_pepper_detect = detection_cylinder.Cylinder_detection(pcd=rotate_salt_and_pepper_pcd)
# gauss_detect = detection_cylinder.Cylinder_detection(pcd=rotate_gauss_pcd)

o3d.visualization.draw_geometries([salt_and_pepper_detect.point_cloud, salt_and_pepper_detect.cylinder])

"""vuruk ezik tespit etme islemleri"""
inlier, outlier = compute_distance_pcd(salt_and_pepper_detect.point_cloud, salt_and_pepper_detect.cylinder)
o3d.visualization.draw_geometries([inlier, outlier, salt_and_pepper_detect.cylinder])
