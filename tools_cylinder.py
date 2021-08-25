import numpy as np
import open3d as o3d
from sympy import *
import math
import copy


def prepare_pcd(pointCloud=None, path=None, x_max=0.45, x_min=-0.01, y_max=0.53, y_min=0.01, z_max=2.3, z_min=1.6, nb_points=1, radius=0.05):
    """Datayi kirpar ve outler temizlemesi yapar"""
    if pointCloud is None:
        pcd = o3d.io.read_point_cloud(path)
    else:
        pcd = pointCloud
    dist = np.asarray(pcd.points)

    ind = np.where(
        (dist[:, 2] > z_min) & (dist[:, 2] < z_max) & (dist[:, 1] > y_min) & (dist[:, 1] < y_max) & (
                    dist[:, 0] < x_max) & (
                dist[:, 0] > x_min))[0]
    pcd_without = pcd.select_by_index(ind)
    cl, ind1 = pcd_without.remove_radius_outlier(nb_points=nb_points, radius=radius)
    inlier_cloud = pcd_without.select_by_index(ind1)
    return inlier_cloud


def rodrigues_rot(P, n0, n1):
    if P.ndim == 1:
        P = P[np.newaxis, :]

    n0 = n0 / np.linalg.norm(n0)
    n1 = n1 / np.linalg.norm(n1)
    k = np.cross(n0, n1)
    k = k / np.linalg.norm(k)
    theta = np.arccos(np.dot(n0, n1))

    P_rot = np.zeros((len(P), 3))
    for i in range(len(P)):
        P_rot[i] = P[i] * np.cos(theta) + np.cross(k, P[i]) * np.sin(theta) + k * np.dot(k, P[i]) * (1 - np.cos(theta))

    return P_rot


def fit_circle_2d(x, y, w=[]):
    A = np.array([x, y, np.ones(len(x))]).T
    b = x ** 2 + y ** 2

    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W, A)
        b = np.dot(W, b)

    # Solve by method of least squares
    c = np.linalg.lstsq(A, b, rcond=None)[0]

    # Get circle parameters from solution c
    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc ** 2 + yc ** 2)
    return xc, yc, r


def generate_circle_by_vectors(t, C, r, n, u):
    n = n/np.linalg.norm(n)
    u = u/np.linalg.norm(u)
    P_circle = r*np.cos(t)[:, np.newaxis]*u + r*np.sin(t)[:, np.newaxis]*np.cross(n,u) + C
    return P_circle


def fit_circle(P):
    """Cember fit etme icin cagrilan fonksiyon"""
    P[:, 1] = P[:, 1].mean()
    P_mean = P.mean(axis=0)
    P_centered = P - P_mean
    U, s, V = np.linalg.svd(P_centered)
    normal = V[2, :]
    P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])
    xc, yc, r = fit_circle_2d(P_xy[:, 0], P_xy[:, 1])
    C = rodrigues_rot(np.array([xc, yc, 0]), [0, 0, 1], normal) + P_mean
    C = C.flatten()
    t = np.linspace(0, 2 * np.pi, 100)
    u = P[0] - C
    P_fitcircle = generate_circle_by_vectors(t, C, r, normal, u)
    return P_fitcircle, r, C


def apply_noise(pcd, sigma, mu=0):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd


def get_circle(center_x, center_y, radius, n):
    circle_array = np.asarray(points_on_circumference(center=(center_x, center_y), r=radius, n=150, y=n[0]))
    for i in range(1, len(n)):
        data = np.asarray(points_on_circumference(center=(center_x, center_y), r=radius, n=150, y=n[i]))
        circle_array = np.vstack((circle_array, data))
    circle = create_point_cloud(np.asarray(circle_array))
    return circle, circle_array


def create_point_cloud(array):
    p_cloud = o3d.geometry.PointCloud()
    p_cloud.points = o3d.utility.Vector3dVector(array)
    return p_cloud


def points_on_circumference(center=(0, 0), r=50, n=100, y=0):
    return [
        (
            center[0] + (math.cos(2 * pi / n * x) * r),  # x
            y,
            center[1] + (math.sin(2 * pi / n * x) * r)  # y

        ) for x in range(0, n + 1)]


def compute_M_R(x_min_points, x_max_points, z_min_points):
    a = Symbol("a")
    b = Symbol("b")
    r = Symbol("r")
    result = solve([((x_min_points[0] - a) ** 2) + ((x_min_points[2] - b) ** 2) - r ** 2,
                    ((x_max_points[0] - a) ** 2) + ((x_max_points[2] - b) ** 2) - r ** 2,
                    ((z_min_points[0] - a) ** 2) + ((z_min_points[2] - b) ** 2) - r ** 2], [a, b, r])
    result = result[0]
    return result


def get_ref_points(point_cloud):
    x_min = np.where(point_cloud[:, 0] == point_cloud[:, 0].min())[0]
    x_max = np.where(point_cloud[:, 0] == point_cloud[:, 0].max())[0]
    x_min_point = point_cloud[x_min][0]
    x_max_point = point_cloud[x_max][0]
    x_center_point = (x_max_point + x_min_point) / 2
    z_min1 = np.where((point_cloud[:, 0] > (x_center_point[0] - 0.05))
                 & (point_cloud[:, 0] < (x_center_point[0] + 0.05)))[0]
    # print(x_center_point)

    z_min_point = point_cloud[z_min1[0]]
    return x_min_point, x_max_point, z_min_point


def salt_pepper(pcd, SNR=0.01, amount=0.1):
    p_cloud = copy.deepcopy(pcd)
    data = np.asarray(p_cloud.points)
    s_vs_p = 0.5   # salt and pepper arasındaki oran
       # amount degeri: datanın yüzde kaçına noise işlemi uygulanacak
    out = np.copy(data)
    lim_snr = np.linspace(0, SNR, 10)

    # Salt mode
    num_salt = np.ceil(amount * data.shape[0] * s_vs_p)
    coords = np.random.choice(lim_snr, int(num_salt))   # salt noise uygulanacak item kadar random gürültü seç
    zero_arr = np.zeros(out.shape[0] - coords.shape[0], dtype=int)  # Geriye kalan item kadar zero matrix oluştur.
    total_arr = np.hstack([coords, zero_arr])    # İki arrayi birleştirip data uzunluğunda array bulunur.
    np.random.shuffle(total_arr)                # birleştirilen array karıştırılır.
    out[:, 2] = out[:, 2] + total_arr           # Salt değeri eklenir.
    # Pepper mode
    num_pepper = np.ceil(amount * data.shape[0] * (1. - s_vs_p))
    coords1 = np.random.choice(lim_snr, int(num_pepper))  # Pepper noise uygulanacak item kadar random gürültü seç
    zero_arr1 = np.zeros(out.shape[0] - coords1.shape[0], dtype=int)
    total_arr1 = np.hstack([coords1, zero_arr1])
    np.random.shuffle(total_arr1)
    out[:, 2] = out[:, 2] - total_arr1
    pcd_salt_pepper = create_point_cloud(out)
    return pcd_salt_pepper

