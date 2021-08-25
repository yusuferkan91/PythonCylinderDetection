import open3d as o3d
import numpy as np
import copy
from sympy import *
import math
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib import style
from time import gmtime, strftime
import pandas as pd


def prepare_pcd(pointCloud=None, path=None, x_max=0.45, x_min=-0.01, y_max=0.53, y_min=0.01, z_max=2.3, z_min=1.6, nb_points=1, radius=0.05):
    if pointCloud is None:
        pcd = o3d.io.read_point_cloud(path)
    else:
        pcd = pointCloud
    dist = np.asarray(pcd.points)

    ind = np.where(
        (dist[:, 2] > z_min) & (dist[:, 2] < z_max) & (dist[:, 1] > y_min) & (dist[:, 1] < y_max) & (
                    dist[:, 0] < x_max) & (
                dist[:, 0] > x_min))[0]
    # print("index shape::", ind.shape)
    pcd_without = pcd.select_by_index(ind)
    cl, ind1 = pcd_without.remove_radius_outlier(nb_points=nb_points, radius=radius)
    inlier_cloud = pcd_without.select_by_index(ind1)
    return inlier_cloud


def create_point_cloud(array):
    p_cloud = o3d.geometry.PointCloud()
    p_cloud.points = o3d.utility.Vector3dVector(array)
    return p_cloud


def get_line(point_a, point_b):
    n = np.linspace(1.6, -0.3, 50)
    normal = (point_b[0] - point_a[0], point_b[1] - point_a[1], point_b[2] - point_a[2])
    line = [(normal[0] * x + point_a[0], normal[1] * x + point_a[1], normal[2] * x + point_a[2]) for x in n]
    line_pcd = create_point_cloud(line)
    return line_pcd


def get_normal(point_a, point_b):
    # print("point a: ", point_a, "point b: ", point_b)
    return [(point_a[0]-point_b[0]), (point_a[1]-point_b[1]), (point_a[2]-point_b[2])]


def get_angle(line1_normal, line2_normal):

    u1_u2 = (line1_normal[0] * line2_normal[0]) + (line1_normal[1] * line2_normal[1]) + (
                line1_normal[2] * line2_normal[2])
    sqr_u1 = np.sqrt(np.square(line1_normal).sum())
    sqr_u2 = np.sqrt(np.square(line2_normal).sum())
    u1_u2 = round(u1_u2, 5)
    sqr_u1 = round(sqr_u1, 5)
    sqr_u2 = round(sqr_u2, 5)
    # print("u::", u1_u2, "s1::", sqr_u1, "s2::", sqr_u2)
    alpha = u1_u2 / (sqr_u1 * sqr_u2)
    # print("alpha::", alpha)
    angle = np.arccos(alpha)
    # print("get angle:", angle)
    # # if angle > 0:
    # angle = -angle
    # angle = (np.pi / 2) - angle
    # angle = angle + (np.pi/2)
    # print("(np.pi)-angle::", (np.pi)-angle)
    return (np.pi)-angle


def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd


def get_half_points(mesh):
    points = np.asarray(mesh.get_box_points())
    max_bound = mesh.get_max_bound()
    min_bound = mesh.get_min_bound()
    center_bound = mesh.get_center()
    bottom_rectangle = points[np.where(points[:, 1] < center_bound[1])]
    top_rectangle = points[np.where(points[:, 1] > center_bound[1])]
    top_x_points = top_rectangle[np.where((top_rectangle[:, 0] < center_bound[0]))]
    y_direction = top_x_points[np.where(top_x_points[:, 2] == top_x_points[:, 2].min())][0]
    # print(y_direction)
    # x_min_points = bottom_rectangle[np.where(bottom_rectangle[:, 0] < center_bound[0])]
    # x_max_points = bottom_rectangle[np.where(bottom_rectangle[:, 0] > center_bound[0])]
    x_min_points = bottom_rectangle[np.where(bottom_rectangle[:, 0] < ((bottom_rectangle[:, 0].max() + bottom_rectangle[:, 0].min())/2))]
    x_max_points = bottom_rectangle[np.where(bottom_rectangle[:, 0] > ((bottom_rectangle[:, 0].max() + bottom_rectangle[:, 0].min())/2))]
    x_direction = x_max_points[np.where(x_max_points[:, 2] == x_max_points[:, 2].min())][0]
    central_point = x_min_points[np.where(x_min_points[:, 2] == x_min_points[:, 2].min())][0]
    z_direction = x_min_points[np.where(x_min_points[:, 2] == x_min_points[:, 2].max())][0]
    return central_point, x_direction, y_direction, z_direction


def get_intersection_mesh(data, mesh_box):
    # bbox = o3d.geometry.OrientedBoundingBox()
    # bbox = bbox.create_from_points(o3d.utility.Vector3dVector(np.asarray(mesh_box.vertices)))
    # try:
    bbox = mesh_box.get_axis_aligned_bounding_box()
    pcd_witout = data.crop(bbox)
    # except:
    #     o3d.visualization.draw_geometries([data])
    return pcd_witout


def get_xz_mesh_list(data):
    data_arr = np.asarray(data.points)
    x_max, x_min = data_arr[:, 0].max(), data_arr[:, 0].min()
    y_max, y_min = data_arr[:, 1].max(), data_arr[:, 1].min()
    z_max, z_min = data_arr[:, 2].max(), data_arr[:, 2].min()

    lines = np.linspace(y_max - 0.05, y_min + 0.05, 5)
    mesh = o3d.geometry.TriangleMesh()
    mesh_box = mesh.create_box(width=0.8, height=0.01,
                               depth=0.8)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    mesh_planes = []
    check_list = []
    for i in lines:
        copy_mesh = copy.deepcopy(mesh_box)
        copy_mesh.paint_uniform_color([0.9, 0.1, 1.0])
        copy_mesh.translate((x_min - 0.01, i, z_min - 0.03))
        check_data = get_intersection_mesh(data, copy_mesh)
        if len(check_data.points) > 10:
            mesh_planes.append(copy_mesh)
            check_list.append(check_data)
    # o3d.visualization.draw_geometries(mesh_planes+[data])
    return mesh_planes, check_list


def fixed_pcd(point_cloud):
    pcd = copy.deepcopy(point_cloud)
    mesh1 = pcd.get_axis_aligned_bounding_box()
    mesh2 = pcd.get_oriented_bounding_box()
    # o3d.visualization.draw_geometries([pcd, mesh2, mesh1])
    mesh_points = np.asarray(mesh1.get_box_points())
    mesh1_points = np.asarray(mesh2.get_box_points())
    central, x_dir, y_dir, z_dir = get_half_points(mesh2)
    z_normal = get_normal(central, z_dir)
    y_normal = get_normal(central, y_dir)
    x_normal = get_normal(central, x_dir)
    line = get_line(central, z_dir)
    line1 = get_line(central, x_dir)
    line2 = get_line(central, y_dir)

    angle_x = get_angle(x_normal, (1, 0, 0))
    angle_y = get_angle(y_normal, (0, 1, 0))
    angle_z = get_angle(z_normal, (0, 0, 1))
    # print(angle_z)
    # print(angle_x)
    # print(np.pi / 16)
    # print(np.pi / 20)
    pcd.rotate(pcd.get_rotation_matrix_from_xyz((-angle_z, 0, 0)))
    pcd.rotate(pcd.get_rotation_matrix_from_xyz((0, 0, -angle_x)))
    pcd.rotate(pcd.get_rotation_matrix_from_xyz((0, angle_y, 0)))
    # o3d.visualization.draw_geometries([pcd, mesh1, mesh2, line, line1, line2])
    return pcd, central, y_dir, angle_z, angle_x, mesh2


def compute_error(source, target):
    dists = source.compute_point_cloud_distance(target)
    dists = np.asarray(dists)
    ind = np.where(dists > 0.009)[0]
    outlier_cloud = source.select_by_index(ind)
    src = np.asarray(source.points)
    outlier = np.asarray(outlier_cloud.points)
    error = (outlier.shape[0] * 100) / src.shape[0]
    # print("line error::", error)
    return error


def rodrigues_rot(P, n0, n1):
    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[np.newaxis, :]

    # Get vector of rotation k and angle theta
    n0 = n0 / np.linalg.norm(n0)
    n1 = n1 / np.linalg.norm(n1)
    k = np.cross(n0, n1)
    k = k / np.linalg.norm(k)
    theta = np.arccos(np.dot(n0, n1))

    # Compute rotated points
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
    P[:, 1] = P[:, 1].mean()
    P_mean = P.mean(axis=0)
    P_centered = P - P_mean
    U, s, V = np.linalg.svd(P_centered)
    normal = V[2, :]
    d = -np.dot(P_mean, normal)
    P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])
    xc, yc, r = fit_circle_2d(P_xy[:, 0], P_xy[:, 1])
    # print("fit radius::", r)
    t = np.linspace(0, 2 * np.pi, 100)
    xx = xc + r * np.cos(t)
    yy = yc + r * np.sin(t)
    C = rodrigues_rot(np.array([xc, yc, 0]), [0, 0, 1], normal) + P_mean
    C = C.flatten()
    t = np.linspace(0, 2 * np.pi, 100)
    # print(P.shape)
    # print(C)
    u = P[0] - C
    P_fitcircle = generate_circle_by_vectors(t, C, r, normal, u)
    return P_fitcircle, r, C


def get_ref_points(point_cloud):
    # print("circle point::", point_cloud.shape)
    x_min = np.where(point_cloud[:, 0] == point_cloud[:, 0].min())[0]
    x_max = np.where(point_cloud[:, 0] == point_cloud[:, 0].max())[0]
    x_min_point = point_cloud[x_min][0]
    x_max_point = point_cloud[x_max][0]
    x_center_point = (x_max_point + x_min_point) / 2
    # print(x_max_point, x_min_point, x_center_point)
    z_min1 = np.where((point_cloud[:, 0] > (x_center_point[0] - 0.05))
                 & (point_cloud[:, 0] < (x_center_point[0] + 0.05)))[0]


    z_min_point = point_cloud[z_min1[0]]
    # print(z_min_point)
    return x_min_point, x_max_point, z_min_point


def compute_M_R(x_min_points, x_max_points, z_min_points):
    a = Symbol("a")
    b = Symbol("b")
    r = Symbol("r")
    result = solve([((x_min_points[0] - a) ** 2) + ((x_min_points[2] - b) ** 2) - r ** 2,
                    ((x_max_points[0] - a) ** 2) + ((x_max_points[2] - b) ** 2) - r ** 2,
                    ((z_min_points[0] - a) ** 2) + ((z_min_points[2] - b) ** 2) - r ** 2], [a, b, r])
    result = result[0]
    return result


def points_on_circumference(center=(0, 0), r=50, n=100, y=0):
    return [
        (
            center[0] + (math.cos(2 * pi / n * x) * r),  # x
            y,
            center[1] + (math.sin(2 * pi / n * x) * r)  # y

        ) for x in range(0, n + 1)]


def get_circle(center_x, center_y, radius, n):
    circle_array = np.asarray(points_on_circumference(center=(center_x, center_y), r=radius, n=150, y=n[0]))
    for i in range(1, len(n)):
        data = np.asarray(points_on_circumference(center=(center_x, center_y), r=radius, n=150, y=n[i]))
        circle_array = np.vstack((circle_array, data))
    # print(circle_array)
    circle = create_point_cloud(np.asarray(circle_array))
    return circle, circle_array


def salt_pepper(pcd_arr, SNR=0.01, amount=0.15):
    s_vs_p = 0.5  # salt and pepper arasındaki oran
    out = np.copy(pcd_arr)
    lim_snr = [SNR, -SNR]
    lim_snr = np.array(lim_snr)
    # Salt mode
    num_salt = np.ceil(amount * pcd_arr.shape[0] * s_vs_p)
    coords = np.random.choice(lim_snr, int(num_salt))  # salt noise uygulanacak item kadar random gürültü seç
    zero_arr = np.zeros(out.shape[0] - coords.shape[0], dtype=int)  # Geriye kalan item kadar zero matrix oluştur.
    total_arr = np.hstack([coords, zero_arr])  # İki arrayi birleştirip data uzunluğunda array bulunur.
    np.random.shuffle(total_arr)  # birleştirilen array karıştırılır.
    out[:, 2] = out[:, 2] + total_arr  # Salt değeri eklenir.
    # Pepper mode
    num_pepper = np.ceil(amount * pcd_arr.shape[0] * (1. - s_vs_p))
    coords1 = np.random.choice(lim_snr, int(num_pepper))  # Pepper noise uygulanacak item kadar random gürültü seç
    zero_arr1 = np.zeros(out.shape[0] - coords1.shape[0], dtype=int)
    total_arr1 = np.hstack([coords1, zero_arr1])
    np.random.shuffle(total_arr1)
    out[:, 2] = out[:, 2] - total_arr1
    return out


def mean_list(list):
    return sum(list)/len(list)


def get_graph(y, y2, x, name1, name2, xName, yName, title):
    style.use('ggplot')
    plt.plot(x, y, 'g', label=name1, linewidth=5)
    plt.plot(x, y2, 'b', label=name2, linewidth=5)
    plt.title(title)
    plt.ylabel(yName)
    plt.xlabel(xName)
    plt.legend()
    plt.grid(True, color='k')
    plt.savefig(title + "_" + strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + ".png")
    plt.show()


def circle_tools(candidate_circle, max_y, min_y):
    best_circle = None
    best_circle_radius = None
    best_circle_points = None
    best_circle_center = None
    err_ind = 1000
    # print("----------------------")
    for i in candidate_circle:
        # cl, ind1 = i.remove_radius_outlier(nb_points=5, radius=0.05)
        # i = i.select_by_index(ind1)
        points_circle = np.asarray(i.points)
        # points_circle[:, 1] = points_circle[:, 1].mean()
        if points_circle.shape[0] > 20:
            fit_circle_arr, circle_radius, C = fit_circle(points_circle)
            pcd_circle = create_point_cloud(fit_circle_arr)
            err = compute_error(i, pcd_circle)

            if err < err_ind:
                # print("error::", err)
                err_ind = err
                best_circle_points = fit_circle_arr
                best_circle = pcd_circle
                best_circle_radius = circle_radius
                best_circle_center = C

    x_min, x_max, x_center = get_ref_points(best_circle_points)
    result = compute_M_R(x_min, x_max, x_center)
    n = np.linspace(max_y[1] + 0.01, min_y[1] - 0.05, 50)
    # print(n)
    # cylinder, _ = get_circle(result[0], result[1], result[2], n)
    return best_circle_center, best_circle_radius


def create_graphics(pcd):
    sigma = np.linspace(0.001, 0.1, 10, dtype=np.float)
    slide = 0.0
    base_center = np.array([0.06121395, 0.04890479, -0.82540863981])
    base_radius = 0.157
    base_angle1 = np.pi / 16
    base_angle2 = np.pi / 20

    abs_gauss_center = []
    abs_gauss_radius = []
    abs_gauss_angle1 = []
    abs_gauss_angle2 = []
    total_gauss_center = []
    total_gauss_radius = []
    total_gauss_angle1 = []
    total_gauss_angle2 = []

    abs_S_P_center = []
    abs_S_P_radius = []
    abs_S_P_angle1 = []
    abs_S_P_angle2 = []
    total_S_P_center = []
    total_S_P_radius = []
    total_S_P_angle1 = []
    total_S_P_angle2 = []
    # print(sigma[4])
    data_SP = None
    data_gauss = None
    xz_planes = None
    xz_planes_sp = None
    for i in sigma:
        # print(np.where(sigma == i))
        i = round(i, 3)
        for j in range(10):
            data_gauss = apply_noise(pcd, 0, i)
            rotate_pcd, min_y, max_y, best_angle, best_angle_h, mesh_gauss = fixed_pcd(data_gauss)
            # print(len(rotate_pcd.points))
            xz_planes, candidate_circles = get_xz_mesh_list(rotate_pcd)
            best_circle_center, best_circle_radius = circle_tools(candidate_circles, max_y, min_y)
            # total_list.append(test_gauss)
            base_center[1] = best_circle_center[1]
            dist_gauss = distance.euclidean(base_center, best_circle_center)
            abs_gauss_radius.append(best_circle_radius)
            abs_gauss_center.append(dist_gauss)
            abs_gauss_angle1.append(best_angle)
            abs_gauss_angle2.append(best_angle_h)
            # if j == 0:
            #     o3d.visualization.draw_geometries([test_gauss.point_cloud, test_gauss.cylinder, test_gauss.xy_planes])
            salt = salt_pepper(np.asarray(pcd.points), SNR=i, amount=0.2)
            data_SP = create_point_cloud(salt)
            rotate_pcd_sp, min_y_sp, max_y_sp, best_angle_sp, best_angle_h_sp, mesh_sp = fixed_pcd(data_SP)
            xz_planes_sp, candidate_circles_sp = get_xz_mesh_list(rotate_pcd_sp)
            best_circle_center_sp, best_circle_radius_sp = circle_tools(candidate_circles_sp, max_y_sp, min_y_sp)
            base_center[1] = best_circle_center_sp[1]
            dist_SP = distance.euclidean(base_center, best_circle_center_sp)
            abs_S_P_radius.append(best_circle_radius_sp)
            abs_S_P_center.append(dist_SP)
            abs_S_P_angle1.append(best_angle_sp)
            abs_S_P_angle2.append(best_angle_h_sp)
        # o3d.visualization.draw_geometries([data_SP, mesh_sp])
        # o3d.visualization.draw_geometries([data_gauss, mesh_gauss])
        total_gauss_radius.append(abs(base_radius - mean_list(abs_gauss_radius)))
        total_gauss_center.append(mean_list(abs_gauss_center))
        total_gauss_angle1.append(abs(base_angle1 - mean_list(abs_gauss_angle1)))
        total_gauss_angle2.append(abs(base_angle2 - mean_list(abs_gauss_angle2)))
        print("Gauss Result   >> Radius:: ", abs(base_radius - mean_list(abs_gauss_radius)),
              " Center:: ", mean_list(abs_gauss_center),
              " Angle1:: ", abs(base_angle1 - mean_list(abs_gauss_angle1)),
              " Angle2:: ", abs(base_angle1 - mean_list(abs_gauss_angle1)))

        total_S_P_radius.append(abs(base_radius - mean_list(abs_S_P_radius)))
        total_S_P_center.append(mean_list(abs_S_P_center))
        total_S_P_angle1.append(abs(base_angle1 - mean_list(abs_S_P_angle1)))
        total_S_P_angle2.append(abs(base_angle2 - mean_list(abs_S_P_angle2)))
        print("S_and_P Result >> Radius:: ", abs(base_radius - mean_list(abs_S_P_radius)),
              " Center:: ", mean_list(abs_S_P_center),
              " Angle1:: ", abs(base_angle1 - mean_list(abs_S_P_angle1)),
              " Angle2:: ", abs(base_angle1 - mean_list(abs_S_P_angle2)))
    dict_lists = { 'Gauss Radius': total_gauss_radius, 'Gauss Center': total_gauss_center,
                   'Gauss Z Angle': total_gauss_angle1, 'Gauss X Angle': total_gauss_angle2,
                   'SaltAndPepper Radius': total_S_P_radius, 'SaltAndPepper Center': total_S_P_center,
                   'SaltAndPepper Z Angle': total_S_P_angle1, 'SaltAndPepper X Angle': total_S_P_angle2,
                   'Sigma': sigma}
    df = pd.DataFrame(dict_lists)
    df.to_csv('Data_for_graphics_50iteration.csv')

    get_graph(total_gauss_radius, total_S_P_radius, sigma, "Gauss", "Salt and Pepper", "Gürültü Şiddeti", "Yarıçap",
              "Yarıçap Hata Grafiği")
    get_graph(total_gauss_center, total_S_P_center, sigma, "Gauss", "Salt and Pepper", "Gürültü Şiddeti",
              "Merkez Noktasının uzaklığı", "Merkez Noktası Hata Grafiği")
    get_graph(total_gauss_angle1, total_S_P_angle1, sigma, "Gauss", "Salt and Pepper", "Gürültü Şiddeti", "Açı",
              "Z Ekseni İle Açısının Hata Grafiği")
    get_graph(total_gauss_angle2, total_S_P_angle2, sigma, "Gauss", "Salt and Pepper", "Gürültü Şiddeti", "Açı",
              "X Ekseni İle Açısının Hata Grafiği")


pcd = o3d.io.read_point_cloud("fake_cloud.pcd")
pcd.rotate(pcd.get_rotation_matrix_from_xyz((0, 0, np.pi / 20)))
create_graphics(pcd)
# noise_pcd = apply_noise(pcd, 0, 0.001)
# rotate_pcd, central, y_dir, angle_z, angle_x, mesh2 = fixed_pcd(pcd)
# mesh1 = rotate_pcd.get_oriented_bounding_box()
# mesh2 = pcd.get_oriented_bounding_box()
# mesh_pcd = create_point_cloud(np.asarray(mesh1.get_box_points()))
# R = mesh_pcd.get_rotation_matrix_from_xyz((1, 0, 0))
# print("R:", R)
# T = np.eye(4)
# T[:3, :3] = mesh_pcd.get_rotation_matrix_from_xyz((0, 1, 0))
# T[0, 3] = 0
# T[1, 3] = 1
# T[2, 3] = 0
#
# xz_planes, candidate_circles = get_xz_mesh_list(rotate_pcd)
# cylindir_pcd, center, radius = circle_tools(candidate_circles, y_dir, central)
# points_mesh = mesh2.get_min_bound()
# tri_mesh = o3d.geometry.TriangleMesh()
# mesh_box = tri_mesh.create_box(width=0.8, height=0.01,
#                            depth=0.8)
# mesh_box.translate((points_mesh[0], points_mesh[1], points_mesh[2]))
# o3d.visualization.draw_geometries([mesh_box, mesh2, pcd])
# o3d.visualization.draw_geometries([rotate_pcd, mesh1, mesh_box])
