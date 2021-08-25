import numpy as np
import open3d as o3d
import copy


def get_yz_mesh_list(data, width_size=0.005, count=5):
    data_arr = np.asarray(data.points)
    x_max, x_min = data_arr[:, 0].max(), data_arr[:, 0].min()
    y_max, y_min = data_arr[:, 1].max(), data_arr[:, 1].min()
    z_max, z_min = data_arr[:, 2].max(), data_arr[:, 2].min()

    lines = np.linspace(x_max - 0.02, x_min + 0.02, count)
    mesh = o3d.geometry.TriangleMesh()
    mesh_box = mesh.create_box(width=width_size, height=abs(abs(y_max) + abs(y_min)) + 0.1,
                               depth=abs(abs(z_max) - abs(z_min)) + 0.1)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    mesh_planes = []
    for i in lines:
        copy_mesh = copy.deepcopy(mesh_box)
        copy_mesh.paint_uniform_color([0.9, 0.1, 1.0])
        copy_mesh.translate((i, y_min - 0.05, z_min - 0.01))
        check_data = get_intersection_mesh(data, copy_mesh)
        if len(check_data.points) >= 30:
            mesh_planes.append(copy_mesh)
    return mesh_planes


def get_xz_mesh_list(data):
    data_arr = np.asarray(data.points)
    x_max, x_min = data_arr[:, 0].max(), data_arr[:, 0].min()
    y_max, y_min = data_arr[:, 1].max(), data_arr[:, 1].min()
    z_max, z_min = data_arr[:, 2].max(), data_arr[:, 2].min()

    lines = np.linspace(y_max - 0.05, y_min + 0.05, 5)
    mesh = o3d.geometry.TriangleMesh()
    mesh_box = mesh.create_box(width=abs(abs(x_max) + abs(x_min)) + 0.1, height=0.01,
                               depth=abs(abs(z_max) - abs(z_min)) + 0.1)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    mesh_planes = []
    for i in lines:
        copy_mesh = copy.deepcopy(mesh_box)
        copy_mesh.paint_uniform_color([0.9, 0.1, 1.0])
        copy_mesh.translate((x_min - 0.05, i, z_min - 0.01))
        mesh_planes.append(copy_mesh)
    return mesh_planes


def get_xy_mesh_list(data, depth_size=0.01):
    data_arr = np.asarray(data.points)
    x_max, x_min = data_arr[:, 0].max(), data_arr[:, 0].min()
    y_max, y_min = data_arr[:, 1].max(), data_arr[:, 1].min()
    z_max, z_min = data_arr[:, 2].max(), data_arr[:, 2].min()
    lines = np.linspace(z_max - 0.05, z_min + 0.05, 5)
    mesh = o3d.geometry.TriangleMesh()
    mesh_box = mesh.create_box(width=0.5, height=abs(y_max - y_min)+0.1,
                               depth=depth_size)

    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    mesh_box.translate((x_min - 0.05, y_min - 0.08, lines[0]))
    pcd_witout = get_intersection_mesh(data, mesh_box)
    if len(pcd_witout.points) <= 3:
        o3d.visualization.draw_geometries([data, mesh_box])
        mesh_box, pcd_witout = get_xy_mesh_list(data, depth_size*2)
    return mesh_box, pcd_witout


def get_intersection_mesh(data, mesh_box):
    copy_pcd = copy.deepcopy(data)
    bbox = mesh_box.get_oriented_bounding_box()
    pcd_witout = copy_pcd.crop(bbox)
    return pcd_witout

