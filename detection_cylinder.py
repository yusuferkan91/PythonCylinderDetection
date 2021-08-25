import open3d as o3d
import numpy as np
import tools_plane as plane
import tools_cylinder as cylinder
import tools_line as line
import copy


class Cylinder_detection:
    def __init__(self, path=None, pcd=None):
        if path is None:
            self.point_cloud = copy.deepcopy(pcd)
        elif pcd is None:
            self.point_cloud = o3d.io.read_point_cloud(path)
        else:
            print("Point cloud yada path girin")

        """datayi kesmek icin y-z duzlemine paralel duzlemler olusturma"""
        self.yz_planes = plane.get_yz_mesh_list(self.point_cloud)
        """y-z duzlemi ile datanın kesisen noktalarini tespit etme"""
        self.candidate_line = [plane.get_intersection_mesh(self.point_cloud, mesh) for mesh in self.yz_planes]

        err_ind = 100
        """kesisen noktalarda dogru parcasi fit etme (for dongusunde ataniyor) hata orani en dusuk dogru parcasi 
        secilir. Dogru parcasi ile z ekseni arasinda kalan aci for icinde hesaplanir"""
        self.best_line = None
        self.best_line_direction = None
        points = np.asarray(self.point_cloud.points)
        self.max_y = points[points[:, 1] == points[:, 1].max()][0]
        self.min_y = points[points[:, 1] == points[:, 1].min()][0]
        self.best_angle = 0
        for i in self.candidate_line:
            fit_line, direction, max_y, min_y = line.fit(i)

            if fit_line is not None:
                err = line.compute_error(i, fit_line)

                if err < err_ind:
                    err_ind = err
                    self.best_line = fit_line
                    self.best_line_direction = direction
                    self.best_angle = line.get_angle(self.best_line_direction, (0, 0, 1), i, True)

        """Tespit edilen aci kadar dondurme islemi yapilir"""
        self.point_cloud.rotate(self.point_cloud.get_rotation_matrix_from_xyz((self.best_angle, 0, 0)))
        self.best_line.rotate(self.best_line.get_rotation_matrix_from_xyz((self.best_angle, 0, 0)))

        points = np.asarray(self.point_cloud.points)
        self.max_x = points[points[:, 0] == points[:, 0].max()][0]
        self.min_x = points[points[:, 0] == points[:, 0].min()][0]
        center_x = (self.max_x[0] + self.min_x[0]) / 2
        crop_x = points[np.where(points[:, 0] < center_x)]
        crop_pcd = cylinder.create_point_cloud(crop_x)
        """Diger aciyi hesaplamak icin xy duzlemine paralel duzlemler olusturulur. """
        self.xy_planes, self.candidate_line_h = plane.get_xy_mesh_list(crop_pcd)
        self.fit_line_h, direction, _, __ = line.fit(self.candidate_line_h)
        self.best_angle_h = line.get_angle(direction, (1, 0, 0), self.candidate_line_h, False)
        """Tespit edilen aci kadar dondurme islemi yapilir."""
        self.point_cloud.rotate(self.point_cloud.get_rotation_matrix_from_xyz((0, 0, self.best_angle_h)))
        """Cember fit etmek icin xz duzlemine paralel duzlemler olusturulur"""
        self.xz_planes = plane.get_xz_mesh_list(self.point_cloud)
        self.candidate_circle = [plane.get_intersection_mesh(self.point_cloud, mesh) for mesh in self.xz_planes]
        """ideal cember, yaricap ve merkez noktası for icinde bulunur."""
        self.best_circle = None
        self.best_circle_radius = None
        self.best_circle_center = None
        err_ind = 100
        for i in self.candidate_circle:
            cl, ind1 = i.remove_radius_outlier(nb_points=5, radius=0.05)
            i = i.select_by_index(ind1)
            points_circle = np.asarray(i.points)
            if points_circle.shape[0] > 10:
                fit_circle, circle_radius, C = cylinder.fit_circle(points_circle)
                pcd_circle = cylinder.create_point_cloud(fit_circle)
                err = line.compute_error(i, pcd_circle)
                if err < err_ind:
                    err_ind = err
                    self.best_circle = pcd_circle
                    self.best_circle_radius = circle_radius
                    self.best_circle_center = C
        """Silindir olusturulur"""
        self.point_cloud.paint_uniform_color([0.9, 0.1, 0.1])
        x_min, x_max, x_center = cylinder.get_ref_points(np.array(self.best_circle.points))
        result = cylinder.compute_M_R(x_min, x_max, x_center)
        n = np.linspace(self.max_y[1] + 0.01, self.min_y[1] - 0.05, 50)
        self.cylinder, _ = cylinder.get_circle(result[0], result[1], result[2], n)
        self.cylinder.paint_uniform_color([0.7, 0.7, 0.7])





