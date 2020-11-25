import math
import random
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def phi(x, y, t, a1=1, a2=1, alpha=1e-4):
    u = max(x/a1, y/a2)**2
    v = min(x/a1, y/a2)**2
    if t < alpha:
        return u * (((1 + (v/u)**(1/alpha))**alpha - 1) * t / alpha + 1)
    else:
        return u * (1 + (v/u)**(1/t))**t


def p(theta, a1, a2, eps):
    phi_tmp = phi(np.abs(np.cos(theta)), np.abs(np.sin(theta)), eps, a1, a2)
    return np.array([np.cos(theta), np.sin(theta)]) / np.sqrt(phi_tmp)


def get_split_angle(ax, ay, eps, alpha, beta):
    return get_split_angle_bis(ax, ay, eps, alpha, beta, alpha, beta)


def get_split_angle_bis(a1, a2, eps, alpha, beta, la, lb):
    theta = (la+lb)/2
    pb = p(beta, a1, a2, eps)
    pt = p(theta, a1, a2, eps)
    pa = p(alpha, a1, a2, eps)
    norm1 = np.linalg.norm(pt - pb)
    norm2 = np.linalg.norm(pa - pt)

    if abs(norm1 - norm2) < 1e-4:
        return theta
    elif norm1 > norm2:
        return get_split_angle_bis(a1, a2, eps, alpha, beta, theta, lb)
    else:
        return get_split_angle_bis(a1, a2, eps, alpha, beta, la, theta)


class SuperellipseSampler:

    def __init__(self, a1, a2, eps, delta, d_max):
        self.a1 = a1
        self.a2 = a2
        self.eps = eps
        self.d_max = d_max
        self.delta = delta

    def sample_se(self, alpha, beta):
        I = np.array([np.pi, np.arctan2(self.a2, -self.a1), np.pi/2, np.arctan2(self.a2, self.a1), 0,
                      np.arctan2(-self.a2, self.a1), -np.pi/2, np.arctan2(-self.a2, -self.a1), -np.pi])

        I = I[np.logical_and(I > alpha, I < beta)]
        I = np.concatenate(([beta], I, [alpha]))
        P = []
        for i in range(1, I.shape[0]):
            P.append(p(I[i-1], self.a1, self.a2, self.eps))
            P += self.sample_segment(I[i-1], I[i], 0)
        P.append(p(I[-1], self.a1, self.a2, self.eps))
        return P

    def sample_segment(self, alpha, beta, d):
        pb = p(beta, self.a1, self.a2, self.eps)
        pa = p(alpha, self.a1, self.a2, self.eps)
        norm = np.linalg.norm(pb - pa)
        if d > self.d_max or norm < self.delta:
            return []

        theta = get_split_angle(self.a1, self.a2, self.eps, alpha, beta)
        P = []
        P += self.sample_segment(alpha, theta, d+1)
        P.append(p(theta, self.a1, self.a2, self.eps))
        P += self.sample_segment(theta, beta, d+1)
        return P


def fexp(x, p):
    """a different kind of exponentiation"""
    return (np.sign(x) * (np.abs(x) ** p))


def tens_fld(A, B, C, P, Q, x0=0, y0=0, z0=0):
    """this module plots superquadratic surfaces with the given parameters"""
    phi, theta = np.mgrid[0:np.pi:80j, 0:2 * np.pi:80j]
    x = A * (fexp(np.sin(phi), P)) * (fexp(np.cos(theta), Q)) + x0
    y = B * (fexp(np.sin(phi), P)) * (fexp(np.sin(theta), Q)) + y0
    z = C * (fexp(np.cos(phi), P)) + z0
    return x, y, z

def sample_sq(ax, ay, az, eps1, eps2, delta, d_max, ignore_center=False):

    se_xy_samp = SuperellipseSampler(ax, ay, eps2, delta, d_max)
    se_z_samp = SuperellipseSampler(1, az, eps1, delta, d_max)

    se_xy = se_xy_samp.sample_se(-np.pi, np.pi)
    diff = np.tan(delta/az/4) if ignore_center else 0
    se_z = se_z_samp.sample_se(-np.pi/2+diff, np.pi/2-diff)

    se_xy = np.array(se_xy)
    se_z = np.array(se_z)

    # Spherical product
    X = np.outer(se_xy[:, 0], se_z[:, 0])
    Y = np.outer(se_xy[:, 1], se_z[:, 0])
    Z = np.outer(np.ones(se_xy.shape[0]), se_z[:, 1])
    return X, Y, Z


class Superquadric:
    def __init__(self, a1, a2, a3, e1, e2, x0=0, y0=0, z0=0, dimension_max=0):
        """
        :param a1: size x
        :param a2: size y
        :param a3: size z
        :param e1: shape param1
        :param e2: shape param2
        :param x0: offset x
        :param y0: offset y
        :param z0: offset z
        """
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.e1 = e1
        self.e2 = e2
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.dimension_max = dimension_max

    def get_z_partial_param(self, z):
        z_partial = (1 - (z/self.a3)**(2/self.e1))**(self.e2/self.e1)
        return z_partial

    def get_x_value(self, y, z):
        z_partial = self.get_z_partial_param(z)
        a = (z_partial - (y/self.a2)**(2/self.e2))**self.e2
        if a < 0:
            return None
        x = self.a1 * math.sqrt(a)
        return x

    def get_y_value(self, x, z):
        z_partial = self.get_z_partial_param(z)
        a = (z_partial - (x/self.a1)**(2/self.e2))
        a = float(a)**self.e2
        if isinstance(a, complex):
            return None
        if a < 0:
            return None
        y = self.a2 * math.sqrt(a)
        return y

    def get_z_value(self, x, y):
        a = ((x/self.a1) ** (2/self.e2) + (y/self.a2) ** (2/self.e2)) ** (self.e2/self.e1)
        if a > 1:
            return None
        z = self.a3 * math.sqrt((1 - a) ** self.e1)
        return z

    def get_grid(self, sample_size, include_normals=True, draw_grid=False):
        normals = None
        x, y, z = sample_sq(ax=self.a1, ay=self.a2, az=self.a3, eps1=self.e1, eps2=self.e2, delta=0.5, d_max=10, ignore_center=False)
        x += self.x0
        y += self.y0
        z += self.z0
        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)
        points = list(map(list, zip(x.flatten(), y.flatten(), z.flatten())))
        sampled_indices = random.sample(range(len(points)), sample_size)
        if include_normals:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals()
            normals = list(np.asarray(pcd.normals))
        if draw_grid:
            o3d.visualization.draw_geometries([pcd], point_show_normal=True)
            # c = [a[i] for i in b]
        sampled_points = [points[i] for i in sampled_indices]
        sampled_normals = [normals[i] for i in sampled_indices]
        return sampled_points, sampled_normals

    @staticmethod
    def draw_point_cloud(xyz_tuples):
        fig = plt.figure(figsize=plt.figaspect(1))
        ax = fig.add_subplot(111, projection='3d')
        vals_x = [item[0] for item in xyz_tuples]
        vals_y = [item[1] for item in xyz_tuples]
        vals_z = [item[2] for item in xyz_tuples]
        ax.scatter(vals_x, vals_y, vals_z)
        plt.show()
        # plt.display()
        plt.waitforbuttonpress()

    @staticmethod
    def get_graph(points):
        pts = np.array(points)
        # Triangulate parameter space to determine the triangles
        tri = ConvexHull(pts)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # plot defining corner points
        ax.plot(pts.T[0], pts.T[1], pts.T[2], "ko")

        for s in tri.simplices:
            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")

        # Make axis label
        for i in ["x", "y", "z"]:
            eval("ax.set_{:s}label('{:s}')".format(i, i))

        plt.show()
        plt.waitforbuttonpress()