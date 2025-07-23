import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


class VisOpen3D:
    def __init__(self, width=1920, height=1080, visible=False):
        self.__vis = o3d.visualization.Visualizer()
        self.__vis.create_window(width=width, height=height, visible=visible)
        self.__width = width
        self.__height = height
        self.opt = self.__vis.get_render_option()
        self.opt.background_color = np.asarray([0.0, 0.0, 0.0])
        
        if visible:
            self.poll_events()
            self.update_renderer()

    def __del__(self):
        self.__vis.destroy_window()

    def render(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()

    def poll_events(self):
        self.__vis.poll_events()

    def update_renderer(self):
        self.__vis.update_renderer()

    def run(self):
        self.__vis.run()

    def destroy_window(self):
        self.__vis.destroy_window()

    def add_geometry(self, data):
        self.__vis.add_geometry(data)

    def remove_geometry(self, data):
        self.__vis.remove_geometry(data)

    def update_view_point(self, intrinsic, extrinsic):
        ctr = self.__vis.get_view_control()
        param = self.convert_to_open3d_param(intrinsic, extrinsic)
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        self.__vis.update_renderer()

    def get_view_point_intrinsics(self):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        intrinsic = param.intrinsic.intrinsic_matrix
        return intrinsic

    def get_view_point_extrinsics(self):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        extrinsic = param.extrinsic
        return extrinsic

    def get_view_control(self):
        return self.__vis.get_view_control()

    def save_view_point(self, filename):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(filename, param)

    def load_view_point(self, filename):
        param = o3d.io.read_pinhole_camera_parameters(filename)
        intrinsic = param.intrinsic.intrinsic_matrix
        extrinsic = param.extrinsic
        self.update_view_point(intrinsic, extrinsic)

    def convert_to_open3d_param(self, intrinsic, extrinsic):
        param = o3d.camera.PinholeCameraParameters()
        param.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        param.intrinsic.intrinsic_matrix = intrinsic
        param.extrinsic = extrinsic
        return param

    def capture_screen_float_buffer(self, show=False):
        image = self.__vis.capture_screen_float_buffer(do_render=True)

        if show:
            plt.imshow(image)
            plt.show()

        return image

    def capture_screen_image(self, filename):
        self.__vis.capture_screen_image(filename, do_render=True)

    def capture_depth_float_buffer(self, show=False):
        depth = self.__vis.capture_depth_float_buffer(do_render=True)

        if show:
            plt.imshow(depth)
            plt.show()

        return depth

    def capture_depth_image(self, filename):
        self.__vis.capture_depth_image(filename, do_render=True)

        # to read the saved depth image file use:
        # depth = open3d.io.read_image(filename)
        # plt.imshow(depth)
        # plt.show()

    def draw_camera(self, intrinsic, extrinsic, scale=1, color=None):
        # intrinsics
        K = intrinsic

        # convert extrinsics matrix to rotation and translation matrix
        extrinsic = np.linalg.inv(extrinsic)
        R = extrinsic[0:3,0:3]
        t = extrinsic[0:3,3]

        width = self.__width
        height = self.__height

        geometries = draw_camera(K, R, t, width, height, scale, color)
        for g in geometries:
            self.add_geometry(g)

    def draw_points3D(self, points3D, color=None):
        geometries = draw_points3D(points3D, color)
        for g in geometries:
            self.add_geometry(g)

#
# Auxiliary funcions
#
def draw_camera(K, R, t, width, height, scale=1, color=None):
    """ Create axis, plane and pyramid geometries in Open3D format
    :   param K     : calibration matrix (camera intrinsics)
    :   param R     : rotation matrix
    :   param t     : translation
    :   param width : image width
    :   param height: image height
    :   param scale : camera model scale
    :   param color : color of the image plane and pyramid lines
    :   return      : camera model geometries (axis, plane and pyramid)
    """

    # default color
    if color is None:
        color = [0.8, 0.2, 0.8]

    # camera model scale
    s = 1 / scale

    # intrinsics
    Ks = np.array([[K[0, 0] * s,            0, K[0,2]],
                   [          0,  K[1, 1] * s, K[1,2]],
                   [          0,            0, K[2,2]]])
    Kinv = np.linalg.inv(Ks)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    axis = create_coordinate_frame(T, scale=scale*0.5)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [width, 0, 1],
        [0, height, 1],
        [width, height, 1],
    ]

    # pixel to camera coordinate system
    points = [scale * Kinv @ p for p in points_pixel]

    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = o3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.transform(T)
    plane.translate(R @ [points[1][0], points[1][1], scale])

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_in_world),
        lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # return as list in Open3D format
    return [axis, plane, line_set]


def create_coordinate_frame(T, scale=0.25):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    frame.transform(T)
    return frame


def draw_points3D(points3D, color=None):
    # color: default value
    if color is None:
        color = [0.8, 0.2, 0.8]

    geometries = []
    for pt in points3D:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=20)
        sphere.translate(pt)
        sphere.paint_uniform_color(np.array(color))
        geometries.append(sphere)

    return geometries
