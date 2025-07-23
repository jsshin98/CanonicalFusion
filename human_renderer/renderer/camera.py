import cv2
import numpy as np
from .glm import ortho


class Camera:
    def __init__(self, width=1600, height=1200, projection='orthogonal'):
        # Focal Length
        # equivalent 50mm
        focal = np.sqrt(width * width + height * height)
        self.focal_x = focal
        self.focal_y = focal
        # Principal Point Offset
        self.principal_x = width / 2
        self.principal_y = height / 2
        # Axis Skew
        self.skew = 0
        # Image Size
        self.width = width
        self.height = height
        self.mode = projection
        self.K = None
        self.R = None
        self.t = None

        if self.mode == 'perspective':
            self.near = 1
            self.far = 600
            self.center = np.array([0, 0, 300.0])
            # self.near = 1
            # self.far = 600
            # self.center = np.array([0, 0, 300.0])
            self.ortho_ratio = None
        else:
            self.near = -128
            self.far = 128
            self.center = np.array([0, 0, 2.0])
            self.ortho_ratio = 0.4 * (512 / width)

        # Camera Center
        self.direction = np.array([0, 0, -1])
        self.right = np.array([1, 0, 0])
        self.up = np.array([0, 1, 0])

    def sanity_check(self):
        self.center = self.center#self.center.reshape([-1])
        self.direction = self.direction.reshape([-1])
        self.right = self.right.reshape([-1])
        self.up = self.up.reshape([-1])

        assert len(self.center) == 3
        assert len(self.direction) == 3
        assert len(self.right) == 3
        assert len(self.up) == 3

    @staticmethod
    def normalize_vector(v):
        v_norm = np.linalg.norm(v)
        return v if v_norm == 0 else v / v_norm

    def get_real_z_value(self, z):
        z_near = self.near
        z_far = self.far
        z_n = 2.0 * z - 1.0
        z_e = 2.0 * z_near * z_far / (z_far + z_near - z_n * (z_far - z_near))
        return z_e

    def get_rotation_matrix(self):
        rot_mat = np.eye(3)
        s = self.right
        s = self.normalize_vector(s)
        rot_mat[0, :] = s
        u = self.up
        u = self.normalize_vector(u)
        rot_mat[1, :] = -u
        rot_mat[2, :] = self.normalize_vector(self.direction)

        return rot_mat

    def get_translation_vector(self):
        rot_mat = self.get_rotation_matrix()
        trans = -np.dot(rot_mat, self.center)
        return trans

    def get_intrinsic_matrix(self):
        int_mat = np.eye(3)

        int_mat[0, 0] = self.focal_x
        int_mat[1, 1] = self.focal_y
        int_mat[0, 1] = self.skew
        int_mat[0, 2] = self.principal_x
        int_mat[1, 2] = self.principal_y

        return int_mat

    def get_projection_matrix(self):
        ext_mat = self.get_extrinsic_matrix()
        int_mat = self.get_intrinsic_matrix()

        return np.matmul(int_mat, ext_mat)

    def get_extrinsic_matrix(self):
        rot_mat = self.get_rotation_matrix()
        int_mat = self.get_intrinsic_matrix()
        trans = self.get_translation_vector()

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rot_mat
        extrinsic[:3, 3] = trans

        return extrinsic[:3, :]

    def set_rotation_matrix(self, rot_mat):
        self.direction = rot_mat[2, :]
        self.up = -rot_mat[1, :]
        self.right = rot_mat[0, :]

    def set_intrinsic_matrix(self, int_mat):
        self.focal_x = int_mat[0, 0]
        self.focal_y = int_mat[1, 1]
        self.skew = int_mat[0, 1]
        self.principal_x = int_mat[0, 2]
        self.principal_y = int_mat[1, 2]

    def set_projection_matrix(self, proj_mat):
        res = cv2.decomposeProjectionMatrix(proj_mat)
        int_mat, rot_mat, camera_center_homo = res[0], res[1], res[2]
        camera_center = camera_center_homo[0:3] / camera_center_homo[3]
        camera_center = camera_center.reshape(-1)
        int_mat = int_mat / int_mat[2][2]

        self.set_intrinsic_matrix(int_mat)
        self.set_rotation_matrix(rot_mat)
        self.center = camera_center

        self.sanity_check()

    def get_gl_matrix(self):
        z_near = self.near
        z_far = self.far
        rot_mat = self.get_rotation_matrix()
        int_mat = self.get_intrinsic_matrix()
        trans = self.get_translation_vector()

        self.K = int_mat
        self.R = rot_mat
        self.t = trans

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rot_mat
        extrinsic[:3, 3] = trans#[2, 0:3]
        axis_adj = np.eye(4)
        axis_adj[2, 2] = -1
        axis_adj[1, 1] = -1
        model_view = np.matmul(axis_adj, extrinsic)

        projective = np.zeros([4, 4])
        projective[:2, :2] = int_mat[:2, :2]
        projective[:2, 2:3] = -int_mat[:2, 2:3]
        projective[3, 2] = -1
        projective[2, 2] = (z_near + z_far)
        projective[2, 3] = (z_near * z_far)

        if self.ortho_ratio is None:
            ndc = ortho(0, self.width, 0, self.height, z_near, z_far)
            # ndc = perspective(self.focal_x, 1, z_near, z_far)
            perspective = np.matmul(ndc, projective)
        else:
            perspective = ortho(-self.width * self.ortho_ratio / 2, self.width * self.ortho_ratio / 2,
                                -self.height * self.ortho_ratio / 2, self.height * self.ortho_ratio / 2,
                                z_near, z_far)

        return rot_mat, int_mat, trans, perspective, model_view

def KRT_from_P(proj_mat, normalize_K=True):
    res = cv2.decomposeProjectionMatrix(proj_mat)
    K, Rot, camera_center_homog = res[0], res[1], res[2]
    camera_center = camera_center_homog[0:3] / camera_center_homog[3]
    trans = -Rot.dot(camera_center)
    if normalize_K:
        K = K / K[2][2]
    return K, Rot, trans

def MVP_from_P(proj_mat, width, height, near=0.1, far=10000):
    '''
    Convert OpenCV camera calibration matrix to OpenGL projection and model view matrix
    :param proj_mat: OpenCV camera projeciton matrix
    :param width: Image width
    :param height: Image height
    :param near: Z near value
    :param far: Z far value
    :return: OpenGL projection matrix and model view matrix
    '''
    res = cv2.decomposeProjectionMatrix(proj_mat)
    K, Rot, camera_center_homog = res[0], res[1], res[2]
    camera_center = camera_center_homog[0:3] / camera_center_homog[3]
    trans = -Rot.dot(camera_center)
    K = K / K[2][2]

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rot
    extrinsic[:3, 3:4] = trans
    axis_adj = np.eye(4)
    axis_adj[2, 2] = -1
    axis_adj[1, 1] = -1
    model_view = np.matmul(axis_adj, extrinsic)

    zFar = far
    zNear = near
    projective = np.zeros([4, 4])
    projective[:2, :2] = K[:2, :2]
    projective[:2, 2:3] = -K[:2, 2:3]
    projective[3, 2] = -1
    projective[2, 2] = (zNear + zFar)
    projective[2, 3] = (zNear * zFar)

    ndc = ortho(0, width, 0, height, zNear, zFar)

    perspective = np.matmul(ndc, projective)

    return perspective, model_view

def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R

def rotateSH(SH, R):
    SHn = SH

    # 1st order
    SHn[1] = R[1,1]*SH[1] - R[1,2]*SH[2] + R[1,0]*SH[3]
    SHn[2] = -R[2,1]*SH[1] + R[2,2]*SH[2] - R[2,0]*SH[3]
    SHn[3] = R[0,1]*SH[1] - R[0,2]*SH[2] + R[0,0]*SH[3]

    # 2nd order
    SHn[4:,0] = rotateBand2(SH[4:,0],R)
    SHn[4:,1] = rotateBand2(SH[4:,1],R)
    SHn[4:,2] = rotateBand2(SH[4:,2],R)

    return SHn

def rotateBand2(x, R):
    s_c3 = 0.94617469575
    s_c4 = -0.31539156525
    s_c5 = 0.54627421529

    s_c_scale = 1.0/0.91529123286551084
    s_c_scale_inv = 0.91529123286551084

    s_rc2 = 1.5853309190550713*s_c_scale
    s_c4_div_c3 = s_c4/s_c3
    s_c4_div_c3_x2 = (s_c4/s_c3)*2.0

    s_scale_dst2 = s_c3 * s_c_scale_inv
    s_scale_dst4 = s_c5 * s_c_scale_inv

    sh0 =  x[3] + x[4] + x[4] - x[1]
    sh1 =  x[0] + s_rc2*x[2] +  x[3] + x[4]
    sh2 =  x[0]
    sh3 = -x[3]
    sh4 = -x[1]

    r2x = R[0][0] + R[0][1]
    r2y = R[1][0] + R[1][1]
    r2z = R[2][0] + R[2][1]

    r3x = R[0][0] + R[0][2]
    r3y = R[1][0] + R[1][2]
    r3z = R[2][0] + R[2][2]

    r4x = R[0][1] + R[0][2]
    r4y = R[1][1] + R[1][2]
    r4z = R[2][1] + R[2][2]

    sh0_x = sh0 * R[0][0]
    sh0_y = sh0 * R[1][0]
    d0 = sh0_x * R[1][0]
    d1 = sh0_y * R[2][0]
    d2 = sh0 * (R[2][0] * R[2][0] + s_c4_div_c3)
    d3 = sh0_x * R[2][0]
    d4 = sh0_x * R[0][0] - sh0_y * R[1][0]

    sh1_x = sh1 * R[0][2]
    sh1_y = sh1 * R[1][2]
    d0 += sh1_x * R[1][2]
    d1 += sh1_y * R[2][2]
    d2 += sh1 * (R[2][2] * R[2][2] + s_c4_div_c3)
    d3 += sh1_x * R[2][2]
    d4 += sh1_x * R[0][2] - sh1_y * R[1][2]

    sh2_x = sh2 * r2x
    sh2_y = sh2 * r2y
    d0 += sh2_x * r2y
    d1 += sh2_y * r2z
    d2 += sh2 * (r2z * r2z + s_c4_div_c3_x2)
    d3 += sh2_x * r2z
    d4 += sh2_x * r2x - sh2_y * r2y

    sh3_x = sh3 * r3x
    sh3_y = sh3 * r3y
    d0 += sh3_x * r3y
    d1 += sh3_y * r3z
    d2 += sh3 * (r3z * r3z + s_c4_div_c3_x2)
    d3 += sh3_x * r3z
    d4 += sh3_x * r3x - sh3_y * r3y

    sh4_x = sh4 * r4x
    sh4_y = sh4 * r4y
    d0 += sh4_x * r4y
    d1 += sh4_y * r4z
    d2 += sh4 * (r4z * r4z + s_c4_div_c3_x2)
    d3 += sh4_x * r4z
    d4 += sh4_x * r4x - sh4_y * r4y

    dst = x
    dst[0] = d0
    dst[1] = -d1
    dst[2] = d2 * s_scale_dst2
    dst[3] = -d3
    dst[4] = d4 * s_scale_dst4

    return dst