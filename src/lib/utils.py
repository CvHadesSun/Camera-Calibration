import os
import json
import numpy as np
from os.path import join
import shutil
import cv2


def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def save_json(file, data):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)


save_annot = save_json


def getFileList(root, ext='.jpg'):
    files = []
    dirs = os.listdir(root)
    while len(dirs) > 0:
        path = dirs.pop()
        fullname = join(root, path)
        if os.path.isfile(fullname) and fullname.endswith(ext):
            files.append(path)
        elif os.path.isdir(fullname):
            for s in os.listdir(fullname):
                newDir = join(path, s)
                dirs.append(newDir)
    files = sorted(files)
    return files


def load_annot_to_tmp(annotname):
    if not os.path.exists(annotname):
        dirname = os.path.dirname(annotname)
        os.makedirs(dirname, exist_ok=True)
        shutil.copy(annotname.replace('_tmp', ''), annotname)
    annot = read_json(annotname)
    return annot


#
def _findChessboardCorners(img, pattern):
    "basic function"
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    retval, corners = cv2.findChessboardCorners(img, pattern,
                                                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS)
    if not retval:
        return False, None
    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
    corners = corners.squeeze()
    return True, corners


def _findChessboardCornersAdapt(img, pattern):
    "Adapt mode"
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, 21, 2)
    return _findChessboardCorners(img, pattern)


def findChessboardCorners(img, annots, pattern):
    if annots['visited']:
        return None
    annots['visited'] = True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    for func in [_findChessboardCornersAdapt, _findChessboardCorners]:
        ret, corners = func(gray, pattern)
        if ret: break
    else:
        return None
    # found the corners
    show = img.copy()
    show = cv2.drawChessboardCorners(show, pattern, corners, ret)
    assert corners.shape[0] == len(annots['keypoints2d'])
    corners = np.hstack((corners, np.ones((corners.shape[0], 1))))
    annots['keypoints2d'] = corners.tolist()
    return show


colors_chessboard_bar = [
    [0, 0, 255],
    [0, 128, 255],
    [0, 200, 200],
    [0, 255, 0],
    [200, 200, 0],
    [255, 0, 0],
    [255, 0, 250]
]


def get_lines_chessboard(pattern=(9, 6)):
    w, h = pattern[0], pattern[1]
    lines = []
    lines_cols = []
    for i in range(w * h - 1):
        lines.append([i, i + 1])
        lines_cols.append(colors_chessboard_bar[i // w])
    return lines, lines_cols


import time
import tabulate


class Timer:
    records = {}
    tmp = None

    @classmethod
    def tic(cls):
        cls.tmp = time.time()

    @classmethod
    def toc(cls):
        res = (time.time() - cls.tmp) * 1000
        cls.tmp = None
        return res

    @classmethod
    def report(cls):
        header = ['', 'Time(ms)']
        contents = []
        for key, val in cls.records.items():
            contents.append(['{:20s}'.format(key), '{:.2f}'.format(sum(val) / len(val))])
        print(tabulate.tabulate(contents, header, tablefmt='fancy_grid'))

    def __init__(self, name, silent=False):
        self.name = name
        self.silent = silent
        if name not in Timer.records.keys():
            Timer.records[name] = []

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        end = time.time()
        Timer.records[self.name].append((end - self.start) * 1000)
        if not self.silent:
            t = (end - self.start) * 1000
            if t > 1000:
                print('-> [{:20s}]: {:5.1f}s'.format(self.name, t / 1000))
            elif t > 1e3 * 60 * 60:
                print('-> [{:20s}]: {:5.1f}min'.format(self.name, t / 1e3 / 60))
            else:
                print('-> [{:20s}]: {:5.1f}ms'.format(self.name, (end - self.start) * 1000))


#
class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        else:
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)

    def __del__(self):
        cv2.FileStorage.release(self.fs)

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            cv2.FileStorage.write(self.fs, key, value)
        elif dt == 'list':
            if self.major_version == 4:  # 4.4
                self.fs.startWriteStruct(key, cv2.FileNode_SEQ)
                for elem in value:
                    self.fs.write('', elem)
                self.fs.endWriteStruct()
            else:  # 3.4
                self.fs.write(key, '[')
                for elem in value:
                    self.fs.write('none', elem)
                self.fs.write('none', ']')

    def read(self, key, dt='mat'):
        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)


def read_intri(intri_name):
    assert os.path.exists(intri_name), intri_name
    intri = FileStorage(intri_name)
    camnames = intri.read('names', dt='list')
    cameras = {}
    for key in camnames:
        cam = {}
        cam['K'] = intri.read('K_{}'.format(key))
        cam['invK'] = np.linalg.inv(cam['K'])
        cam['dist'] = intri.read('dist_{}'.format(key))
        cameras[key] = cam
    return cameras


def write_intri(intri_name, cameras):
    intri = FileStorage(intri_name, True)
    results = {}
    camnames = list(cameras.keys())
    intri.write('names', camnames, 'list')
    for key_, val in cameras.items():
        key = key_.split('.')[0]
        K, dist = val['K'], val['dist']
        assert K.shape == (3, 3), K.shape
        assert dist.shape == (1, 5) or dist.shape == (5, 1), dist.shape
        intri.write('K_{}'.format(key), K)
        intri.write('dist_{}'.format(key), dist.reshape(1, 5))


def write_extri(extri_name, cameras):
    extri = FileStorage(extri_name, True)
    results = {}
    camnames = list(cameras.keys())
    extri.write('names', camnames, 'list')
    for key_, val in cameras.items():
        key = key_.split('.')[0]
        extri.write('R_{}'.format(key), val['Rvec'])
        extri.write('Rot_{}'.format(key), val['R'])
        extri.write('T_{}'.format(key), val['T'])
    return 0


def read_camera(intri_name, extri_name, cam_names=[]):
    assert os.path.exists(intri_name), intri_name
    assert os.path.exists(extri_name), extri_name

    intri = FileStorage(intri_name)
    extri = FileStorage(extri_name)
    cams, P = {}, {}
    cam_names = intri.read('names', dt='list')
    for cam in cam_names:
        # 内参只读子码流的
        cams[cam] = {}
        cams[cam]['K'] = intri.read('K_{}'.format(cam))
        cams[cam]['invK'] = np.linalg.inv(cams[cam]['K'])
        Rvec = extri.read('R_{}'.format(cam))
        Tvec = extri.read('T_{}'.format(cam))
        R = cv2.Rodrigues(Rvec)[0]
        RT = np.hstack((R, Tvec))

        cams[cam]['RT'] = RT
        cams[cam]['R'] = R
        cams[cam]['T'] = Tvec
        P[cam] = cams[cam]['K'] @ cams[cam]['RT']
        cams[cam]['P'] = P[cam]

        cams[cam]['dist'] = intri.read('dist_{}'.format(cam))
    cams['basenames'] = cam_names
    return cams


def write_camera(camera, path):
    from os.path import join
    intri_name = join(path, 'intri.yml')
    extri_name = join(path, 'extri.yml')
    intri = FileStorage(intri_name, True)
    extri = FileStorage(extri_name, True)
    results = {}
    camnames = [key_.split('.')[0] for key_ in camera.keys()]
    intri.write('names', camnames, 'list')
    extri.write('names', camnames, 'list')
    for key_, val in camera.items():
        if key_ == 'basenames':
            continue
        key = key_.split('.')[0]
        intri.write('K_{}'.format(key), val['K'])
        intri.write('dist_{}'.format(key), val['dist'])
        if 'Rvec' not in val.keys():
            val['Rvec'] = cv2.Rodrigues(val['R'])[0]
        extri.write('R_{}'.format(key), val['Rvec'])
        extri.write('Rot_{}'.format(key), val['R'])
        extri.write('T_{}'.format(key), val['T'])


def camera_from_img(img):
    height, width = img.shape[0], img.shape[1]
    # focal = 1.2*max(height, width) # as colmap
    focal = 1.2 * min(height, width)  # as colmap
    K = np.array([focal, 0., width / 2, 0., focal, height / 2, 0., 0., 1.]).reshape(3, 3)
    camera = {'K': K, 'R': np.eye(3), 'T': np.zeros((3, 1)), 'dist': np.zeros((1, 5))}
    return camera


class Undistort:
    @staticmethod
    def image(frame, K, dist):
        return cv2.undistort(frame, K, dist, None)

    @staticmethod
    def points(keypoints, K, dist):
        # keypoints: (N, 3)
        assert len(keypoints.shape) == 2, keypoints.shape
        kpts = keypoints[:, None, :2]
        kpts = np.ascontiguousarray(kpts)
        kpts = cv2.undistortPoints(kpts, K, dist, P=K)
        keypoints[:, :2] = kpts[:, 0]
        return keypoints

    @staticmethod
    def bbox(bbox, K, dist):
        keypoints = np.array([[bbox[0], bbox[1], 1], [bbox[2], bbox[3], 1]])
        kpts = Undistort.points(keypoints, K, dist)
        bbox = np.array([kpts[0, 0], kpts[0, 1], kpts[1, 0], kpts[1, 1], bbox[4]])
        return bbox


def undistort(camera, frame=None, keypoints=None, output=None, bbox=None):
    # bbox: 1, 7
    print('This function is deprecated')
    raise NotImplementedError


def get_fundamental_matrix(cameras, basenames):
    skew_op = lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    fundamental_op = lambda K_0, R_0, T_0, K_1, R_1, T_1: np.linalg.inv(K_0).T @ (
            R_0 @ R_1.T) @ K_1.T @ skew_op(K_1 @ R_1 @ R_0.T @ (T_0 - R_0 @ R_1.T @ T_1))
    fundamental_RT_op = lambda K_0, RT_0, K_1, RT_1: fundamental_op(K_0, RT_0[:, :3], RT_0[:, 3], K_1,
                                                                    RT_1[:, :3], RT_1[:, 3])
    F = np.zeros((len(basenames), len(basenames), 3, 3))  # N x N x 3 x 3 matrix
    F = {(icam, jcam): np.zeros((3, 3)) for jcam in basenames for icam in basenames}
    for icam in basenames:
        for jcam in basenames:
            F[(icam, jcam)] += fundamental_RT_op(cameras[icam]['K'], cameras[icam]['RT'], cameras[jcam]['K'],
                                                 cameras[jcam]['RT'])
            if F[(icam, jcam)].sum() == 0:
                F[(icam, jcam)] += 1e-12  # to avoid nan
    return F


def undistortion(image, cam_intrinstic, distof):
    '''to undistorte the image distorted.'''
    undist_image = cv2.undistort(image, cam_intrinstic, distof)
    return undist_image


def read_camera_params(yml_file):
    import yaml
    cam_dict = {}
    f = open(yml_file,'rb')
    cam_info = yaml.load(f)
    print(cam_info)
    cam_dict = \
    {
        "name": cam_info.name,
        "K": cam_info.K_01.data,
        "dist": cam_info.dist_01.data
    }

    return cam_dict