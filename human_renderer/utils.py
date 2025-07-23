import os
import numpy as np
import trimesh
from PIL import Image
from pyglet.resource import texture

from human_renderer.renderer.mesh import load_file2mesh, load_file2info, compute_tangent


def load_textured_mesh(path2mesh, filename, image_only=False):
    """
    Load textured mesh
    Initially, find filename.bmp/jpg/tif/jpeg/png as a texture map.
     > if there are multiple images, take the first one.
     > otherwise, find material_0.bmp/jpe/tif/jpeg/png as a texture map.
    :param path2mesh: path to the textured mesh (.obj file only)
    :param filename: name of the current mesh
    :return: mesh with texture (texture will not be defined, if the texture map does not exist)
    """
    exts = ['.tif', '.bmp', '.jpg', '.jpeg', '.png', '_0.png']
    # text_file = os.path.join(self.path2obj, filename, filename)
    text_file = [path2mesh.replace('.obj', ext) for ext in exts
                 if os.path.isfile(path2mesh.replace('.obj', ext))]
    if len(text_file) == 0:
        # text_file = os.path.join(self.path2obj, filename, 'material_0')
        text_file = path2mesh.replace(filename + '.obj', 'material_0')
        text_file = [text_file + ext for ext in exts if os.path.isfile(text_file + ext)]

    obj = path2mesh.split('/')[-1]
    if len(text_file) == 0:
        text_file = path2mesh.replace(obj, 'material_0.jpeg')
        if os.path.isfile(text_file):
            text_file = [text_file]

    # for RP_T dataset
    if len(text_file) == 0:
        obj = path2mesh.split('/')[-1]
        text_file = os.path.join(path2mesh.replace(obj, ''), 'tex', filename.replace('FBX', 'dif.jpg'))
        if os.path.isfile(text_file):
            text_file = [text_file]
    if len(text_file) > 0 and os.path.isfile(text_file[0]):
        im = Image.open(text_file[0])
        texture_image = np.array(im)
    else:
        texture_image = None

    if image_only:
        return texture_image
    else:
        if texture_image is not None:
            mesh = load_file2mesh(path2mesh, texture=texture_image)
        else:
            mesh = trimesh.load_mesh(path2mesh)  # normal mesh
        return mesh, texture_image
