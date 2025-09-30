import json
from ptsemseg.loader.cityscapes_loader import cityscapesLoader
from ptsemseg.loader.cityscapesSR_loader import cityscapesSRLoader
from ptsemseg.loader.image_folder_loader import folderLoader


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "cityscapesSR": cityscapesSRLoader,
        "folder": folderLoader,
    }[name]
