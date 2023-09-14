import configparser
import os


import numpy as np

import gmsh
import networkx as nx

VOX_SIZE = 1.0


def compute_loops(entitiesUp, entitiesDown):
    """
    Retrieve loops of connected entities, e.g. closed surface loops defining the boundaries of different volumes.
    """
    g = nx.Graph()
    for ent in entitiesUp:
        tag = ent[1]
        g.add_node(tag)

    for ent in entitiesDown:
        up, down = gmsh.model.getAdjacencies(ent[0], ent[1])
        g.add_edge(up[0], up[1], boundary=ent[1])

    loops = []
    for comp in nx.connected_components(g):
        loops.append(list(comp))

    return loops


def create_exterior_surfaces(bopos=5.0, topos=265.0):
    """
    Identify triangle elements at the bottom and top surfaces of the mesh based on node coordinates.
    """
    # get triangle elements (Type=2)
    elementTags, nodeTags = gmsh.model.mesh.getElementsByType(2)
    el2Dnodes = np.reshape(nodeTags, (-1, 3))

    # retrieve surface nodes of the strut:
    nodeTags, nodeCoords = gmsh.model.mesh.getNodesForPhysicalGroup(dim=2, tag=10000)
    coords = np.reshape(nodeCoords, (-1, 3))

    # retrieve the nodeTags of 2D elements at the bottom surface:
    elBot = set()
    nodesBot = []
    for coord in coords[np.where(coords[:, 0] < bopos * VOX_SIZE)]:
        elTags = gmsh.model.mesh.getElementsByCoordinates(
            coord[0], coord[1], coord[2], dim=2, strict=True
        )
        for el in elTags:
            elBot.add(el)
    for el in list(elBot):
        tmp = el2Dnodes[np.where(elementTags == el)]
        nodesBot.extend(list(tmp[0]))
    print("Info: ", len(elBot), "elements on bottom surface.")

    # retrieve the nodeTags of 2D elements at the top surface:
    elTop = set()
    nodesTop = []
    for coord in coords[np.where(coords[:, 0] > topos * VOX_SIZE)]:
        elTags = gmsh.model.mesh.getElementsByCoordinates(
            coord[0], coord[1], coord[2], dim=2, strict=True
        )
        for el in elTags:
            elTop.add(el)
    for el in list(elTop):
        tmp = el2Dnodes[np.where(elementTags == el)]
        nodesTop.extend(list(tmp[0]))
    print("Info: ", len(elTop), "elements on top surface.")

    physnum = 20001
    gmsh.model.addDiscreteEntity(2, physnum)
    gmsh.model.mesh.addElementsByType(physnum, 2, [], nodesBot)
    gmsh.model.addPhysicalGroup(2, [physnum], physnum, name="bottom")

    physnum = 20002
    gmsh.model.addDiscreteEntity(2, physnum)
    gmsh.model.mesh.addElementsByType(physnum, 2, [], nodesTop)
    gmsh.model.addPhysicalGroup(2, [physnum], physnum, name="top")

    physnum = 1000001
    gmsh.model.addDiscreteEntity(0, physnum)
    gmsh.model.mesh.addElementsByType(physnum, 15, [], nodesBot[0:1])
    gmsh.model.addPhysicalGroup(0, [physnum], physnum, name="MasterBot")

    physnum = 1000002
    gmsh.model.addDiscreteEntity(0, physnum)
    gmsh.model.mesh.addElementsByType(physnum, 15, [], nodesTop[0:1])
    gmsh.model.addPhysicalGroup(0, [physnum], physnum, name="MasterTop")

    # gmsh.write(os.path.join(os.path.dirname(image), 'out.msh'))


def read_config_file(path: str):
    cfg = configparser.ConfigParser()
    cfg.read(path)

    try:
        cfg_dict = {
            "processing_dir": cfg.get("DIRECTORIES", "processing_dir"),
            "datasets": [path for _, path in cfg.items("DATASETS")],
            "overwrite": cfg.get("FLAGS", "overwrite"),
            "energy": cfg.get("PHASE", "energy"),
            "distance_entry": cfg.get("ENTRIES", "distance"),
            "pixel_size_m": cfg.get("PHASE", "pixel_size_m"),
        }
    except configparser.NoSectionError:
        cfg_dict = {
            "processing_dir": cfg.get("DIRECTORIES", "processing_dir"),
            "datasets": [path for _, path in cfg.items("DATASETS")],
            "overwrite": False,
            "energy": cfg.get("PHASE", "energy"),
            "distance_entry": cfg.get("ENTRIES", "distance"),
            "pixel_size_m": cfg.get("PHASE", "pixel_size_m"),
        }

    return cfg_dict


def get_dataset_name(path: str):
    return os.path.splitext(path)[0].split("/")[-1]


def build_tiff_path(path: str):
    return os.path.splitext(path)[0] + ".tiff"
