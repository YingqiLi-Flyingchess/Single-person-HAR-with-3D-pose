import matplotlib.pyplot as plt
from constants import KINECT_LIMB_CONNECTIONS, KINECT_JOINT_NAMES, ROOM_BOUNDS, VIDEO_FRAME_DIMS, BODY8_LIMB_CONNECTIONS
from transformations import _3d_to_digital, _3d_pose_to_pc
import numpy as np

def plot_pc_3d(cloud, poses=None, kpt_scale=1e-3, figsize=(8, 6), boundless=False):
    px, py, pz, pd, ps, tid = cloud.T
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(px, py, pz, c='r', s=1)

    if poses is not None:
        P = _3d_pose_to_pc(poses, scale=kpt_scale)
        # scatter all joints
        ax.scatter(P['X'].ravel(), P['Y'].ravel(), P['Z'].ravel(), c='b', s=5)

        # draw bones
        name_to_idx = {n: i for i, n in enumerate(KINECT_JOINT_NAMES)}
        for person in range(poses.shape[0]):
            for l, r in KINECT_LIMB_CONNECTIONS:
                i, j = name_to_idx[l], name_to_idx[r]
                ax.plot([P['X'][person, i], P['X'][person, j]],
                        [P['Y'][person, i], P['Y'][person, j]],
                        [P['Z'][person, i], P['Z'][person, j]],
                        c='b', linewidth=1)

    if not boundless:
        ax.set_xlim(*ROOM_BOUNDS[0])
        ax.set_ylim(*ROOM_BOUNDS[1])
        ax.set_zlim(*ROOM_BOUNDS[2])

    fig.tight_layout()
    plt.show()

def plot_pc_2d_pose(cloud, poses2d=None, poses3d=None, pc_scale=1e3, figsize=(15, 8), boundless=False):
    fig, ax = plt.subplots(figsize=figsize)
    pts3d = cloud[:, :3] * pc_scale
    # reorder axes: (x, y, z) → (x, –z, y)
    cam_coords = np.column_stack([pts3d[:,0], pts3d[:,2], pts3d[:,1]])
    proj_pc = np.stack([_3d_to_digital(p) for p in cam_coords])
    ax.scatter(proj_pc[:,0], proj_pc[:,1], c='r', s=1)
    
    if poses2d is not None:
        print(poses2d.shape)
        all_joints2d = np.vstack(poses2d)
        ax.scatter(all_joints2d[:,0], VIDEO_FRAME_DIMS[1]-all_joints2d[:,1], c='b', s=2)
        M, J, _ = poses2d.shape
        for person_idx in range(M):
            for l, r in BODY8_LIMB_CONNECTIONS:
                x0, y0, _ = poses2d[person_idx, l]
                x1, y1, _ = poses2d[person_idx, r]
                ax.plot([x0, x1], [VIDEO_FRAME_DIMS[1]-y0, VIDEO_FRAME_DIMS[1]-y1], c='b', linewidth=2)

    if poses3d is not None:
        all_joints3d = np.vstack(poses3d)
        print(all_joints3d.shape)
        cam_joints = np.column_stack([all_joints3d[:,0], all_joints3d[:,1], all_joints3d[:,2]])
        print(cam_joints.shape)
        proj_joints = np.stack([_3d_to_digital(p) for p in cam_joints])
        ax.scatter(proj_joints[:,0], VIDEO_FRAME_DIMS[1]-proj_joints[:,1], c='b', s=2)
        name_to_idx = {n:i for i,n in enumerate(KINECT_JOINT_NAMES)}
        M, J, _ = poses3d.shape
        proj_joints = proj_joints.reshape(M, J, 2)
        for person_idx in range(M):
            for l, r in KINECT_LIMB_CONNECTIONS:
                i, j = name_to_idx[l], name_to_idx[r]
                x0, y0 = proj_joints[person_idx, i]
                x1, y1 = proj_joints[person_idx, j]
                ax.plot([x0, x1], [VIDEO_FRAME_DIMS[1]-y0, VIDEO_FRAME_DIMS[1]-y1], c='b', linewidth=2)
    
    if not boundless:
        ax.set_xlim([0,VIDEO_FRAME_DIMS[0]])
        ax.set_ylim([0,VIDEO_FRAME_DIMS[1]])


def plot_pc_2d_planes(cloud, poses=None, kpt_scale=1e-3, figsize=(15, 5), boundless=False):
    # prepare cloud coords dict
    labels = ('X', 'Y', 'Z')
    coords = {lab: cloud[:, i] for i, lab in enumerate(labels)}

    # prepare poses if given
    P = _3d_pose_to_pc(poses, scale=kpt_scale) if poses is not None else None
    name_to_idx = {n: i for i, n in enumerate(KINECT_JOINT_NAMES)}

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plane_pairs = [('X','Y'), ('X','Z'), ('Y','Z')]
    idx_map = {'X': 0, 'Y': 1, 'Z': 2}

    for ax, (a, b) in zip(axes, plane_pairs):
        ax.scatter(coords[a], coords[b], c='r', s=1)
        if P is not None:
            ax.scatter(P[a].ravel(), P[b].ravel(), c='b', s=5)
            for person in range(poses.shape[0]):
                for l, r in KINECT_LIMB_CONNECTIONS:
                    i, j = name_to_idx[l], name_to_idx[r]
                    ax.plot([P[a][person,i], P[a][person,j]],
                            [P[b][person,i], P[b][person,j]],
                            c='b', linewidth=1)

        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_aspect('equal', 'box')

        if not boundless:
            ax.set_xlim(*ROOM_BOUNDS[idx_map[a]])
            ax.set_ylim(*ROOM_BOUNDS[idx_map[b]])

    fig.tight_layout()
    plt.show()