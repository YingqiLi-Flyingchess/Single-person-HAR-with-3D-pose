from constants import KINECT_KPT_DIMS, KINECT_JOINT_NAMES, BODY8_JOINT_NAMES, BODY8_KPT_DIMS

def parse_poses3d(poses3d):
    if poses3d.shape[0] == 0:
        return None
    return poses3d.reshape(-1, len(KINECT_JOINT_NAMES), len(KINECT_KPT_DIMS))

def parse_poses2d(poses2d):
    if poses2d.shape[0] == 0:
        return None, None
    poses = poses2d[:,:-1].reshape(-1,len(BODY8_JOINT_NAMES), len(BODY8_KPT_DIMS))
    bbox_scores = poses2d[:,-1]
    return poses, bbox_scores