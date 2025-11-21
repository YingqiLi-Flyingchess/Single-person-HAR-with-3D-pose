import numpy as np

def _3d_pose_to_pc(poses, scale=1e-3):
    """
    Turn poses of shape (N_people, N_joints, 3) in (x,y,z) meters
    into (x, z, -y) in the same shape, all scaled to meters.
    """
    # poses[...,0]=x, [...,1]=y, [...,2]=z
    x = poses[..., 0] * scale
    y = poses[..., 2] * scale
    z = -poses[..., 1] * scale
    return {'X': x, 'Y': y, 'Z': z}

#Kinect intrinsic parameters
color_calibration_params = {'cx':638.5570678710938,
                            'cy':368.1064147949219,
                            'fx':611.3757934570312,
                            'fy':611.5018920898438,
                            'k1':0.49009519815444946,
                            'k2':-2.669199228286743,
                            'k3':1.5499054193496704,
                            'k4':0.36899223923683167,
                            'k5':-2.490204334259033,
                            'k6':1.47518789768219,
                            'codx':0.0,
                            'cody':0.0,
                            'p1':0.0005917315138503909,
                            'p2':-0.0002122735750162974,
                            'metric_radius':1.7000000476837158}

translation = [-32.0264778137207,-1.8919668197631836,3.9770002365112305]
rotation = [0.9999954104423523,0.0030102587770670652,-0.00032541988184675574,-0.0029597715474665165,0.9945205450057983,0.10449939966201782,0.0006382070132531226,-0.10449796169996262,0.9945248961448669]

#mimics behavior of transformation_extrinsics_transform_point_3 for calibration.extrinsics[0][1] (0 is source (DEPTH) and 1 is target (COLOR))
#See: https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/src/transformation/extrinsic_transformation.c#L18
def transformation_extrinsics_transform_point_3(xyz):
    '''
    Args:
        xyz (numpy): shape(3,) xyz coordinate before extrinsic transformation

    Returns:
        (numpy): shape(3,) xyz coordinate after extrinsic transformation
    '''
    xyz_tr = np.zeros_like(xyz)
    a = xyz[0]
    b = xyz[1]
    c = xyz[2]
    xyz_tr[0] = rotation[0]*a + rotation[1]*b + rotation[2]*c + translation[0]
    xyz_tr[1] = rotation[3]*a + rotation[4]*b + rotation[5]*c + translation[1]
    xyz_tr[2] = rotation[6]*a + rotation[7]*b + rotation[8]*c + translation[2]

    return xyz_tr


#mimics the behavior of transformation_project and transformation_project_internal for DEPTH to COLOR 3d to 2d conversion
#See: https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/src/transformation/intrinsic_transformation.c#L14
#def transformation_project_internal(xyz):
def _3d_to_digital(xyz):
    '''
    Args:
        xyz (numpy): shape(3,) xyz coordinate from kinect_camera_data.csv, before 3D->2D projection

    Returns:
        (numpy: shape(2,) uv coordinate after transformations and 3D->2D projection
    '''
    cx = color_calibration_params['cx']
    cy = color_calibration_params['cy']
    fx = color_calibration_params['fx']
    fy = color_calibration_params['fy']
    k1 = color_calibration_params['k1']
    k2 = color_calibration_params['k2']
    k3 = color_calibration_params['k3']
    k4 = color_calibration_params['k4']
    k5 = color_calibration_params['k5']
    k6 = color_calibration_params['k6']
    codx = color_calibration_params['codx']
    cody = color_calibration_params['cody']
    p1 = color_calibration_params['p1']
    p2 = color_calibration_params['p2']
    max_radius_for_projection = color_calibration_params['metric_radius']
    
    xyz = transformation_extrinsics_transform_point_3(xyz)

    uv = np.zeros((2))
    uv[0] = xyz[0]/xyz[2]
    uv[1] = xyz[1]/xyz[2]

    xp = uv[0] - codx
    yp = uv[1] - cody
    xp2 = xp * xp
    yp2 = yp * yp
    xyp = xp * yp
    rs = xp2 + yp2
    # if rs > max_radius_for_projection * max_radius_for_projection:
    #     #invalid projection
    #     print(f"invalid projection, rs: {rs}")
    #     return None
    rss = rs * rs
    rsc = rss * rs
    a = 1.0 + k1 * rs + k2 * rss + k3 * rsc
    b = 1.0 + k4 * rs + k5 * rss + k6 * rsc
    bi = 1
    if b != 0:
        bi = 1.0 / b
    
    d = a * bi
    xp_d = xp * d
    yp_d = yp * d
    rs_2xp2 = rs + 2.0 * xp2
    rs_2yp2 = rs + 2.0 * yp2
    #since our setup uses the K4A_CALIBRATION_LENS_DISTORTION_MODEL_BROWN_CONRADY model:
    xp_d += rs_2xp2 * p2 + 2.0 * xyp * p1
    yp_d += rs_2yp2 * p1 + 2.0 * xyp * p2

    xp_d_cx = xp_d + codx
    yp_d_cy = yp_d + cody

    uv[0] = xp_d_cx * fx + cx
    uv[1] = yp_d_cy * fy + cy
    
    return uv
