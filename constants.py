OOB_TID = 254
NOISE_TARGET_IDS = [253, OOB_TID, 255]
'''
The GTRACK algorithm was used to filter out noisy points.
Each point is associated with a target id, defined as
    0-252  not noise, target detected as a person with identification number 0-252 
    253    noise, signal to noise ratio is too week 
    254    noise, located outside of the physical room boundary
    255    noise, target not associated as a person
'''
ROOM_BOUNDS = [
    [-3.0, 2.0], #x1,x2
    [0.0, 5.0],  #y1,y2
    [-1.5, 3.0]  #z1,z2
]

VIDEO_FRAME_DIMS = [1280, 720]

KINECT_JOINT_NAMES = [
        'pelvis',
        'spine - navel',
        'spine - chest',
        'neck',
        'left clavicle',
        'left shoulder',
        'left elbow',
        'left wrist',
        'left hand',
        'left handtip',
        'left thumb',
        'right clavicle',
        'right shoulder',
        'right elbow',
        'right wrist',
        'right hand',
        'right handtip',
        'right thumb',
        'left hip',
        'left knee',
        'left ankle',
        'left foot',
        'right hip',
        'right knee',
        'right ankle',
        'right foot',
        'head',
        'nose',
        'left eye',
        'left ear',
        'right eye',
        'right ear'
]

KINECT_KPT_DIMS = ["px", "py", "pz", "conf"]

BODY8_JOINT_NAMES = ["Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle", "Head_Apex", "Neck", "Hip_Center", "L_BigToe", "R_BigToe", "L_SmallToe", "R_SmallToe", "L_Heel", "R_Heel"]
BODY8_LIMB_CONNECTIONS = [(0,1),(0,2),(2,1),(2,4),(1,3),(4,6),(3,5),(5,7),(6,8),(7,9),(8,10),(19,11),(19,12),(11,13),(12,14),(13,15),(14,16),(15,24),(16,25),(15,20),(15,22),(16,21),(16,23),(19,18),(18,17),(18,5),(18,6)]
BODY8_KPT_DIMS = ["px", "py", "conf"]

KINECT_LIMB_CONNECTIONS = [
    ('pelvis', 'spine - navel'),
    ('spine - navel', 'spine - chest'),
    ('spine - chest', 'neck'),
    ('neck', 'head'),

    # Left arm
    ('spine - chest', 'left clavicle'),
    ('left clavicle', 'left shoulder'),
    ('left shoulder', 'left elbow'),
    ('left elbow', 'left wrist'),
    ('left wrist', 'left hand'),
    ('left hand', 'left handtip'),
    ('left hand', 'left thumb'),

    # Right arm
    ('spine - chest', 'right clavicle'),
    ('right clavicle', 'right shoulder'),
    ('right shoulder', 'right elbow'),
    ('right elbow', 'right wrist'),
    ('right wrist', 'right hand'),
    ('right hand', 'right handtip'),
    ('right hand', 'right thumb'),

    # Left leg
    ('pelvis', 'left hip'),
    ('left hip', 'left knee'),
    ('left knee', 'left ankle'),
    ('left ankle', 'left foot'),

    # Right leg
    ('pelvis', 'right hip'),
    ('right hip', 'right knee'),
    ('right knee', 'right ankle'),
    ('right ankle', 'right foot'),
]
