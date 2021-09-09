import os

class GlobalConfig:
    """ base architecture configurations """

	# Data
    root_dir = '/is/rg/avg/kchitta/carla9-10_data/2021/apv3'
    train_towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10']
    val_towns = ['Town01_long', 'Town02_long', 'Town03_long', 'Town04_long', 'Town05_long', 'Town06_long']
    train_data, val_data = [], []
    for town in train_towns:
        train_data.append(os.path.join(root_dir, town))
        train_data.append(os.path.join(root_dir, town+'_small'))
    for town in val_towns:
        val_data.append(os.path.join(root_dir, town))

    num_camera = 3 # camera configuration (1 or 3)
    pred_len = 4 # future waypoints predicted
    seq_len = 1 # input timesteps
    tot_len = seq_len + pred_len

    scale = 1 # image pre-processing, downscale factor
    crop = 256 # image pre-processing

    scale_topdown = 1 # bird view preprocessing
    crop_topdown = 512 # bird view preprocessing
    resolution = 1/5.5 # resolution of scene representation (from LBC)
    axis = 256 # width/height of scene representation (from LBC)
    offset = 128 # offset along y dimension to the origin (from LBC)

    num_class = 5
    classes = {
        0: [0, 0, 0],        # unlabeled
        1: [0, 0, 255],    # obstacle
        2: [128, 64, 128],   # road
        3: [255, 0, 0],     # red light
        4: [0, 255, 0],     # green light
        # 5: [157, 234,  50],    # road line
    }

    converter = [
    0,    # unlabeled
    0,    # building
    0,    # fence
    0,    # other
    1,    # ped
    0,    # pole
    2,    # road line
    2,    # road
    0,    # sidewalk
    0,    # vegetation
    1,    # vehicle
    0,    # wall
    0,    # traffic sign
    0,    # sky
    0,    # ground
    0,    # bridge
    0,    # rail track
    0,    # guard rail
    0,    # traffic light
    0,    # static
    0,    # dynamic
    0,    # water
    0,    # terrain
    3,    # red light
    3,    # yellow light
    4,    # green light
    0,    # stop sign
    ]

    points_per_class = 64 # number of scene points sampled per class for perception training
    t_height = 2.0 # distance above and below LiDAR (z-dimension) where points are considered obstacles 

    # Optimization
    lr = 1e-4
    loss_perc = 1.0
    loss_plan = 0.1
    iter_losses = [0.1, 1.0]

    # Conv Encoder
    vert_anchors = 8
    horz_anchors = 8
    anchors = vert_anchors * horz_anchors

	# Transformer
    n_layer = 2
    n_embd = 512
    block_exp = 4
    n_head = 4

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

	# Implicit Function
    attention_iters = 2
    onet_hidden_size = 128
    onet_blocks = 5

    # Controller
    plan_points = 1 # length of grid sampled for locating waypoints
    light_x_steps = 16 # horizontal grid samples for red lights (to right of vehicle)
    light_y_steps = 32 # vertical grid samples for red lights (to front of vehicle)
    plan_iters = 1 # recurrence count for locating waypoints
    plan_scale = 0.1 # (x,y) range from which to sample points for planning

    aim_dist = 4.0 # distance to search around for aim point
    angle_thresh = 0.3 # outlier control detection angle
    dist_thresh = 10 # target point y-distance for outlier filtering
    turn_KP = 0.75
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40 # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40 # buffer size

    red_light_mult = 0.0 # reduction in desired speed when red light detected
    max_throttle = 0.75 # upper limit on throttle signal
    brake_speed = 0.4 # desired speed below which brake is triggered
    brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.25 # maximum change in speed input to logitudinal controller

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)