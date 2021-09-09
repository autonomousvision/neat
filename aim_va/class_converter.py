import numpy as np

# https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera

sub_classes = {}

sub_classes['no_stop'] = np.uint8([
    0,      # unlabeled    
    0,      # building     
    0,      # fence
    0,      # other
    1,      # pedestrian 
    0,      # pole
    3,      # road line
    5,      # road
    4,      # sidewalk
    0,      # vegetation
    2,     # vehicle
    0,     # wall
    0,     # traffic sign
    0,     # sky
    0,     # ground
    0,     # bridge
    0,     # rail track
    0,     # guard rail
    0,     # traffic light
    0,     # static
    0,     # dynamic
    0,     # water
    0,     # terrain
    6,     # red lights
    6,     # yellow light
    0,     # green light
    0,     # stop sign
    3,     # stop lane marking
    ])

sub_classes['full'] = np.uint8([
    0,      # unlabeled    
    1,      # building     
    2,      # fence
    3,      # other
    4,      # pedestrian 
    5,      # pole
    6,      # road line
    7,      # road
    8,      # sidewalk
    9,      # vegetation
    10,     # vehicle
    11,     # wall
    12,     # traffic sign
    13,     # sky
    14,     # ground
    15,     # bridge
    16,     # rail track
    17,     # guard rail
    18,     # traffic light
    19,     # static
    20,     # dynamic
    21,     # water
    22,     # terrain
    23,     # red lights
    24,     # yellow light
    25,     # green light
    26,     # stop sign
    27,     # stop lane marking
    ])

sub_classes['6_classes'] = np.uint8([
    0,      #0 unlabeled    
    0,      #1 building     
    0,      #2 fence
    1,      #3 other
    2,      #4 pedestrian 
    0,      #5 pole
    3,      #6 road line
    1,      #7 road
    0,      #8 sidewalk
    0,      #9 vegetation
    4,     #10 vehicle
    0,     #11 wall
    0,     #12 traffic sign
    0,     #13 sky
    0,     #14 ground
    0,     #15 bridge
    0,     #16 rail track
    0,     #17 guard rail
    0,     #18 traffic light
    0,     #19 static
    0,     #20 dynamic
    0,     #21 water
    0,     #22 terrain
    5,     #23 red lights
    5,     #24 yellow light
    0,     #25 green light
    0,     #26 stop sign
    3,     #27 stop lane marking
])

