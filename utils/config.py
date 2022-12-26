"""
Get config from file config.ini
"""
import configparser

config_path = "config/config.ini"
config = configparser.ConfigParser()
config.read(config_path)

print(config.sections())


# Get image size
def get_image_size():
    image_weight = config['image_size']['w']
    return image_weight


# Select device GPU or CPU
def get_divice():
    device = config['device_selection']['device']
    return device


# Set confidence threshold
def get_confidence_threshold():
    conf_thres = config['confidence_threshold']['conf_thres']
    return conf_thres


# Set iou_threshold
def get_iou_threshold():
    iou_thres = config['iou_threshold']['iou_thres']
    return iou_thres


# Select model
def get_model():
    model = config['weights']['model']
    return model
