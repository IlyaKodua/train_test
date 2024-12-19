from calculator import VelocityCalculator
import yaml
import os
import shutil

with open('config.yml') as f:
    config = yaml.safe_load(f)


if os.path.exists(config['out_dir']):
    shutil.rmtree(config['out_dir'])
os.mkdir(config['out_dir'])

output_path = os.path.join(config['out_dir'], config['output_video_name'])

calculator = VelocityCalculator(frame_step = config['frame_step'])


calculator.get_video_with_velocity(config['input_video_path'],
                                   output_path)