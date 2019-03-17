import os
import scenedetect
import subprocess
import glob
import shutil

training_path = 'training'

# delete training folder
if os.path.exists(training_path) and os.path.isdir(training_path):
  shutil.rmtree(training_path)

# create training folder
os.makedirs(training_path)

# generate frames for each scene from video db
for filename in sorted(glob.glob('video_db/*.mp4')):
  file_path = os.path.join(filename)
  subprocess.call(['scenedetect','--input', file_path, '--output', training_path, 'save-images','detect-content'])
