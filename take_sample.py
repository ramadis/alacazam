import os
import scenedetect
import subprocess
import glob
import shutil

sample_path = 'sample'
testing_path = 'testing'

# delete training folder
if os.path.exists(testing_path) and os.path.isdir(testing_path):
  shutil.rmtree(testing_path)

# create training folder
os.makedirs(testing_path)

# generate frames for each scene from video db
for filename in sorted(glob.glob('sample/*.mp4')):
  file_path = os.path.join(filename)
  subprocess.call(['scenedetect','--input', file_path, '--output', testing_path, 'save-images','-n','1','detect-content'])
