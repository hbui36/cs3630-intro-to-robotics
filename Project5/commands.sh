docker run -it --name project5 -v /home/duckie/projects/project5/src/project/packages:/code/catkin_ws/src/project/packages --rm --net=host alexma3312/cs3630-project:v6 /bin/bash

rosrun project5 test_image_quality.py -hostname duckie019

rosrun project5 move_forward_and_capture_images.py -hostname duckie019 -duration 4 -vel 0.4 -trim 0.025

scp -r duckie@duckie019.local:/home/duckie/projects/project5/src/project/packages/project5/src/saved_images/ .

