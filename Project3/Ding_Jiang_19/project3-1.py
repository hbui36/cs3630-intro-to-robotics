# -*- coding: utf-8 -*-
"""Project 3 Part 1 Instructions.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BnEwqFgcJKLm2OHzatCjWX0Mzd0pcKix

# Project 3: Differential Drive (Part 1)

Due Date: Monday, February 10, 2020 @ 11:59 P.M.

Student #1 Name: Fei Ding

Student #2 Name: Zhen Jiang

In this project, you will be experimenting with differential drive to make the DuckieBot move in a certain way. Then you will see if the robot actually moves in a desired way by measuring the robot's trajectory.

## Setup

You will finally use the DuckieBot for this project!

To get the DuckieBot running, you first need to flash the SD card that is used for the Raspberry Pi on the robot.

You will need a Linux-based system to do that.  

Follow the instructions on the link below to flash the SD card. Instructions for installing a virtual machine is included for people with a Mac or Windows PC.

After you successfully get the robot running, you will have to connect to the robot wirelessly. This is done through a home router or phone hotspot.
Detailed instructions are in the link below. 

[SD flash and connect link](https://docs.google.com/presentation/d/1jAONDBIMahUPJKV61F9lSqVBLig5t9Q3k-OZBJAri3U/edit?usp=sharing)

Make sure to follow the instructions carefully. If you encounter any problems, post questions on Piazza or come to office hours to get help.

## Coding on the DuckieBot

Now that you are connected to the DuckieBot using the Chrome shell, you are ready to code!

The Raspberry Pi on the DuckieBot sends and receives commands through [Robot Operating System(ROS)](http://wiki.ros.org/ROS/Tutorials) and [ROS cheat sheet](https://mirror.umd.edu/roswiki/attachments/de/ROScheatsheet.pdf).  
ROS in an open source framework used in many robots. The ROS packages for the Duckiebot is wrapped in a container in a Docker environment.  
To find out more about Docker and how it works, visit the [Docker website](https://docs.docker.com/v17.09/engine/userguide/storagedriver/imagesandcontainers/).

When you [ssh](https://en.wikipedia.org/wiki/Secure_Shell) into the DuckeiBot through the chrome shell, you should be in the /home/duckie folder. 

Use the command   
`pwd`  
in the shell to verify your current path. 

In the /home/duckie folder, you have to clone the repository that contains the source code for project 3.  
```
git clone https://github.gatech.edu/CS3630-TAs/CS3630-Assignments.git project3
```  
Then, pull a docker image that contains the environment to execute the project 3 ROS package.  
```
docker pull alexma3312/cs3630-project:v5
```  
To check if the image was successfully downloaded, do
```
docker images
```
To see all docker images. You should see an image named `alexma3312/cs3630-project:v5`. v5 is the tag of the image.  

Next, run the docker image with
```
docker run -it --name project3 \
-v /home/duckie/project3/project3/src/project/packages:/code/catkin_ws/src/project/packages \
--rm --net=host alexma3312/cs3630-project:v5 /bin/bash
```  
You may see a message saying `VEHICLE_NAME is not set`. Ignore this as this is not an error.  
After this step you should be inside the docker container. The user of the shell will change from `duckie@duckieXXX` to `root@duckieXXX`.  
To check if you are in the container, do
```
lsb_release -a
```
You will see `Distributor ID:	Ubuntu` if you are in the container.

For this project, you will write your code on __move_in_circle.py__ 
The file path is `/code/catkin_ws/src/project/packages/project3/src/move_in_circle.py` .  
To edit the python file, you have to use a built-in text editor called nano.  
Enter command
```
nano packages/project3/src/move_in_circle.py
```  
to open and edit the file.  
If you are not familiar with nano, here's a link to a nice [tutorial](https://www.howtogeek.com/howto/42980/the-beginners-guide-to-nano-the-linux-command-line-text-editor/).   
After editing, you can simply do ctrl+x to save and exit.  

To run the project inside the container, do
```
rosrun project3 move_in_circle.py
```
The code will not run initially! You have to fill in the required sections.   
To exit the container, do
```
exit
```
To shut down the Raspberry Pi, do
```
sudo poweroff
```
then plug off the power cable and turn the battery off.

## 1. Making the robot move in a circle around a point [30+20 points]

In this part you will make the DuckieBot move in a circle around a point with a set radius.  
As you learned in class, by varying the velocities of the two wheels, you can vary the trajectories that the robot takes.  
To make the robot go in a circle of radius R, you will have to know exactly how much velocity to give to each wheel.  
You will write code on move_in_circle.py 

__Differential Drive Kinematics.__ You will implement calculate_radius for this part. Given two velocities, calculate the radius of the circle R the robot will move in.    
Check [this link](http://www.cs.columbia.edu/~allen/F17/NOTES/icckinematics.pdf) for the math.  
Implement the code below. Test on few sets of velocities to see if your implementation returns the right R.  
After that, copy your code to calculate_radius in move_in_circle.py using nano. Instructions for this is written above.
"""

def calculate_radius(velocity_left, velocity_right):
    """ Calculate radius with given left and right velocity values.
    Parameters:
        velocity left: range [-1,1], (implement assert for this part)
        velocity right: range [-1,1]    
    Return:
        radius: radius of the circle
    """
    assert velocity_left>=-1 and velocity_left<=1, "The range of velocity is [-1,1]"
    assert velocity_right>=-1 and velocity_right<=1, "The range of velocity is [-1,1]"
    # distance between centers of two wheels
    WHEEL_DIST = 0.102
    
    ############################
    ### TODO: YOUR CODE HERE ###
    if velocity_left == velocity_right:
      radius = 0
    else:
      diff = velocity_left - velocity_right
      if diff < 0:
        diff = diff * -1
      radius = WHEEL_DIST * (velocity_left + velocity_right) / (2 * diff)
    # raise NotImplementedError('`calculate_radius` function needs '
    #   + 'to be implemented')
    ### END OF STUDENT CODE ####
    ############################
    
    return radius

"""Run this to check if your `calculate_radius` passes the given test cases. (you wil have to run calculate_radius before you run the unit test!)"""

import unittest

class TestCalculateRadius(unittest.TestCase):
    def test_calculate_radius(self):
        """
        Checks if calculate radius is working properly.
        """
        assert calculate_radius(0.5, 1) == 0.153,\
            "Calculate radius does not return correctly"
        assert calculate_radius(1, 1) == 0,\
            "0 radius when left and right velocity is equal."
        try:
            calculate_radius(2, 1)
            print("The range of velocity is [-1,1]")
        except:
            pass
        try:
            calculate_radius(1, -2)
            print("The range of velocity is [-1,1]")
        except:
            pass

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestCalculateRadius("test_calculate_radius"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

"""__Making the Robot Move. [30 points]__

In `__main__`, you will see a part to write the left and right wheel velocities. After you fill in the velocities, run the ros node using command
```
rosrun project3 move_in_circle.py
```
You will see the radius of the circle calculated using calculate_radius you implemented. The robot wil also start running.  
ctrl+c out of the running rosnode to make the robot stop. 

---  
Is the robot running in a consistent circle?  
Is the radius similar to the radius you calculated?  
What can you do to make the robot move in a consistent circle of the desired radius?  
Experiment and answer the questions in the reflection.

__Video of moving robot [20 points]__
When you are done, take a video of the DuckieBot moving. The video should show the robot moving in a circle. You don't _have_ to make the robot move in a consistent circle for now. Record at least __three__ consecutive rotations.  
Please limit the video length to under __20 seconds!__

Save the video as {LASTNAME1_LASTNAME2_GROUPNUMBER}.mp4  
(other file formats are okay)

## 3. Reflection [50 points]

Answer the questions in the proj3_part1_report_template.pptx . 
You can find the pptx file in the files tab on Canvas.  
Save the file as a PDF and rename it to {FirstName_LastName}.pdf

## Rubric

- 30 pts: calculate_radius()
- 20 pts: video of DuckieBot moving in circles
- 50 pts: reflection writeup

## Submission Details
### Deliverables

A zip file named {LASTNAME1_LASTNAME2_GROUPNUMBER}.zip with the following files:
- project3.py - convert your colab ipynb file to py format. (File -> Download as `.py`)
- {LASTNAME1_LASTNAME2_GROUPNUMBER}.mp4 - Video of the Duckiebot running in a circle.
- {LASTNAME1_LASTNAME2_GROUPNUMBER}.pdf: The reflection slides converted to PDF form. 

Submit the zip file to Canvas. Only one person per group should upload the submission.
"""