'''
Teach An Agent To Drive A Car In A Virtual Environment

@author: Dewang Shah, Anant Vignesh Mahadhevan and Rakesh Ramesh
'''

#Importing necessary packages
import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model

#Importing CARLA environment
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

#Setting CARLA environmental parameters 
showPreview = False
imageWidth = 640
imageHeight = 480
secondsPerEpisode = 10
replayMemorySize = 5_000
minReplayMemorySize = 1_000
modelName = "Xception"

#Class to create CARLA environment
class CarlaEnvironment:
    showCam = showPreview
    steerAmt = 1.0
    imageWidth = imageWidth
    imageHeight = imageHeight
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000) #Run CARLA as the server in port 2000
        self.client.set_timeout(10.0) #Set server environment timeout as 10 seconds
        self.world = self.client.get_world() #Initialize CARLA world
        self.blueprint_library = self.world.get_blueprint_library() #Initialize CARLA world blueprint
        self.model_3 = self.blueprint_library.filter("model3")[0] #Initialize Tesla Model 3 car blueprint

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        #Get a random spawn point from the map and spawn the vehicle in the spawn point
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        #Add the vehicle to the list of actors in the environment
        self.actor_list.append(self.vehicle)

        #Set the RGB Camera sensor attributes
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.imageWidth}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.imageHeight}")
        self.rgb_cam.set_attribute("fov", f"110")

        #Add the RGB Camera Sensor to the car
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        #Add the camera sensor to the list of actors in the environment
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.processImage(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        #Add the Collision Sensor to the car
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        #Add the collision sensor to the list of actors in the environment
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collisionData(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def processImage(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.imageHeight, self.imageWidth, 4))
        i3 = i2[:, :, :3]
        if self.showCam:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def collisionData(self, event):
        self.collision_hist.append(event)

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.steerAmt))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.steerAmt))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + secondsPerEpisode < time.time():
            done = True

        return self.front_camera, reward, done, None


class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=replayMemorySize)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{modelName}-{int(time.time())}")
        self.target_update_counter = 0
        self.graph = tf.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        base_model = Xception(weights=None, include_top=False, input_shape=(imageHeight, imageWidth,3))

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        predictions = Dense(3, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        return model