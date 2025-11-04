import carla
import random
import signal

random.seed(89)

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(2000)
world = client.get_world()
blueprint_library = world.get_blueprint_library()
# pick out blueprints
vehicle_tesla_bp = blueprint_library.find("vehicle.tesla.model3")
vehicle_random1_bp = blueprint_library.find("vehicle.audi.a2")
# vehicle_random7_bp = blueprint_library.find("vehicle.jeep.wrangler_rubicon")
vehicle_random9_bp = blueprint_library.find("vehicle.citroen.c3")
crossbike_bp = blueprint_library.find("vehicle.micro.microlino")
pedestrian_bp4 = blueprint_library.find("walker.pedestrian.0004")
pedestrian_bp5 = blueprint_library.find("walker.pedestrian.0005")
pedestrian_bp6 = blueprint_library.find("walker.pedestrian.0006")
pedestrian_bp7 = blueprint_library.find("walker.pedestrian.0007")
pedestrian_bp8 = blueprint_library.find("walker.pedestrian.0008")
bike_bp = blueprint_library.find("vehicle.nissan.patrol")
bin_bp = blueprint_library.find("static.prop.bin")
vehicle_random4_bp = blueprint_library.find("vehicle.bmw.grandtourer")

# chainbarrier = blueprint_library.find("static.prop.chainbarrier")

transform1 = carla.Transform(carla.Location(x=-200, y=12, z=0.2),
                             carla.Rotation(roll=-0.001, pitch=0, yaw=-90))

transform2 = carla.Transform(carla.Location(x=131.8, y=226.5, z=0.4),
                             carla.Rotation(roll=-0.001, pitch=0.350, yaw=90))

transform3 = carla.Transform(carla.Location(x=137.3, y=223.5, z=0.5),
                             carla.Rotation(roll=-0.001, pitch=0.350, yaw=90))

transform4 = carla.Transform(carla.Location(x=131.7, y=214.4, z=0.4),
                             carla.Rotation(roll=-0.001, pitch=0, yaw=-90))  # 138

transform5 = carla.Transform(carla.Location(x=179.5, y=186.5, z=0.4),
                             carla.Rotation(roll=-0.001, pitch=0.350, yaw=0.606))

transform6 = carla.Transform(carla.Location(x=188.5, y=180.8, z=0.5),
                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=90))

transform7 = carla.Transform(carla.Location(x=194, y=172.2, z=0.4),
                             carla.Rotation(roll=-0.001, pitch=0.350, yaw=-90))

transform8 = carla.Transform(carla.Location(x=193, y=172, z=0.4),
                             carla.Rotation(roll=-0.001, pitch=0, yaw=90))

transform9 = carla.Transform(carla.Location(x=190, y=227, z=0.5),
                             carla.Rotation(roll=-0.001, pitch=0.350, yaw=90))

transform10 = carla.Transform(carla.Location(x=193.5, y=212, z=0.5),
                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=-90))

transform11 = carla.Transform(carla.Location(x=116.8, y=236, z=0.5),
                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=180))

transform12 = carla.Transform(carla.Location(x=117.5, y=236.8, z=0.5),
                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=180))

transform13 = carla.Transform(carla.Location(x=134.1, y=205.6, z=0.5),
                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=90))

transform14 = carla.Transform(carla.Location(x=159.0, y=187, z=0.5),
                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=180))

transform15 = carla.Transform(carla.Location(x=159.0, y=191.8, z=0.5),
                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=180))

transform16 = carla.Transform(carla.Location(x=158.6, y=192.8, z=0.5),
                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=180))

# transform17 = carla.Transform(carla.Location(x= 180.0, y=187.5, z=0.5),
#                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=90))

# transform18 = carla.Transform(carla.Location(x= 179.0, y=190, z=0.5),
#                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=90))

# transform19 = carla.Transform(carla.Location(x= 180.0, y=191.8, z=0.5),
#                             carla.Rotation(roll=-0.001, pitch=0.350, yaw=90))

transform20 = carla.Transform(carla.Location(x=188.5, y=180.8, z=0.5),
                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=90))

transform21 = carla.Transform(carla.Location(x=185.8, y=189.5, z=0.5),
                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=180))

transform35 = carla.Transform(carla.Location(x=132, y=229.6, z=0.5),
                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=90))

transform36 = carla.Transform(carla.Location(x=135.2, y=228, z=0.5),
                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=-90))

transform37 = carla.Transform(carla.Location(x=123.8, y=234.3, z=0.5),
                               carla.Rotation(roll=-0.001, pitch=0.350, yaw=90))
def spawn_npc():
    try:
        actor1 = world.spawn_actor(vehicle_random4_bp, transform1)
        actor1.set_simulate_physics(False)
        # actor2 = world.spawn_actor(crossbike_bp, transform2)
        # actor2.set_simulate_physics(False)
        # actor3 = world.spawn_actor(pedestrian_bp4, transform3)
        # actor3.set_simulate_physics(False)
        # actor4 = world.spawn_actor(vehicle_random1_bp, transform4)
        # actor4.set_simulate_physics(False)
        # actor5 = world.spawn_actor(bin_bp, transform5)
        # # actor5.set_simulate_physics(False)
        # actor6 = world.spawn_actor(vehicle_tesla_bp, transform6)
        # actor6.set_simulate_physics(False)
        # actor7 = world.spawn_actor(pedestrian_bp6, transform7)
        # actor7.set_simulate_physics(False)
        # actor8 = world.spawn_actor(pedestrian_bp7, transform8)
        # actor8.set_simulate_physics(False)
        # actor9 = world.spawn_actor(vehicle_tesla_bp, transform9)
        # actor10 = world.spawn_actor(vehicle_random9_bp, transform10)

        # actor11 = world.spawn_actor(pedestrian_bp4, transform11)
        # actor12 = world.spawn_actor(pedestrian_bp5, transform12)
        # actor13 = world.spawn_actor(crossbike_bp, transform13)
        # actor14 = world.spawn_actor(pedestrian_bp7, transform14)
        # actor15 = world.spawn_actor(pedestrian_bp5, transform15)
        # actor16 = world.spawn_actor(pedestrian_bp4, transform16)
        # actor17 = world.spawn_actor(pedestrian_bp7, transform17)
        # actor18 = world.spawn_actor(pedestrian_bp5, transform18)
        # actor19 = world.spawn_actor(pedestrian_bp4, transform19)
        # actor20 = world.spawn_actor(vehicle_tesla_bp, transform20)
        # actor21 = world.spawn_actor(vehicle_tesla_bp, transform21)

        while True:
            pass

    except KeyboardInterrupt as e:
        actor1.destroy()
        # actor2.destroy()
        # actor3.destroy()
        # actor4.destroy()
        #
        # actor5.destroy()
        # actor6.destroy()
        # actor7.destroy()
        # actor8.destroy()
        # actor9.destroy()
        # actor10.destroy()

        # actor11.destroy()
        # actor12.destroy()
        # actor13.destroy()
        # actor14.destroy()
        # actor15.destroy()
        # actor16.destroy()
        # actor17.destroy()

        # actor18.destroy()
        # actor19.destroy()
        # actor20.destroy()
        # actor21.destroy()


if __name__ == "__main__":
    spawn_npc()