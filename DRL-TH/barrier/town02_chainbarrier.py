import carla
import random

random.seed(89)


# transform1 = carla.Transform(carla.Location(x= 98, y=237.2, z=0.5),
#                              carla.Rotation(roll=-0.001, pitch=0, yaw=0))

# transform2 = carla.Transform(carla.Location(x= 113, y=241.3, z=0.5),
#                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=0))

# transform3 = carla.Transform(carla.Location(x= 128, y=236.8, z=0.5),
#                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=0))


# transform4 = carla.Transform(carla.Location(x= 143, y=241, z=0.5),
#                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=0))


# transform5 = carla.Transform(carla.Location(x= 158, y=237, z=0.5),
#                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=0.606))

# transform6 = carla.Transform(carla.Location(x= 173, y=241.3, z=0.5),
#                              carla.Rotation(roll=-0.001, pitch=0, yaw=180))


# transform7 = carla.Transform(carla.Location(x= 193.3, y=253, z=0.5),
#                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=90))


# transform8 = carla.Transform(carla.Location(x= 190, y=268, z=0.5),
#                              carla.Rotation(roll=-0.001, pitch=0, yaw=-90))

# #
# transform9 = carla.Transform(carla.Location(x= 193.5, y=227, z=0.5),
#                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=90))

# transform10 = carla.Transform(carla.Location(x= 190, y=212, z=0.5),
#                              carla.Rotation(roll=-0.001, pitch=0.350, yaw=-90))



transform11 = carla.Transform(carla.Location(x= 138.8, y = 230.5, z = 0.5), 
                              carla.Rotation(roll = 0, pitch = 0,yaw = 260))

transform12 = carla.Transform(carla.Location(x= 139.8, y = 232.5, z = 0.5), 
                              carla.Rotation(yaw = 235))

transform13 = carla.Transform(carla.Location(x= 141.5, y = 233.9, z = 0.5), 
                              carla.Rotation(roll = 0, pitch = 0,yaw = 215))

transform14 = carla.Transform(carla.Location(x= 143.6, y = 234.7, z = 0.5), 
                              carla.Rotation(roll = 0, pitch = 0,yaw = 205))

transform15 = carla.Transform(carla.Location(x= 141, y = 243.2, z = 0.5), 
                              carla.Rotation(yaw = 0))
transform16 = carla.Transform(carla.Location(x= 138, y = 243.2, z = 0.5),
                              carla.Rotation(roll = 0, pitch = 0,yaw = 0))

transform17 = carla.Transform(carla.Location(x= 135, y = 243.2, z = 0.5), 
                              carla.Rotation(roll = 0, pitch = 0,yaw = 0))

transform18 = carla.Transform(carla.Location(x= 132, y = 243.2, z = 0.5), 
                              carla.Rotation(yaw = 0))
transform19 = carla.Transform(carla.Location(x= 129, y = 243.2, z = 0.5), 
                              carla.Rotation(roll = 0, pitch = 0,yaw = 0))

transform20 = carla.Transform(carla.Location(x= 126, y = 243.2, z = 0.5), 
                              carla.Rotation(roll = 0, pitch = 0,yaw = 0))

transform21 = carla.Transform(carla.Location(x= 125, y = 243.2, z = 0.5),   # x=123
                              carla.Rotation(roll = 0, pitch = 0,yaw = 0))

transform22 = carla.Transform(carla.Location(x= 144, y = 243.2, z = 0.5), 
                              carla.Rotation(yaw = 0))

# the second cernor
transform23 = carla.Transform(carla.Location(x= 138.4, y = 199.5, z = 0.5),
                              carla.Rotation(roll = 0, pitch = 0,yaw = 100))

transform24 = carla.Transform(carla.Location(x= 139.1, y = 197.4, z = 0.5),
                              carla.Rotation(roll = 0, pitch = 0,yaw = 110))
transform25 = carla.Transform(carla.Location(x= 139.8, y = 196.1, z = 0.5),
                              carla.Rotation(roll = 0, pitch = 0,yaw = 120))

transform26 = carla.Transform(carla.Location(x= 141.8, y = 194.6, z = 0.5),
                              carla.Rotation(roll = 0, pitch = 0,yaw = 150))

transform27 = carla.Transform(carla.Location(x= 144, y = 194, z = 0.5),
                              carla.Rotation(roll = 0, pitch = 0,yaw = 165))


transform28 = carla.Transform(carla.Location(x= 125, y = 234.6, z = 0.5),
                              carla.Rotation(yaw = -20))

transform29 = carla.Transform(carla.Location(x= 127, y = 233.9, z = 0.5),
                              carla.Rotation(roll = 0,pitch = 0, yaw = -45))

transform30 = carla.Transform(carla.Location(x= 128.5, y = 232.7, z = 0.5),
                               carla.Rotation(yaw = -60))

# transform31 = carla.Transform(carla.Location(x= 183.5, y = 243.6, z = 0.5),
#                               carla.Rotation(yaw = 25))

transform32 = carla.Transform(carla.Location(x= 132.2, y = 185.5, z = 0.5),
                              carla.Rotation(roll = 0,pitch = 0, yaw = 0))

transform33 = carla.Transform(carla.Location(x= 134, y = 185.5, z = 0.5),
                               carla.Rotation(roll = 0,pitch = 0, yaw = 0))

transform34 = carla.Transform(carla.Location(x= 136, y = 185.5, z = 0.5),
                               carla.Rotation(roll = 0,pitch = 0, yaw = 0))

transform35 = carla.Transform(carla.Location(x= 138, y = 185.5, z = 0.5),
                              carla.Rotation(roll = 0,pitch = 0, yaw = 0))

transform36 = carla.Transform(carla.Location(x= 140, y = 185.5, z = 0.5),
                               carla.Rotation(roll = 0,pitch = 0, yaw = 0))

transform37 = carla.Transform(carla.Location(x= 142, y = 185.5, z = 0.5),
                               carla.Rotation(roll = 0,pitch = 0, yaw = 0))

transform38 = carla.Transform(carla.Location(x= 144, y = 185.5, z = 0.5),
                               carla.Rotation(roll = 0,pitch = 0, yaw = 0))
def spawn_npc():
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(2000)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    # pick out blueprints
    vehicle_tesla_bp = blueprint_library.find("vehicle.tesla.model3")
    vehicle_random1_bp = blueprint_library.find("vehicle.audi.a2")
    # vehicle_random7_bp = blueprint_library.find("vehicle.jeep.wrangler_rubicon")
    vehicle_random9_bp = blueprint_library.find("vehicle.citroen.c3")
    # crossbike_bp = blueprint_library.find("vehicle.bmw.isetta")
    chainbarrier = blueprint_library.find("static.prop.chainbarrier")

    # blueprints1 = world.get_blueprint_library().find('static.prop.clothcontainer')
    # blueprints2 = world.get_blueprint_library().find('static.prop.container')
    # blueprints3 = world.get_blueprint_library().find('static.prop.kiosk_01')
    # blueprints4 = world.get_blueprint_library().find('static.prop.vendingmachine')
    # blueprints5 = world.get_blueprint_library().find('static.prop.trashcan04')
    # blueprints6 = world.get_blueprint_library().find('static.prop.trashcan05')
    # blueprints7 = world.get_blueprint_library().find('static.prop.box02')

    try:
        # actor1 = world.spawn_actor(vehicle_tesla_bp, transform1)
        # actor2 = world.spawn_actor(vehicle_random1_bp, transform2)
        # actor3 = world.spawn_actor(vehicle_tesla_bp, transform3)
        # actor4 = world.spawn_actor(vehicle_tesla_bp, transform4)
        # actor5 = world.spawn_actor(crossbike_bp, transform5)
        # actor6 = world.spawn_actor(vehicle_random9_bp, transform6)
        # actor7 = world.spawn_actor(vehicle_tesla_bp, transform7)
        # actor8 = world.spawn_actor(vehicle_random9_bp, transform8)
        # actor9 = world.spawn_actor(vehicle_tesla_bp, transform9)
        # actor10 = world.spawn_actor(vehicle_random9_bp, transform10)

        actor11 = world.spawn_actor(chainbarrier, transform11)
        actor12 = world.spawn_actor(chainbarrier, transform12)
        actor13 = world.spawn_actor(chainbarrier, transform13)
        actor14 = world.spawn_actor(chainbarrier, transform14)
        actor15 = world.spawn_actor(chainbarrier, transform15)
        actor16 = world.spawn_actor(chainbarrier, transform16)
        actor17 = world.spawn_actor(chainbarrier, transform17)
        actor18 = world.spawn_actor(chainbarrier, transform18)
        actor19 = world.spawn_actor(chainbarrier, transform19)
        actor20 = world.spawn_actor(chainbarrier, transform20)
        actor21 = world.spawn_actor(chainbarrier, transform21)
        actor22 = world.spawn_actor(chainbarrier, transform22)
        actor23 = world.spawn_actor(chainbarrier, transform23)
        actor24 = world.spawn_actor(chainbarrier, transform24)
        actor25 = world.spawn_actor(chainbarrier, transform25)
        actor26 = world.spawn_actor(chainbarrier, transform26)
        actor27 = world.spawn_actor(chainbarrier, transform27)

        actor28 = world.spawn_actor(chainbarrier, transform28)
        actor29 = world.spawn_actor(chainbarrier, transform29)
        actor30 = world.spawn_actor(chainbarrier, transform30)
        # actor31 = world.spawn_actor(chainbarrier, transform31)
        actor32 = world.spawn_actor(chainbarrier, transform32)
        actor33 = world.spawn_actor(chainbarrier, transform33)
        actor34 = world.spawn_actor(chainbarrier, transform34)
        actor35 = world.spawn_actor(chainbarrier, transform35)
        actor36 = world.spawn_actor(chainbarrier, transform36)
        actor37 = world.spawn_actor(chainbarrier, transform37)
        actor38 = world.spawn_actor(chainbarrier, transform38)
        # actor39 = world.spawn_actor(chainbarrier, transform33)
        # actor40 = world.spawn_actor(chainbarrier, transform34)


        while True:
            pass

    except KeyboardInterrupt as e:
        # actor1.destroy()
        # actor2.destroy()
        # actor3.destroy()
        # actor4.destroy()
        # actor5.destroy()
        # actor6.destroy()
        # actor7.destroy()
        # actor8.destroy()
        # actor9.destroy()
        # actor10.destroy()
        actor11.destroy()
        actor12.destroy()
        actor13.destroy()
        actor14.destroy()
        actor15.destroy()
        actor16.destroy()
        actor17.destroy()
        actor18.destroy()
        actor19.destroy()
        actor20.destroy()
        actor21.destroy()
        actor22.destroy()
        actor23.destroy()
        actor24.destroy()
        actor25.destroy()
        actor26.destroy()
        actor27.destroy()

        actor28.destroy()
        actor29.destroy()
        actor30.destroy()
        # actor31.destroy()
        actor32.destroy()
        actor33.destroy()
        actor34.destroy()
        actor35.destroy()
        actor36.destroy()
        actor37.destroy()
        actor38.destroy()



if __name__ == "__main__":
    spawn_npc()
