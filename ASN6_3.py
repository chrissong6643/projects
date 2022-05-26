from djitellopy import Tello

def main():
    print('Call any method here to run')
    
def turningFunction(sides, distance):
    tello = Tello()
    angle = 360 // sides
    tello.connect()
    print(tello.get_battery())
    tello.takeoff()

    for i in range(sides):
        tello.move_forward(distance)
        tello.rotate_counter_clockwise(angle)

    tello.land()    

def hexagon(distance):
    turningFunction(6, distance)

def star(distance):
    tello = Tello()

    tello.connect()
    tello.takeoff()

    for i in range(5):
        tello.move_forward(distance)
        tello.rotate_clockwise(72)
        tello.move_forward(distance)
        tello.rotate_counter_clockwise(144)

    tello.land()

def stairCase(height, distance):
    tello = Tello()

    tello.connect()
    tello.takeoff()

    for i in range(2):
        tello.move_forward(distance)
        tello.move_up(height)
    
    tello.move_forward(distance)

    tello.land()



if __name__ == "__main__":
    main()