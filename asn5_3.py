from djitellopy import Tello

def main():
    print('Call any method here to run')
    


def droneStraight():
    tello = Tello()
    tello.connect()
    print(tello.get_battery())
    tello.takeoff()

    tello.move_forward(100)

    tello.land()

def droneSquareFacingForward():
    tello = Tello()

    tello.connect()
    tello.takeoff()

    tello.move_forward(100)
    tello.move_left(100)
    tello.move_back(100)
    tello.move_right(100)

    tello.land()

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

def droneSquareTurning():
    turningFunction(4, 100)

if __name__ == "__main__":
    main()