import ctypes


class TagPositioning(object):
    class UWBMsg(ctypes.Structure):
        _fields_ = [("x", ctypes.c_double),
                    ("y", ctypes.c_double),
                    ("z", ctypes.c_double)]
        
    def __init__(self, tag_id:int) -> None:
        self.tag_id = tag_id
        # change the path on your local machine
        self.positioning = ctypes.cdll.LoadLibrary('./trilateration.so')
        self.distanceArray = (ctypes.c_int * 8)(-1)
        self.current_location = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0
        }

    def positioning_service(self, tag_data:dict, anchorArray:list) -> dict:
        anchor_id = tag_data.get("AnchorID")
        distance = tag_data.get("Distance")
        self.distanceArray[anchor_id] = int(distance*1000)
        for i in range(8):
            if self.distanceArray[i] == -1:
                return self.current_location

        location = self.UWBMsg()
        _ = self.positioning.GetLocation(ctypes.byref(location), anchorArray, self.distanceArray)
        self.current_location = {
            "x": location.x,
            "y": location.y,
            "z": location.z
        }

        print("anchorId: ", anchor_id, "distance: ", distance, "location: ", location.x, location.y, location.z)

        return self.current_location