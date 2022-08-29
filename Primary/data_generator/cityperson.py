
class CityPersonDataset(object):
    def __init__(self,data):
        self.Image_Height=float(data["imgHeight"])
        self.Image_Width=float(data["imgWidth"])
        self.Objects=data["objects"]

    def get_width(self):
        return self.Image_Width

    def get_height(self):
        return self.Image_Height

    def get_num_objects(self):
        return len(self.Objects)

    def get_label(self,idx):
       return self.Objects[idx]["label"]

    def get_visibile_box(self,idx):
        return self.Objects[idx]["bboxVis"]

    def get_bounding_box(self,idx):
        return self.Objects[idx]["bbox"]

    def InstanceId(self,idx):
        return self.Objects[idx]["instanceId"]
