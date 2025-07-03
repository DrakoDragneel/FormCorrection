
class PlankDetector:
    def __init__(self):
        self.active = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def get_status(self):
        return {"active": self.active}
