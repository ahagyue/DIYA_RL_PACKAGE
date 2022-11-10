class EpsilonGenerator:
    def __init__(self, start = 1.0, end = 0.1, frame_num: int = 1e4, ftype : str = "linear"):
        self.start = start
        self.end = end
        self.frame_num = frame_num
        self.ftype = ftype
    
    def epsilon(self, frame: int) -> float:
        if self.ftype == "linear":
            return (self.end - self.start) * frame / self.frame_num + self.start
        
        if self.ftype == "exp":
            return (self.end / self.start) ** (frame / self.frame_num) * self.start