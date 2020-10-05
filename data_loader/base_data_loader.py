
class BaseDataLoader():
    def __init__(self):
        pass
    
    def initialize(self, opt):
        self.opt = opt
        self.opt.loadSize = 32
        self.opt.batchSize = 32
        self.opt.serial_batches = 0
        self.opt.nThreads = 2
        self.opt.max_dataset_size=float("inf")
        pass

    def load_data():
        return None

        
        
