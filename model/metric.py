def select_metric(metric_name, device = 'cpu', num_class = None):
    if metric_name == 'binary_AUC':
        return binary_AUC(device = device)
    if metric_name == 'ACC':
        if num_class is None:
            raise ValueError('num_class should be given when using ACC as metric')
        return ACC(num_class = num_class, device = device)
    else:
        raise NotImplementedError(f'{metric_name} is not implemented, please check the metric name')
    
class binary_AUC():
    def __init__(self, device = 'cpu') :
        from torcheval.metrics import BinaryAUROC
        self.name = 'AUC'
        self.method = BinaryAUROC().to(device)
        
    def update(self, pred, label):
        '''
        pred: [batch_size]
        label: [batch_size]
        '''
        if len(pred.shape) == 2:
            pred = pred[:,1]
        self.method.update(pred, label)
        
    def compute(self):
        return self.method.compute()
    
    def reset(self):
        self.method.reset()

class ACC():
    def __init__(self, num_class = None, device = 'cpu') :
        from torcheval.metrics import MulticlassAccuracy
        self.name = 'ACC'
        self.device = device
        self.num_class = num_class
        self.method = MulticlassAccuracy(num_classes = num_class).to(device)
    def update(self, pred, label):
        '''
        pred: [batch_size, num_class]
        label: [batch_size]
        '''
        self.method.update(pred, label)
    def compute(self):
        return self.method.compute()
    def reset(self):
        self.method.reset()