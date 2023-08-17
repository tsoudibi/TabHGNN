from utils.utils import *

def select_metric(metric_name, device = 'cpu'):
    if metric_name == 'binary_AUC':
        return binary_AUC(device = device)
    if metric_name == 'ACC':
        return ACC(device = device)
    if metric_name == 'R2':
        return R2(device = device)
    else:
        raise NotImplementedError(f'{metric_name} is not implemented, please check the metric name')
    
class binary_AUC():
    def __init__(self, device = 'cpu') :
        from torcheval.metrics import BinaryAUROC
        self.name = 'AUC'
        self.method = BinaryAUROC().to(device)
        # check if the model is binary classification
        if get_num_classes() != 2:
            raise ValueError('binary_AUC can only be used in binary classification task, getting num_class = ', get_num_classes())
        
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
    def __init__(self, device = 'cpu') :
        from torcheval.metrics import MulticlassAccuracy
        self.name = 'ACC'
        self.method = MulticlassAccuracy(num_classes = get_num_classes()).to(device)
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
        
class R2():
    def __init__(self, device = 'cpu'):
        from torcheval.metrics import R2Score
        self.name = 'R2'
        self.method = R2Score().to(device)
        if get_task() != 'regression':
            raise ValueError('R2 can only be used in regression task, getting task = ', get_task())
        pass
    def update(self, pred, label):
        '''
        pred: [batch_size]
        label: [batch_size]
        '''
        self.method.update(pred, label)
    def compute(self):
        return self.method.compute()
    def reset(self):
        self.method.reset()