class Logger():
    def __init__(self,
                 config,
                 name = None
                 ):
        if name == None:
            from datetime import datetime
            now = datetime.now().strftime("%m_%d-%H_%M")
            print('log name is ', config['run_config']['log_name']+'_'+config['run_config']['dataset'] + '_' + now)
            self.name = config['run_config']['log_name']+'_'+config['run_config']['dataset'] + '_' + now
        else:
            self.name = name
        self.LOG = [{},{}]
        for i in range(2):
            self.LOG[i]['name'] = self.name
            self.LOG[i]['used_metric'] = config['run_config']['metric']
            self.LOG[i]['loss'] = []
            self.LOG[i]['train_metric'] = []
            self.LOG[i]['test_metric'] = []
            self.LOG[i]['best_loss'] = 10000000000
            self.LOG[i]['best_train_metric'] = -100000
            self.LOG[i]['best_test_metric'] = -100000
            self.LOG[i]['best_epoch'] = -1
        
    def update(self, epoch, loss, train_metric_ACC, test_metric_ACC, train_metric_AUC, test_metric_AUC):
        # metrics: [(train_metric, test_metric)]
        self.LOG[0]['loss'].append(loss)
        self.LOG[0]['train_metric'].append(train_metric_ACC)
        self.LOG[0]['test_metric'].append(test_metric_ACC)
        if test_metric_ACC >= self.LOG[0]['best_test_metric']:
            self.LOG[0]['best_loss'] = loss
            self.LOG[0]['best_train_metric'] = train_metric_ACC
            self.LOG[0]['best_test_metric'] = test_metric_ACC
            self.LOG[0]['best_epoch'] = epoch
        # metrics: [(train_metric, test_metric)]
        self.LOG[1]['loss'].append(loss)
        self.LOG[1]['train_metric'].append(train_metric_AUC)
        self.LOG[1]['test_metric'].append(test_metric_AUC)
        if test_metric_AUC >= self.LOG[1]['best_test_metric']:
            self.LOG[1]['best_loss'] = loss
            self.LOG[1]['best_train_metric'] = train_metric_AUC
            self.LOG[1]['best_test_metric'] = test_metric_AUC
            self.LOG[1]['best_epoch'] = epoch
    
    def get_best(self,i):
        return self.LOG[i]['name'], self.LOG[i]['best_loss'], self.LOG[i]['best_train_metric'], self.LOG[i]['best_test_metric'], self.LOG[i]['best_epoch']
    
    def save(self):
        import json
        import os 
        path = os.path.join('./logs', self.name)
        # check if the path exists
        if not os.path.exists(path):
            print('creating path ', path)
            os.makedirs(path)
        path = os.path.join(path, 'log.json')
        
        # overwrite the file
        with open(path, 'w') as f:
            json.dump(self.LOG, f)
            print('log file saved to ', path)
        
    
    def load(self, path):
        import json
        with open(path, 'r') as f:
            self.LOG_loaded = json.load(f)
            print('log file loaded from ', path)
    
    def plot_metric(self):
        import matplotlib.pyplot as plt
        
        plt.plot(self.LOG['train_metric'], label = self.LOG['used_metric']+'_train')
        plt.plot(self.LOG['test_metric'], label = self.LOG['used_metric']+'_test')
        plt.title(self.LOG['name'])
        plt.legend()
        plt.show()
        # save the plot
        import os
        path = os.path.join('./logs', self.name)
        # check if the path exists
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, 'metric_log.png')
        # check if the file exists
        if os.path.exists(path):
            raise ValueError('plot file already exists')
        plt.savefig(path)
        
    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(self.LOG['loss'], label = 'loss')
        plt.title(self.LOG['name'])
        plt.legend()
        plt.show()
        # save the plot
        import os
        path = os.path.join('./logs', self.name)
        # check if the path exists
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, 'loss_log.png')
        # check if the file exists
        if os.path.exists(path):
            raise ValueError('plot file already exists')
        plt.savefig(path)
