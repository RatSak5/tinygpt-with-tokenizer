import torch


class Adam:
    def __init__(self, parameters, lr, beta1, beta2, eps):
        self.label = "adam"
        
        self.parameters = parameters
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 1
        self.accum_mt = [torch.zeros_like(p) for p in parameters]
        self.accum_vt = [torch.zeros_like(p) for p in parameters]
        
    def step(self):
        for i in range(len(self.parameters)):
            if self.parameters[i].grad is None:
                continue
            self.accum_mt[i] = self.beta1 * self.accum_mt[i] + (1 - self.beta1) * self.parameters[i].grad
            self.accum_vt[i] = self.beta2 * self.accum_vt[i] + (1 - self.beta2) * self.parameters[i].grad ** 2

            mt_hat = self.accum_mt[i] / (1 - self.beta1 ** self.t)
            vt_hat = self.accum_vt[i] / (1 - self.beta2 ** self.t)

            update = (mt_hat / (vt_hat ** 0.5 + self.eps))
            self.parameters[i].data += - self.lr * update
        self.t += 1
            
    def zero_grad(self, set_to_none=False):
        for p in self.parameters:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
    def describe(self):
        print(f"""
Optimizer    : {self.label}
              Learning rate: {self.lr}
              Eps          : {self.eps}
              Beta 1       : {self.beta1}
              Beta 2       : {self.beta2}      
              """)
    
    def parameters(self):
        return []

class AdaGrad:
    def __init__(self, parameters, lr, eps):
        self.label = "adagrad"
        
        self.parameters = parameters
        self.lr = lr
        self.eps = eps
        self.t = 0
        self.accum_grad = [torch.zeros_like(p) for p in parameters]
        
    def step(self):
        for i in range(len(self.parameters)):
            if self.parameters[i].grad is None:
                continue
            self.accum_grad[i] = self.accum_grad[i] + self.parameters[i].grad ** 2
            update = (self.parameters[i].grad / (self.accum_grad[i] + self.eps) ** 0.5)
            self.parameters[i].data += - self.lr * update
        self.t += 1
            
    def zero_grad(self, set_to_none=False):
        for p in self.parameters:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def describe(self):
        print(f"""
Optimizer    : {self.label}
              Learning rate: {self.lr}
              Eps          : {self.eps}     
              """)

class RMSprop:
    def __init__(self, parameters, lr, gamma, eps):
        self.label = "rmsprop"
        
        self.parameters = parameters
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.t = 0
        self.accum_grad_exp = [torch.zeros_like(p) for p in parameters]
        
    def step(self):
        for i in range(len(self.parameters)):
            if self.parameters[i].grad is None:
                continue
            self.accum_grad_exp[i] = self.gamma * self.accum_grad_exp[i] + (1 - self.gamma) * self.parameters[i].grad ** 2
            update = (self.parameters[i].grad / (self.accum_grad_exp[i] + self.eps) ** 0.5)
            self.parameters[i].data = self.parameters[i].data - self.lr * update
            self.t += 1
            
    def zero_grad(self, set_to_none=False):
        for p in self.parameters:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def describe(self):
        print(f"""
Optimizer    : {self.label}
              Learning rate: {self.lr}
              Eps          : {self.eps}
              Gamma        : {self.gamma}
              """)

class SGD:
    def __init__(self, parameters, lr, momentum, dampening = 1):
        self.label = "sgd"
        
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.t = 1
        self.velocities = [torch.zeros_like(p) for p in parameters]
        
    def step(self):
        momentum = self.momentum if self.t > 1 else 1
        for i in range(len(self.parameters)):
            if self.parameters[i].grad is None:
                continue
            self.velocities[i] = (1 - self.dampening) * self.parameters[i].grad + momentum * self.velocities[i]
            update = self.velocities[i]
            self.parameters[i].data += - self.lr * update
        self.t += 1
            
    def zero_grad(self, set_to_none=False):
        for p in self.parameters:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
                
    def describe(self):
        print(f"""
Optimizer    : {self.label}
              Learning rate: {self.lr}
              Momentum     : {self.momentum}
              Dampening    : {self.dampening}
              """)

class AdamW:
    def __init__(self, parameters, lr, beta1, beta2, eps, weight_decay=0.1):
        self.label = "adamw"
 
        self.parameters = parameters
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.t = 1
        self.accum_mt = [torch.zeros_like(p) for p in parameters]
        self.accum_vt = [torch.zeros_like(p) for p in parameters]
 
    def step(self):
        for i in range(len(self.parameters)):
            if self.parameters[i].grad is None:
                continue
 
            self.accum_mt[i] = self.beta1 * self.accum_mt[i] + (1 - self.beta1) * self.parameters[i].grad
            self.accum_vt[i] = self.beta2 * self.accum_vt[i] + (1 - self.beta2) * self.parameters[i].grad ** 2
 
            mt_hat = self.accum_mt[i] / (1 - self.beta1 ** self.t)
            vt_hat = self.accum_vt[i] / (1 - self.beta2 ** self.t)
 
            update = (mt_hat / (vt_hat ** 0.5 + self.eps))

            if self.weight_decay > 0.0 and self.parameters[i].dim() >= 2:
                self.parameters[i].data -= self.lr * self.weight_decay * self.parameters[i].data
 
            self.parameters[i].data += - self.lr * update
        self.t += 1
 
    def zero_grad(self, set_to_none=False):
        for p in self.parameters:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
 
    def describe(self):
        print(f"""
Optimizer    : {self.label}
              Learning rate: {self.lr}
              Eps          : {self.eps}
              Beta 1       : {self.beta1}
              Beta 2       : {self.beta2}
              Weight decay : {self.weight_decay}
              """)
 
    def parameters(self):
        return []
