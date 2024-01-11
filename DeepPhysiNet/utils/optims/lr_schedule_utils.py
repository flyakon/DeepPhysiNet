'''
@Project : SSLRemoteSensing 
@File    : lr_schedule_utils.py
@Author  : Wenyuan Li
@Date    : 2021/5/16 14:56 
@Desc    :  
'''
import torch
from bisect import bisect_right
import warnings

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1 / 3,
            warmup_iters=100,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs]

class WarmupStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self,optimizer,
            milestones,
            start_epoch=400,
            step_size=200,
            step_gamma=0.9,
            warmup_gamma=0.1,
            warmup_factor=0.1,
            warmup_iters=100,
            warmup_method="linear",
            last_epoch=-1):
        self.stepLR=torch.optim.lr_scheduler.StepLR(optimizer,step_size,step_gamma)
        self.warmup_factor=warmup_factor
        self.base_lr=optimizer.param_groups[0]['lr']
        self.start_lr=self.base_lr*self.warmup_factor
        # print(self.base_lr)
        # self.warmupLR=WarmupMultiStepLR(optimizer,milestones,warmup_gamma,
        #     warmup_factor,
        #     warmup_iters,
        #     warmup_method=warmup_method,
        #     last_epoch=last_epoch)
        self.start_epoch=start_epoch
        super(WarmupStepLR, self).__init__(optimizer,last_epoch)

    def get_lr(self) -> float:
        # print(self.last_epoch,self.base_lr)
        if self.last_epoch<self.start_epoch:
            lr=self.start_lr+(self.last_epoch/self.start_epoch)*(self.base_lr-self.start_lr)

            return [lr]
        else:
            return self.stepLR.get_lr()

    def step(self, epoch=None) -> None:
        if epoch is None:
            self.last_epoch=0
        else:

            self.last_epoch=epoch
        if self.last_epoch<self.start_epoch:

            # Raise a warning if old pattern is detected
            # https://github.com/pytorch/pytorch/issues/20124
            if self._step_count == 1:
                if not hasattr(self.optimizer.step, "_with_counter"):
                    warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                                  "initialization. Please, make sure to call `optimizer.step()` before "
                                  "`lr_scheduler.step()`. See more details at "
                                  "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                                  UserWarning)

                # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
                elif self.optimizer._step_count < 1:
                    warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                                  "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                                  "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                                  "will result in PyTorch skipping the first value of the learning rate schedule. "
                                  "See more details at "
                                  "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                                  UserWarning)
            self._step_count += 1

            class _enable_get_lr_call:

                def __init__(self, o):
                    self.o = o

                def __enter__(self):
                    self.o._get_lr_called_within_step = True
                    return self

                def __exit__(self, type, value, traceback):
                    self.o._get_lr_called_within_step = False

            with _enable_get_lr_call(self):
                if epoch is None:
                    self.last_epoch += 1
                    values = self.get_lr()
                else:
                    self.last_epoch = epoch
                    if hasattr(self, "_get_closed_form_lr"):
                        values = self._get_closed_form_lr()
                    else:
                        values = self.get_lr()

            for i, data in enumerate(zip(self.optimizer.param_groups, values)):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        else:
            self.stepLR.step(epoch-self.start_epoch)




