import os
import sys
import time
from typing import DefaultDict

from torch.utils.tensorboard import SummaryWriter

class BenchmarkWriter(SummaryWriter):
    '''SummaryWriter that can summarise average time for steps.'''

    def __init__(self, log_dir=None, **kwargs):
        # [number of GPUs]x[GPU type]
        self.main_tag = "x".join(os.environ["SLURM_GPUS_PER_NODE"].split(":")[::-1])
        if log_dir is None:
            main_file = sys.argv[0]
            if main_file[-3:]==".py":
                main_file = main_file[:-3]
            log_dir = f"logs/{main_file}_{self.main_tag}"
        super().__init__(log_dir=log_dir, **kwargs)
        self.scalars = DefaultDict(lambda: DefaultDict(list))
    
    def add_scalar(
        self,
        tag,
        scalar_value,
        global_step=None,
        walltime=None,
    ):
        self.scalars[self.main_tag][tag].append(scalar_value)
        self.scalars[self.main_tag]["global_step"].append(
            int(global_step)
            if global_step is not None
            else len(self.scalars[self.main_tag]["global_step"])
        )
        self.scalars[self.main_tag]["walltime"].append(
            walltime
            if walltime is not None
            else time.perf_counter()
        )
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
        )
    
    def add_scalars(self, tag_scalar_dict, global_step=None, walltime=None):
        for tag, scalar_value in tag_scalar_dict.items():
            self.scalars[self.main_tag][tag].append(scalar_value)
        self.scalars[self.main_tag]["global_step"].append(
            int(global_step)
            if global_step is not None
            else len(self.scalars[self.main_tag]["global_step"])
        )
        self.scalars[self.main_tag]["walltime"].append(
            walltime
            if walltime is not None
            else time.perf_counter()
        )
        super().add_scalars(
            self.main_tag,
            tag_scalar_dict,
            global_step=None,
            walltime=None,
        )
    
    def benchmark_results(self, burn_in_steps=0, step_unit="step"):
        steps = self.scalars[self.main_tag]["global_step"][burn_in_steps:]
        times = self.scalars[self.main_tag]["walltime"][burn_in_steps:]
        steps, times = sorted((s, t) for s, t in zip(steps, times) if s >= burn_in_steps)
        n_steps = steps[-1] - steps[0]
        tot_time = times[-1] - times[0]
        print(f"{self.main_tag}: {tot_time / n_steps} s/{step_unit}")
