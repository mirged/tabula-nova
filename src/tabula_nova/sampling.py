"""Deterministic rank-partitioned row sampling with O(1) memory."""

import math
import random

class StepSampler:
    """O(1) memory affine permutations, indexed by global optimizer step.

    Every rank takes a disjoint slice within an epoch. Crossing an epoch boundary
    may repeat rows in one update; no DistributedSampler padding duplicates arise.
    A composite multiplier is advanced until coprime with dataset length.
    """
    def __init__(self, size, config, rank=0, world_size=1, start_step=0):
        if size < world_size * config.micro_batch_size:
            raise ValueError("dataset too small for one global microbatch")
        self.size, self.config = size, config
        self.rank, self.world_size, self.start_step = rank, world_size, start_step

    def __len__(self):
        return (self.config.max_steps-self.start_step)*self.config.accumulation_steps*self.config.micro_batch_size

    def __iter__(self):
        c = self.config
        count = c.micro_batch_size*c.accumulation_steps
        current_epoch, multiplier, shift = -1, 1, 0
        for local in range(self.start_step*count, c.max_steps*count):
            micro, slot = divmod(local, c.micro_batch_size)
            position = micro*self.world_size*c.micro_batch_size + self.rank*c.micro_batch_size + slot
            epoch, offset = divmod(position, self.size)
            if epoch != current_epoch:
                rng = random.Random(c.seed+epoch)
                multiplier = rng.randrange(1, self.size+1)
                while math.gcd(multiplier, self.size) != 1:
                    multiplier = multiplier % self.size + 1
                shift, current_epoch = rng.randrange(self.size), epoch
            yield (multiplier*offset+shift) % self.size

