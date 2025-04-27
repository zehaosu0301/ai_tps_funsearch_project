# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for sampling new programs."""
from __future__ import annotations

from collections.abc import Collection, Sequence
import time
from typing import Collection, Sequence, Type
import numpy as np
import os
import openai
from abc import ABC, abstractmethod

from implementation import evaluator
from implementation import programs_database


class LLM(ABC):
    """Language model that predicts continuation of provided source code."""

    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """Returns a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]
    

class Sampler:
    """Node that samples program continuations and sends them for analysis."""
    _global_samples_nums: int = 1 
    def __init__(
        self,
        database: programs_database.ProgramsDatabase,
        evaluators: Sequence[evaluator.Evaluator],
        samples_per_prompt: int,
        llm_class: Type[LLM] = LLM,
        max_sample_nums: int | None = None,
    ):
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self._samples_per_prompt = samples_per_prompt

    def sample(self, **kwargs):
        """Continuously gets prompts, samples programs, sends them for analysis.
        """
        while True:
            # stop the search process if hit global max sample nums
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break
            try:
                prompt = self._database.get_prompt()
                reset_time = time.time()
                
                # Add timeout handling and better error handling for sample generation
                try:
                    samples = self._llm.draw_samples(prompt.code)
                except Exception as e:
                    print(f"Error during sample generation: {type(e).__name__}: {str(e)}")
                    # Add a delay to prevent tight loop in case of persistent errors
                    time.sleep(1)
                    continue
                    
                sample_time = (time.time() - reset_time) / self._samples_per_prompt
                print(f"Generated {len(samples)} samples in {sample_time:.2f}s per sample")
                
                # This loop can be executed in parallel on remote evaluator machines.
                for i, sample in enumerate(samples):
                    try:
                        self._global_sample_nums_plus_one()
                        cur_global_sample_nums = self._get_global_sample_nums()
                        chosen_evaluator = np.random.choice(self._evaluators)
                        
                        print(f"Processing sample #{cur_global_sample_nums} (sample {i+1}/{len(samples)})")
                        
                        # Add timeout handling for evaluation
                        analyse_result = chosen_evaluator.analyse(
                            sample,
                            prompt.island_id,
                            prompt.version_generated,
                            **kwargs,
                            global_sample_nums=cur_global_sample_nums,
                            sample_time=sample_time
                        )
                        
                        print(f"Completed analysis of sample #{cur_global_sample_nums}")
                        
                    except Exception as e:
                        print(f"Error during sample analysis: {type(e).__name__}: {str(e)}")
                        # Continue to next sample instead of restarting the whole loop
                        continue
                        
            except Exception as e:
                print(f"Error in sampling main loop: {type(e).__name__}: {str(e)}")
                # Add a delay to prevent tight loop in case of persistent errors
                time.sleep(2)

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1
