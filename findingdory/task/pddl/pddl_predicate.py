# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, Dict, List, Optional

from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate
from findingdory.task.pddl.pddl import FindingDoryPddlEntity, FindingDoryPddlSimInfo

class FindingDoryPredicate(Predicate):

    def is_true(self, sim_info: FindingDoryPddlSimInfo) -> bool:
        """
        Returns if the predicate is satisfied in the current simulator state.
        Potentially returns the cached truth value of the predicate depending
        on `sim_info`.
        """
        self_repr = repr(self)
        if (
            sim_info.pred_truth_cache is not None
            and self_repr in sim_info.pred_truth_cache
        ):  
            # If the current task requires moving to different goals sequentially, we cache the truth value when it becomes True
            if isinstance(sim_info.sequential_goals, Dict):
                if sim_info.pred_truth_cache[self_repr]:
                    return sim_info.pred_truth_cache[self_repr]
            else:
                # Return the cached value.
                return sim_info.pred_truth_cache[self_repr]

        # Recompute and potentially cache the result.
        if self._is_valid_fn is None:
            result = True
        else:
            result = self._is_valid_fn(
                sim_info=sim_info, **self._create_kwargs()
            )
        
        # For predicates which require checking if robot has visited some specific receptacle/object, we also check if the robot invoked an intermediate stop action
        if self._name == "robot_at_object" or self._name == "robot_at_receptacle" or self._name == "robot_in_room":
            # Check if PDDL intermediate stop action has been invoked
            result = result and sim_info.does_want_intermediate_stop
                    
        if sim_info.pred_truth_cache is not None:
            sim_info.pred_truth_cache[self_repr] = result
        return result

    def clone(self):
        p = FindingDoryPredicate(
            self._name, self._is_valid_fn, self._set_state_fn, self._args
        )
        if self._arg_values is not None:
            p.set_param_values(self._arg_values)
        return p
    
    def sub_in_clone(self, sub_dict: Dict[FindingDoryPddlEntity, FindingDoryPddlEntity]):
        p = FindingDoryPredicate(
            self._name,
            self._is_valid_fn,
            self._set_state_fn,
            self._args,
        )
        if self._arg_values is not None:
            p.set_param_values(
                [sub_dict.get(entity, entity) for entity in self._arg_values]
            )
        return p