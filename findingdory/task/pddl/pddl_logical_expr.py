# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Dict, List, Optional, Union

from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    PddlEntity,
    PddlSimInfo,
)
from habitat.tasks.rearrange.multi_task.pddl_logical_expr import (
    LogicalExprType,
    LogicalQuantifierType,
    LogicalExpr,
)

from findingdory.task.pddl.pddl import FindingDoryPddlEntity
from findingdory.task.pddl.pddl_predicate import FindingDoryPredicate


class FindingDoryLogicalExprType(Enum):
    """Enum for types of logical expressions supported in FindingDory."""
    AND = LogicalExprType.AND.value
    NAND = LogicalExprType.NAND.value
    OR = LogicalExprType.OR.value
    NOR = LogicalExprType.NOR.value
    AND_SEQUENTIAL = "and_sequential"
    AND_SEQUENTIAL_ORDERED = "and_sequential_ordered"


class FindingDoryLogicalQuantifierType(Enum):
    """Enum for types of logical quantifiers supported in FindingDory."""
    FORALL = LogicalQuantifierType.FORALL.value
    EXISTS = LogicalQuantifierType.EXISTS.value
    FORALL_SEQUENTIAL_UNORDERED = "forall_sequential_unordered"
    FORALL_SEQUENTIAL_ORDERED = "forall_sequential_ordered"


class FindingDoryLogicalExpr(LogicalExpr):
    """
    Extended LogicalExpr class that adds functionality to cycle through truth value checks
    for matched entities from the simulator. This is required for tasks that involve 
    sequentially reaching multiple goals where truth values are satisfied at different 
    steps during the episode rollout.
    """

    def __init__(
        self,
        expr_type: FindingDoryLogicalExprType,
        sub_exprs: List[Union["LogicalExpr", Predicate]],
        inputs: List[FindingDoryPddlEntity],
        quantifier: Optional[FindingDoryLogicalQuantifierType],
        sequential_goals: Optional[Dict] = None,
        connected_recep_names: Optional[Dict] = None,
        pddl_entities: Optional[Dict] = None,
    ):
        """
        Initialize a FindingDoryLogicalExpr instance.

        Args:
            expr_type: Type of logical expression
            sub_exprs: List of sub-expressions or predicates
            inputs: List of input entities
            quantifier: Type of quantifier if any
            sequential_goals: Dictionary containing goal validation sequence
            connected_recep_names: Dictionary containing episode interacted receptacle names and their connected receptacle surfaces
            pddl_entities: Dictionary containing all PDDL entities in the current simulator state
        """
        super().__init__(expr_type, sub_exprs, inputs, quantifier)
        
        # Store the desired goal validation sequence as a list of integers
        if sequential_goals is not None:
            self._goal_validation_sequence = sequential_goals['sub_expr_sequence']
        
        # Store info regarding various surfaces of a particular receptacle
        self._connected_recep_names = connected_recep_names
        self._pddl_entities = pddl_entities
        
        self._permanently_false = False
        self._num_subgoals_satisfied = 0

    def is_true(self, sim_info: PddlSimInfo) -> bool:
        """Evaluate if the logical expression is true given simulator info."""
        return self._is_true(lambda p: p.is_true(sim_info))

    def _is_true(self, is_true_fn) -> bool:
        # For all non-AND_SEQUENTIAL types, reinitialize truth values
        if self._expr_type not in (FindingDoryLogicalExprType.AND_SEQUENTIAL, FindingDoryLogicalExprType.AND_SEQUENTIAL_ORDERED):
            self._truth_vals = [None] * len(self._sub_exprs)
        
        if self._expr_type in (FindingDoryLogicalExprType.AND, FindingDoryLogicalExprType.NAND):
            result = True
            for i, sub_expr in enumerate(self._sub_exprs):
                truth_val = is_true_fn(sub_expr)
                
                self._is_bool(truth_val)
                self._truth_vals[i] = truth_val
                result = result and truth_val
                if not result:
                    break
        elif self._expr_type in (FindingDoryLogicalExprType.OR, FindingDoryLogicalExprType.NOR):
            result = False
            for i, sub_expr in enumerate(self._sub_exprs):
                truth_val = is_true_fn(sub_expr)
                
                self._is_bool(truth_val)
                self._truth_vals[i] = truth_val
                result = result or truth_val
                if result:
                    break
        
        elif self._expr_type == FindingDoryLogicalExprType.AND_SEQUENTIAL:
            # For AND_SEQUENTIAL, we want to persist previously satisfied subgoals.
            # Initialize _truth_vals only if it is not already set or its length is mismatched.
            if (not hasattr(self, '_truth_vals') or self._truth_vals is None or
                    len(self._truth_vals) != len(self._sub_exprs)):
                self._truth_vals = [False] * len(self._sub_exprs)

            result = True
            if self._permanently_false:
                result = False
            else:
                # This flag ensures that only one new subgoal becomes true per evaluation.
                new_subgoal_marked = False
                for i, sub_expr in enumerate(self._sub_exprs):
                    current_val = is_true_fn(sub_expr)
                    
                    # Check if the current subgoal is satisfied by considering adjacent connected receptacles as interacted goals
                    cur_recep_name = sub_expr.sub_exprs[0]._arg_values[0].name
                    if not current_val and cur_recep_name in self._connected_recep_names:
                        connected_recep_true = self._check_if_connected_recep_is_true(
                            sub_expr,
                            self._pddl_entities[cur_recep_name],
                            self._connected_recep_names[cur_recep_name],
                            is_true_fn
                        )
                        current_val = current_val or connected_recep_true

                    self._is_bool(current_val)
                    # If this subgoal was already locked in as True, keep it.
                    if self._truth_vals[i]:
                        truth_val = True
                    else:
                        # Otherwise, if it evaluates to True now and no new subgoal has been locked in this round, lock it.
                        if current_val and not new_subgoal_marked:
                            truth_val = True
                            new_subgoal_marked = True
                        else:
                            truth_val = False
                    self._truth_vals[i] = truth_val
                    result = result and truth_val
        
        elif self._expr_type == FindingDoryLogicalExprType.AND_SEQUENTIAL_ORDERED:
            # Initialize persistent truth values if not already set.
            if not hasattr(self, '_truth_vals') or self._truth_vals is None or len(self._truth_vals) != len(self._sub_exprs):
                self._truth_vals = [False] * len(self._sub_exprs)
                
            result = True
            # If permanently set to False, short-circuit evaluation.
            if self._permanently_false:
                result = False
            else:
                new_subgoal_marked = False  # Flag to ensure only one new subgoal is locked per evaluation.
                # Iterate through subgoals in the defined order.
                for idx, expected_index in enumerate(self._goal_validation_sequence):
                    # If a subgoal was already locked in in a previous evaluation, preserve its truth.
                    if self._truth_vals[expected_index]:
                        truth_val = True
                    else:
                        # Evaluate the current subgoal.
                        truth_val = is_true_fn(self._sub_exprs[expected_index])
                        
                        # Check if the current subgoal is satisfied by considering adjacent connected receptacles as interacted goals
                        cur_recep_name = self._sub_exprs[expected_index].sub_exprs[0]._arg_values[0].name
                        if not truth_val and cur_recep_name in self._connected_recep_names:
                            connected_recep_true = self._check_if_connected_recep_is_true(
                                self._sub_exprs[expected_index],
                                self._pddl_entities[cur_recep_name],
                                self._connected_recep_names[cur_recep_name],
                                is_true_fn
                            )
                            truth_val = truth_val or connected_recep_true
                        
                        self._is_bool(truth_val)
                        
                        # If this subgoal evaluates to True:
                        if truth_val:
                            if not new_subgoal_marked:
                                # Lock in the first new subgoal this round.
                                new_subgoal_marked = True
                            else:
                                # If a new subgoal was already locked in during this evaluation,
                                # ignore this simultaneous True by forcing it to False.
                                truth_val = False
                        
                        # Enforce ordering: for any subgoal beyond the first, its immediate predecessor in the sequence must be locked in.
                        if idx > 0 and truth_val and not self._truth_vals[self._goal_validation_sequence[idx - 1]]:
                            # This subgoal is evaluated as True before its predecessor is locked in – out of order.
                            # Mark the entire expression permanently false and break.
                            self._permanently_false = True
                            result = False
                            break                            
                           
                    self._truth_vals[expected_index] = truth_val
                    if not truth_val:
                        result = False  # The overall result is only True if all subgoals in order are locked in.
    
        else:
            raise ValueError(
                f"Got unexpected expr_type: {self._expr_type} of type {type(self._expr_type)}"
            )

        if (
            self._expr_type == FindingDoryLogicalExprType.NAND
            or self._expr_type == FindingDoryLogicalExprType.NOR
        ):
            # Invert the entire result for NAND and NOR expressions.
            result = not result

        return result
    
    def _is_bool(self, value):
        if not isinstance(value, bool):
            raise ValueError(
                f"Predicate returned non truth value: {value=}, {type(value)=}"
            )
        return value

    def sub_in_clone(self, sub_dict: Dict[PddlEntity, PddlEntity]) -> "FindingDoryLogicalExpr":
        """Create a clone with substituted entities."""
        return FindingDoryLogicalExpr(
            self._expr_type,
            [e.sub_in_clone(sub_dict) for e in self._sub_exprs],
            self._inputs,
            self._quantifier,
        )
        
    def _check_if_connected_recep_is_true(self, orig_logical_expr, orig_pddl_entity, connected_recep_names, is_true_fn):
        '''
        Check for PDDL success at each of the connected receptacles by creating a new PDDL expression in which the connected receptacle entity properties are exactly the same as the targetinteracted receptacle entity properties.
        We need to do this because the connected receptacle entities are marked as non-interacted for multi-goal tasks to prevent explicit PDDL success verification at each receptacle.
        '''
        truth_val = False
        for recep_name in connected_recep_names:
            connected_pddl_entity = self._pddl_entities[recep_name]     # NOTE: The connected receptacle entities are marked as non-interacted for multi-goal tasks to prevent explicit PDDL success verification at each receptacle 
            
            # Update the connected receptacle entity with interaction properties of the original interacted receptacle entity for on-the-fly PDDL verification
            connected_pddl_entity_with_interaction = FindingDoryPddlEntity(
                connected_pddl_entity.name,
                connected_pddl_entity.expr_type,
                orig_pddl_entity.properties
            )
            
            # Create the PDDL expression which will be used for on-the-fly PDDL verification
            new_expr = orig_logical_expr.sub_in_clone(
                sub_dict={orig_pddl_entity : connected_pddl_entity_with_interaction}
            )
            truth_val = truth_val or is_true_fn(new_expr)
            
        return truth_val