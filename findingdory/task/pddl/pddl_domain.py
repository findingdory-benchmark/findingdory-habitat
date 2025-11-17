#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import itertools
import os.path as osp
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)
from collections import defaultdict

import yaml  # type: ignore[import]

from habitat.config.default import get_full_habitat_config_path
from habitat.tasks.rearrange.multi_task.pddl_action import PddlAction
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    ExprType,
    parse_func,
)
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlDomain, _parse_callable
from findingdory.task.pddl.pddl_predicate import FindingDoryPredicate

from findingdory.dataset.findingdory_dataset import FindingDoryEpisode
import findingdory.task.pddl.pddl as findingdory_pddl
from findingdory.task.pddl.pddl import (
    FindingDoryPddlSimInfo,
    FindingDorySimulatorObjectType,
    FindingDoryPddlEntity,
    TaskPosData,
)
from findingdory.task.pddl.pddl_logical_expr import (
    FindingDoryLogicalExpr,
    FindingDoryLogicalExprType,
    FindingDoryLogicalQuantifierType,
)
import findingdory.config as findingdory_config

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PddlDomainFindingDory(PddlDomain):
    """
    Manages the information from the PDDL domain and task definition.
    """

    def __init__(
        self,
        domain_file_path: str,
        cur_task_config: Optional["DictConfig"] = None,
        read_config: bool = True,
    ):
        """
        :param domain_file_path: Either an absolute path or a path relative to
            `habitat/task/rearrange/multi_task/domain_configs/`.
        :param cur_task_config: The task config (`habitat.task`). Needed if the
            PDDL system will set the simulator state.
        """
        self._sim_info: Optional[FindingDoryPddlSimInfo] = None
        self._config = cur_task_config
        self._orig_actions: Dict[str, PddlAction] = {}

        if not osp.isabs(domain_file_path):
            parent_dir = osp.dirname(__file__)
            domain_file_path = osp.join(
                parent_dir, "domain_configs", domain_file_path
            )

        if "." not in domain_file_path.split("/")[-1]:
            domain_file_path += ".yaml"

        with open(get_full_habitat_config_path(domain_file_path), "r") as f:
            domain_def = yaml.safe_load(f)

        self._added_entities: Dict[str, FindingDoryPddlEntity] = {}
        self._added_expr_types: Dict[str, ExprType] = {}

        self._parse_expr_types(domain_def)
        self._parse_constants(domain_def)
        self._parse_predicates(domain_def)
        self._parse_actions(domain_def)

    def add_permanent_expr_types(
        self,
        obj_cats: List[str],
        rec_cats: List[str],
        room_cats: List[str],
        agent_name: str,
    ) -> None:
        """
        Adds the permanent types to the domain. This is needed to define the
        types of objects and receptacles that are always present in the domain.
        """
        # Add permanent expr types. (Object types and receptacles).
        for cat in obj_cats:
            self._expr_types[cat] = ExprType(
                cat,
                self.expr_types[FindingDorySimulatorObjectType.MOVABLE_ENTITY.value]
            )

        for cat in rec_cats:
            self._expr_types[cat] = ExprType(
                cat,
                self.expr_types[FindingDorySimulatorObjectType.STATIC_RECEPTACLE_ENTITY.value]
            )

        # Add robot type.
        self._expr_types[agent_name] = ExprType(
            agent_name,
            self.expr_types[FindingDorySimulatorObjectType.ROBOT_ENTITY.value]
        )

        # Add position type.
        for k in TaskPosData().keys():
            self._expr_types[k] = ExprType(
                k,
                self.expr_types[FindingDorySimulatorObjectType.POSITIONAL_ENTITY.value]
            )
        
        # add room type
        for room in room_cats:
            self._expr_types[room] = ExprType(
                room,
                self.expr_types[FindingDorySimulatorObjectType.ROOM_ENTITY.value]
            )


    def _parse_predicates(self, domain_def) -> None:
        """
        Fetches the PDDL predicates into `self.predicates`.
        """

        self.predicates: Dict[str, FindingDoryPredicate] = {}
        for pred_d in domain_def["predicates"]:
            arg_entities = [
                FindingDoryPddlEntity(arg["name"], self.expr_types[arg["expr_type"]])
                for arg in pred_d["args"]
            ]

            if "set_state_fn" not in pred_d:
                set_state_fn = None
            else:
                set_state_fn = _parse_callable(pred_d["set_state_fn"])

            if "is_valid_fn" not in pred_d:
                is_valid_fn = None
            else:
                is_valid_fn = _parse_callable(pred_d["is_valid_fn"])

            pred = FindingDoryPredicate(
                pred_d["name"],
                is_valid_fn,
                set_state_fn,
                arg_entities,
            )
            self.predicates[pred.name] = pred

    def _parse_constants(self, domain_def) -> None:
        """
        Fetches the constants into `self._constants`.
        """

        self._constants: Dict[str, FindingDoryPddlEntity] = {}
        if domain_def["constants"] is None:
            return
        for c in domain_def["constants"]:
            self._constants[c["name"]] = FindingDoryPddlEntity(
                c["name"],
                self.expr_types[c["expr_type"]],
            )

    def _parse_expr_types(self, domain_def):
        """
        Fetches the types from the domain into `self._expr_types`.
        """
        # Always add the default `expr_types` from the simulator.
        base_entity = ExprType(FindingDorySimulatorObjectType.BASE_ENTITY.value, None)
        self._expr_types: Dict[str, ExprType] = {
            FindingDorySimulatorObjectType.BASE_ENTITY.value: base_entity
        }
        self._expr_types.update(
            {
                obj_type.value: ExprType(obj_type.value, base_entity)
                for obj_type in FindingDorySimulatorObjectType
                if obj_type.value != FindingDorySimulatorObjectType.BASE_ENTITY.value
            }
        )

        for parent_type, sub_types in domain_def["types"].items():
            if parent_type not in self._expr_types:
                self._expr_types[parent_type] = ExprType(
                    parent_type, base_entity
                )
            for sub_type in sub_types:
                if sub_type in self._expr_types:
                    self._expr_types[sub_type].parent = self._expr_types[
                        parent_type
                    ]
                else:
                    self._expr_types[sub_type] = ExprType(
                        sub_type, self._expr_types[parent_type]
                    )

    def parse_predicate(
        self,
        pred_str: str,
        existing_entities: Optional[Dict[str, FindingDoryPddlEntity]] = None,
    ) -> FindingDoryPredicate:
        """
        Instantiates a predicate from call in string such as "in(X,Y)".
        :param pred_str: The string to parse such as "in(X,Y)".
        :param existing_entities: The valid entities for arguments in the
            predicate. If not specified, uses all defined entities.
        """
        if existing_entities is None:
            existing_entities = {}

        func_name, func_args = parse_func(pred_str)
        pred = self.predicates[func_name].clone()
        arg_values = []
        for func_arg in func_args:
            if func_arg in self.all_entities:
                v = self.all_entities[func_arg]
            elif func_arg in existing_entities:
                v = existing_entities[func_arg]
            else:
                raise ValueError(
                    f"Could not find entity {func_arg} in predicate `{pred_str}` (args={func_args} name={func_name})"
                )
            arg_values.append(v)
        try:
            pred.set_param_values(arg_values)
        except Exception as e:
            raise ValueError(
                f"Problem setting predicate values {pred} with {arg_values}"
            ) from e
        return pred

    def _parse_expr(
        self, load_d, existing_entities: Dict[str, FindingDoryPddlEntity]
    ) -> Union[FindingDoryLogicalExpr, FindingDoryPredicate]:
        """
        Similar to `self.parse_predicate` for logical expressions. If `load_d`
        is a string, it will be parsed as a predicate.
        """
        if load_d is None:
            return FindingDoryLogicalExpr(FindingDoryLogicalExprType.AND, [], [], None)

        if isinstance(load_d, str):
            # This can be assumed to just be a predicate
            return self.parse_predicate(load_d, existing_entities)
        if isinstance(load_d, list):
            raise TypeError(
                f"Could not parse logical expr {load_d}. You likely need to nest the predicate list in a logical expression"
            )

        try:
            expr_type = FindingDoryLogicalExprType[load_d["expr_type"]]
        except Exception as e:
            raise ValueError(f"Could not load expr_type from {load_d}") from e

        input_strs = load_d.get("inputs", [])
        inputs = []
        for x in input_strs:
            entity_properties = self._parse_entity_properties(x)

            inputs.append(
                FindingDoryPddlEntity(
                    x["name"],
                    self.expr_types[x["expr_type"]],
                    entity_properties,
                )
            )

        sub_exprs = [
            self._parse_expr(
                sub_expr, {**existing_entities, **{x.name: x for x in inputs}}
            )
            for sub_expr in load_d["sub_exprs"]
        ]
        quantifier = load_d.get("quantifier", None)
        if quantifier is not None:
            quantifier = FindingDoryLogicalQuantifierType[quantifier]
        return FindingDoryLogicalExpr(expr_type, sub_exprs, inputs, quantifier)

    def _parse_entity_properties(self, input_str):
        """
        Parses the properties of an entity.
        """
        properties = input_str.get("properties", None)
        if properties is None:
            return None
        
        entity_properties = []
        for property in properties:
            cls = getattr(
                findingdory_pddl,
                property["_target_"]
            )
            entity_properties.append(
                cls(
                    property["name"],
                    property["value"]
                )
            )
        
        return frozenset(entity_properties)

    def bind_to_instance(
        self,
        sim,
        env,
        episode: FindingDoryEpisode,
    ) -> None:
        """
        Attach the domain to the simulator. This does not bind any entity
        values, but creates `self._sim_info` which is needed to check simulator
        backed values (like truth values of predicates).
        """
        self._added_entities = {}
        self._added_expr_types = {}

        id_to_name = {}
        for k, i in sim.handle_to_object_id.items():
            id_to_name[i] = k
        
        receptacles_viewpoints_dict = {}
        receptacles_viewpoints_list = episode.candidate_goal_receps + \
            episode.candidate_goal_receps_noninteracted + \
            episode.candidate_start_receps + \
            episode.candidate_start_receps_noninteracted
        for receptacle in receptacles_viewpoints_list:
            receptacles_viewpoints_dict[receptacle.object_name] = receptacle

        self._sim_info = FindingDoryPddlSimInfo(
            sim=sim,
            env=env,
            episode=episode,
            expr_types=self.expr_types,
            obj_ids=sim.handle_to_object_id,
            target_ids={
                f"TARGET_{id_to_name[idx]}": idx
                for idx in sim.get_targets()[0]
            },
            art_handles={k.handle: i for i, k in enumerate(sim.art_objs)},
            marker_handles=sim.get_all_markers(),
            robot_ids={
                f"robot_{agent_id}": agent_id
                for agent_id in range(sim.num_articulated_agents)
            },
            all_entities=self.all_entities,
            predicates=self.predicates,
            receptacles=sim.receptacles,
            receptacles_viewpoints=receptacles_viewpoints_dict,
            sequential_goals=episode.instructions[env._chosen_instr_idx].sequential_goals,
        )
        # Ensure that all objects are accounted for.
        for entity in self.all_entities.values():
            self._sim_info.search_for_entity(entity)

    def get_true_predicates(self) -> List[FindingDoryPredicate]:
        """
        Get all the predicates that are true in the current simulator state.
        """

        all_entities = self.all_entities.values()
        true_preds: List[FindingDoryPredicate] = []
        for pred in self.predicates.values():
            for entity_input in itertools.permutations(
                all_entities, pred.n_args
            ):
                if not pred.are_args_compatible(list(entity_input)):
                    continue

                use_pred = pred.clone()
                use_pred.set_param_values(entity_input)

                if use_pred.is_true(self.sim_info):
                    true_preds.append(use_pred)
        return true_preds

    def get_possible_predicates(self) -> List[FindingDoryPredicate]:
        """
        Get all predicates that COULD be true. This is independent of the
        simulator state and is the set of compatible predicate and entity
        arguments. The same ordering of predicates is returned every time.
        """

        all_entities = self.all_entities.values()
        poss_preds: List[FindingDoryPredicate] = []
        for pred in self.predicates.values():
            for entity_input in itertools.combinations(
                all_entities, pred.n_args
            ):
                if not pred.are_args_compatible(entity_input):
                    continue

                use_pred = pred.clone()
                use_pred.set_param_values(entity_input)
                poss_preds.append(use_pred)
        return sorted(poss_preds, key=lambda pred: pred.compact_str)

    def find_entities(self, entity_type: ExprType) -> Iterable[FindingDoryPddlEntity]:
        """
        Returns all the entities that match the condition.
        """
        for entity in self.all_entities.values():
            if entity.expr_type.is_subtype_of(entity_type):
                yield entity

    def expand_quantifiers(
        self, expr: FindingDoryLogicalExpr, override_entities=None
    ) -> Tuple[FindingDoryLogicalExpr, List[Dict[FindingDoryPddlEntity, FindingDoryPddlEntity]]]:
        """
        Expand out a logical expression that could involve a quantifier into
        only logical expressions that don't involve any quantifier. Doesn't
        require the simulation to be grounded and expands using the current
        defined types.

        :returns: The expanded expression and the list of substitutions in the
            case of an EXISTS quantifier.
        """
        expr.sub_exprs = [
            (
                self.expand_quantifiers(subexpr)[0]
                if isinstance(subexpr, FindingDoryLogicalExpr)
                else subexpr
            )
            for subexpr in expr.sub_exprs
        ]

        if expr.quantifier == FindingDoryLogicalQuantifierType.FORALL:
            combine_type = FindingDoryLogicalExprType.AND
        elif expr.quantifier == FindingDoryLogicalQuantifierType.EXISTS:
            combine_type = FindingDoryLogicalExprType.OR
        elif expr.quantifier == FindingDoryLogicalQuantifierType.FORALL_SEQUENTIAL_UNORDERED:
            combine_type = FindingDoryLogicalExprType.AND_SEQUENTIAL
        elif expr.quantifier == FindingDoryLogicalQuantifierType.FORALL_SEQUENTIAL_ORDERED:
            combine_type = FindingDoryLogicalExprType.AND_SEQUENTIAL_ORDERED
        elif expr.quantifier is None:
            return expr, []
        else:
            raise ValueError(f"Unrecognized {expr.quantifier}")
        
        # Override the PDDL entities that will be used for verification. If None, then just use all the current exisitng entities
        if override_entities is None:
            entities_to_consider = self.all_entities
        else:
            entities_to_consider = override_entities

        t_start = time.time()
        assigns: List[List[FindingDoryPddlEntity]] = [[]]

        for expand_entity in expr.inputs:
            entity_assigns = []
            for e in entities_to_consider.values():
                if not e.expr_type.is_subtype_of(expand_entity.expr_type):
                    continue
                if expand_entity.properties is not None:
                    if not expand_entity.properties.issubset(e.properties):
                        continue
                for cur_assign in assigns:
                    if e in cur_assign:
                        continue
                    entity_assigns.append([*cur_assign, e])
            assigns = entity_assigns
        if self._sim_info is not None:
            self.sim_info.sim.add_perf_timing("assigns_search", t_start)
        
        # Create a dict which will be used to track metrics for each assign which will be used to analyse failure scenarios for high-level/low-level policy. This dict is updated in the is_robot_at_position() PDDL predicate
        self._sim_info.env._hl_pddl_per_assign_metrics = {}
        self._sim_info.env._ll_pddl_per_assign_metrics = {}
        for ele in assigns:
            self._sim_info.env._hl_pddl_per_assign_metrics[ele[0].name] = {
                "success": None,
                "dist_to_goal": -1.,
                "angle_to_goal": -1.,
                "semantic_cov": -1.,
                "d2g_exceed": None,
                "a2g_exceed": None,
                "tgt_not_in_view": None,
            }
            self._sim_info.env._ll_pddl_per_assign_metrics[ele[0].name] = {
                "success": None,
                "dist_to_goal": -1.,
                "angle_to_goal": -1.,
                "semantic_cov": -1.,
                "d2g_exceed": None,
                "a2g_exceed": None,
                "tgt_not_in_view": None,
            }

        t_start = time.time()
        
        # For ordered sequential goals, create a mapping between the target obj/recep IDs and the dynamically generated logical sub_expr IDs based on the matched entities (assigns)
        if expr.quantifier == FindingDoryLogicalQuantifierType.FORALL_SEQUENTIAL_ORDERED: 
            target_name_to_target_obj_id = {}
            
            if assigns[0][0].expr_type.parent.name == FindingDorySimulatorObjectType.MOVABLE_ENTITY.value:
                for k,v in self.sim_info.sim._candidate_obj_idx_to_rigid_obj_handle.items():
                    target_name_to_target_obj_id[v] = k
            
            elif assigns[0][0].expr_type.parent.name == FindingDorySimulatorObjectType.STATIC_RECEPTACLE_ENTITY.value:
                recep_name_to_obj_id = defaultdict(list)
                
                if self.sim_info.episode.instructions[self.sim_info.env._chosen_instr_idx].goal_expr['inputs'][0]['properties'][1]['name'] == "start_receptacle":    
                    recep_name_to_recep_id = {}
                    obj_name_to_obj_id = {}

                    for idx, recep in enumerate(self.sim_info.sim.ep_info.candidate_start_receps):
                        recep_name_to_recep_id[recep.object_name] = idx
                    for k, v in self.sim_info.sim._candidate_obj_idx_to_rigid_obj_handle.items():
                        obj_name_to_obj_id[v] = k
                    for obj_name, recep_name in self.sim_info.episode.name_to_receptacle.items():
                        if obj_name in obj_name_to_obj_id and recep_name in recep_name_to_recep_id:
                            recep_name_to_obj_id[recep_name].append(obj_name_to_obj_id[obj_name])

                elif self.sim_info.episode.instructions[self.sim_info.env._chosen_instr_idx].goal_expr['inputs'][0]['properties'][1]['name'] == "goal_receptacle":
                    for idx,recep in enumerate(self.sim_info.sim.ep_info.goal_receptacles):
                        recep_name_to_obj_id[recep[0]].append(idx)
                
                target_name_to_target_obj_id = recep_name_to_obj_id
                    
            target_idx_to_sub_expr_idx = {}
            for idx,assign in enumerate(assigns):
                if isinstance(target_name_to_target_obj_id[assign[0].name], list):
                    for id in target_name_to_target_obj_id[assign[0].name]:
                        target_idx_to_sub_expr_idx[id] = idx
                else:
                    target_idx_to_sub_expr_idx[target_name_to_target_obj_id[assign[0].name]] = idx
            
            sub_expr_seq = [-1] * len(self.sim_info.sequential_goals['sequence'])
            for idx,ele in enumerate(self.sim_info.sequential_goals['sequence']):
                sub_expr_seq[idx] = target_idx_to_sub_expr_idx[ele]

            self.sim_info.sequential_goals.update({'sub_expr_sequence': sub_expr_seq})

        assigns = [dict(zip(expr.inputs, assign)) for assign in assigns]
        expanded_exprs = []
        for assign in assigns:
            expanded_exprs.append(expr.sub_in_clone(assign))
        if self._sim_info is not None:
            self.sim_info.sim.add_perf_timing("expand_exprs_set", t_start)

        inputs: List[FindingDoryPddlEntity] = []
        multi_goal_combine_types = [FindingDoryLogicalExprType.AND_SEQUENTIAL_ORDERED, FindingDoryLogicalExprType.AND_SEQUENTIAL]
        return (
            FindingDoryLogicalExpr(
                combine_type,
                expanded_exprs,
                inputs,
                None,
                self.sim_info.sequential_goals if combine_type == FindingDoryLogicalExprType.AND_SEQUENTIAL_ORDERED else None,
                self.sim_info.env.connected_recep_names if combine_type in multi_goal_combine_types else None,
                self.sim_info.env.pddl.all_entities if combine_type in multi_goal_combine_types else None
            ),
            assigns,
        )

    def _parse_actions(self, domain_def) -> None:
        """
        Fetches the PDDL actions into `self.actions`
        """

        for action_d in domain_def["actions"]:
            parameters = [
                FindingDoryPddlEntity(p["name"], self.expr_types[p["expr_type"]])
                for p in action_d["parameters"]
            ]
            name_to_param = {p.name: p for p in parameters}

            pre_cond = self.parse_only_logical_expr(
                action_d["precondition"], name_to_param
            )

            # Include the precondition quantifier inputs.
            postcond_entities = {
                **{x.name: x for x in pre_cond.inputs},
                **name_to_param,
            }
            post_cond = [
                self.parse_predicate(p, postcond_entities)
                for p in action_d["postcondition"]
            ]

            action = PddlAction(
                action_d["name"], parameters, pre_cond, post_cond
            )
            self._orig_actions[action.name] = action
        self._actions = dict(self._orig_actions)


def get_pddl(
        task_config,
        agent_name,
        obj_cats,
        rec_cats,
        room_cats,
    ) -> PddlDomainFindingDory:
    config_path = osp.dirname(inspect.getfile(findingdory_config))
    domain_file_path = osp.join(
        config_path,
        task_config.pddl_domain_def_path,
    )
    pddl = PddlDomainFindingDory(
        domain_file_path,
        task_config,
    )

    pddl.add_permanent_expr_types(
        obj_cats,
        rec_cats,
        room_cats,
        agent_name,
    )

    return pddl