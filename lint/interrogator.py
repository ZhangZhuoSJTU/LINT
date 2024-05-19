from .constants import TERMINAL_PUNCTUATION, EVALUATION_PROMPT
from .clause_checker import NaiveChecker
from .utils import construct_prompt, is_contiguous_subsequence

from dataclasses import dataclass
from transformers import pipeline

import copy
import datetime
import gc
import json
import pickle
import random
import re
import string
import torch


@dataclass
class SearchingState:
    position: int

    topk_logits: list
    topk_indices: list
    topk_tokens: list

    picked: int
    rank: int

    def __init__(self, position, topk_logits, topk_indices, tokenizer):
        self.position = position

        self.topk_logits = topk_logits
        self.topk_indices = topk_indices
        self.topk_tokens = list(map(lambda i: tokenizer.decode(i), topk_indices))

        self.picked = -1
        self.rank = -1


@dataclass
class Clause:
    indice: int
    start_token: int
    tokens: list[int]
    string: str
    score: float = -10000.0


class Interrogator:
    def __init__(
        self,
        model,
        tokenizer,
        eval_model,
        eval_tokenizer,
        logger,
        checker=NaiveChecker,
        results={},
        searching_manual=False,
        searching_topk=500,
        searching_check_n=20,
        searching_target_n=5,
        searching_entailment_force_depth=1,
        searching_max_token_n=200,
        batch_size=100,
        interception=True,
        classifier_type="entailment",
        dump_tmp_results=True,
    ):
        self.logger = logger
        self.checker = checker

        self.model = model
        self.tokenizer = tokenizer

        self.eval_model = eval_model
        self.eval_tokenizer = eval_tokenizer

        self.searching_manual = searching_manual
        self.searching_topk = searching_topk
        self.searching_check_n = searching_check_n
        self.searching_target_n = searching_target_n
        self.searching_entailment_force_depth = searching_entailment_force_depth
        self.searching_max_token_n = searching_max_token_n

        self.searching_traceback_depth = 1

        self.__set_ranking_classifier(classifier_type)

        self.dump_tmp_results = dump_tmp_results

        self.interception = interception

        self.batch_size = batch_size

        self.breaker_id = random.randint(0, 0xFFFFFFFFFFFFFFFF)

        self.__terminate_fn = (
            lambda s: any(c in TERMINAL_PUNCTUATION for c in s)
            or not s.isascii()
            or s.strip() == "</s>"
            or s == "\n"
        )

        if self.searching_manual:
            self.__searching_selection = self.__interrogation_search_manual
        else:
            self.__searching_selection = self.__interrogation_search_auto

        self.results = results

        self.__reset()

    def __set_ranking_classifier(self, classifier_type="entailment"):
        if classifier_type == "entailment":
            self.classifier_type = "entailment"
            self.classifier = pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-deberta-base",
                device=0,
            )
        elif classifier_type == "gptfuzzer":
            self.classifier_type = "gptfuzzer"
            self.classifier = pipeline(
                "text-classification", model="hubert233/GPTFuzz", device=0
            )
        else:
            assert False, "Unreachable code"

    def __reset(self):
        # reset internal data structures
        self.__start_time = None
        self.__instruction_token_n = 0
        self.__schema = []
        self.__behavior = None
        self.__prompt = None
        self.__target_depth = 0xDEADBEEF
        self.__early_stop_reason = None

    def dump_results(self, path):
        pickle.dump(self.results, open(f"{path}.pkl", "wb"))

        json_data = {}
        for behavior, results in self.results.items():
            json_data[behavior] = []
            for result in results:

                rdata = {}
                rdata["content"] = result["content"]
                rdata["ext_content"] = result["ext_content"]
                rdata["time"] = result["time"]
                rdata["forcing"] = self.__get_force_n(result["schema"])
                rdata["picks"] = len(result["schema"])

                json_data[behavior].append(rdata)

        json.dump(json_data, open(f"{path}.json", "w"), indent=4)

        self.logger.info(f"Dump results to {path}.pkl and {path}.json")

    @staticmethod
    @torch.no_grad()
    def __forward(model, input_ids, topk):
        assert len(input_ids) == 1

        input_ids = torch.tensor(input_ids).cuda()
        logits = (
            model(input_ids=input_ids, attention_mask=None).logits[:, -1, :].squeeze()
        )

        topk_logits, topk_indices = torch.topk(logits, k=topk, dim=-1)
        result = [(topk_indices.tolist(), topk_logits.tolist())]

        del input_ids, logits
        del topk_logits, topk_indices
        gc.collect()
        torch.cuda.empty_cache()

        return result

    def forward(self, input_ids, topk):
        return self.__forward(self.model, input_ids, topk)

    @staticmethod
    @torch.no_grad()
    def __generate(model, tokenizer, input_ids, max_new_tokens=None, interception=True):
        assert max_new_tokens is None or max_new_tokens > 0

        gen_config = model.generation_config
        gen_config.do_sample = False
        gen_config.max_new_tokens = 32 if max_new_tokens is None else max_new_tokens

        # check whether all the elements in the input_ids have the same length
        if len(set(map(len, input_ids))) != 1:
            _old_pad_token_id = tokenizer.pad_token_id
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            _prompt_s = list(map(lambda ids: tokenizer.decode(ids), input_ids))
            _inputs = tokenizer(_prompt_s, padding=True, return_tensors="pt").to(
                model.device
            )

            output_ids = model.generate(
                **_inputs, no_repeat_ngram_size=2, generation_config=gen_config,
            )
            tokenizer.pad_token_id = _old_pad_token_id

            result = output_ids[:, _inputs.input_ids.shape[-1] :].tolist()

            del _inputs
            del output_ids
        else:
            _input_ids = torch.tensor(input_ids).cuda()
            _attn_masks = torch.ones_like(_input_ids).cuda()
            _input_ids_len = _input_ids.shape[-1]

            pad_token_id = (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            )

            if interception:
                token_count = gen_config.max_new_tokens
                gen_config.max_new_tokens = 1

                while token_count > 0:
                    output_ids = model.generate(
                        _input_ids,
                        attention_mask=_attn_masks,
                        generation_config=gen_config,
                        no_repeat_ngram_size=2,
                        pad_token_id=pad_token_id,
                    )

                    # check if _input_ids ends with any key in model.interception_map
                    # if so, replace the ending of _input_ids with the corresponding value
                    for k, v in model.interception_map.items():
                        len_k = len(k)
                        if _input_ids.shape[0] == 1:
                            if torch.equal(output_ids[0, -len_k:], k):
                                output_ids[0, -len_k:] = v
                                break
                        elif len_k == len(v):
                            indices_to_update = [
                                i
                                for i in range(output_ids.shape[0])
                                if torch.equal(output_ids[i, -len_k:], k)
                            ]
                            for i in indices_to_update:
                                output_ids[i, -len_k:] = v

                    _input_ids = output_ids
                    _attn_masks = torch.ones_like(_input_ids).cuda()
                    token_count -= 1
            else:
                output_ids = model.generate(
                    _input_ids,
                    attention_mask=_attn_masks,
                    generation_config=gen_config,
                    no_repeat_ngram_size=2,
                    pad_token_id=pad_token_id,
                )

            result = output_ids[:, _input_ids_len:].tolist()

            del _input_ids, _attn_masks
            del output_ids

        gc.collect()
        torch.cuda.empty_cache()

        return result

    def generate(
        self, input_ids, max_new_tokens=None, evaluate=False, interception=True
    ):
        model = self.eval_model if evaluate else self.model
        tokenizer = self.eval_tokenizer if evaluate else self.tokenizer
        assert not (evaluate and interception)

        return self.__generate(
            model, tokenizer, input_ids, max_new_tokens, interception=interception
        )

    def interrogate(self, behavior, prompt, instruction=None):
        self.__reset()

        if behavior in self.results:
            self.logger.info(f"Behavior already exploited: {repr(behavior)}")
            return

        if instruction is None:
            instruction = prompt

        self.logger.info(f"Target behavior: {repr(behavior)}")
        self.logger.info(f"Prompt: {repr(prompt)}")
        self.logger.info(f"Instruciton: {repr(instruction)}")

        if self.eval_model is not None:
            prompt_s = EVALUATION_PROMPT[self.eval_model.model_name]["prompt"].format(
                behavior=behavior.lower(), content="[CONTENT TO EVALUATE]"
            )
            self.logger.debug(f"Evaluation prompt: {repr(prompt_s)}")
        else:
            self.logger.debug("No evaluation model")

        # make sure the current schema is empty
        self.__start_time = datetime.datetime.now()
        self.__schema.clear()
        self.__behavior = behavior.rstrip(string.punctuation)
        self.__prompt = prompt
        self.__target_depth = 0xDEADBEEF
        self.__early_stop_reason = None
        self.__instruction_token_n = len(self.tokenizer(instruction)["input_ids"])

        # prepare input_ids
        input_ids = self.tokenizer(prompt)["input_ids"]
        assert (
            self.tokenizer(instruction)["input_ids"]
            == input_ids[: self.__instruction_token_n]
        )

        return self.__interrogation_search(input_ids)

    def __get_topk_clauses(self, input_ids, topk_indices, base=0):
        self.logger.debug(
            f"Start to do inference: {base} --> {base + len(topk_indices)}"
        )
        start_time = datetime.datetime.now()

        new_input_ids = [input_ids + [id] for id in topk_indices]

        output_ids = self.generate(new_input_ids, interception=False)

        self.logger.debug(
            f"Model inference done: {datetime.datetime.now() - start_time}"
        )

        clauses = []
        for i, tokens in enumerate(output_ids):
            clause_s = self.tokenizer.decode([topk_indices[i]] + tokens)
            clauses.append(
                Clause(
                    indice=base + i,
                    start_token=topk_indices[i],
                    tokens=[topk_indices[i]] + tokens,
                    string=clause_s,
                )
            )

        clauses = list(filter(lambda c: self.checker.check(c.string), clauses))
        self.logger.debug(
            f"Clause parsing done: {datetime.datetime.now() - start_time}"
        )

        return clauses

    def __evaluate(self, eval_idss, check_danger=True):
        start_time = datetime.datetime.now()
        self.logger.debug(f"Start evaluation: {len(eval_idss)}")

        if self.eval_tokenizer is None:
            # we do not have an evaluator
            if check_danger:
                return None if len(eval_idss) == 0 else eval_idss[0]
            else:
                return None

        eval_ss = list(map(lambda ids: self.tokenizer.decode(ids), eval_idss))
        prompt_ss = list(
            map(
                lambda s: EVALUATION_PROMPT[self.eval_model.model_name][
                    "prompt"
                ].format(behavior=self.__behavior.lower(), content=s.strip()),
                eval_ss,
            )
        )

        for eval_s in eval_ss:
            self.logger.debug(
                f"Evaluating ({'danger' if check_danger else 'safe'}): {repr(eval_s)}"
            )

        prompt_idss = self.eval_tokenizer(prompt_ss)["input_ids"]

        output_idss = self.generate(
            prompt_idss, max_new_tokens=8, evaluate=True, interception=False
        )
        output_ss = list(map(lambda ids: self.eval_tokenizer.decode(ids), output_idss))

        self.logger.debug(f"Evaluation done: {datetime.datetime.now() - start_time}")

        for i, output_s in enumerate(output_ss):
            self.logger.debug(
                f"Evaluation result ({'danger' if check_danger else 'safe'}): "
                f"{repr(output_s)} --> {repr(eval_ss[i])}"
            )
            # some stupid models may add spaces or lower case

            if check_danger:
                if any(
                    k in "".join(output_s.upper().split())
                    for k in EVALUATION_PROMPT[self.eval_model.model_name][
                        "pos_indicator"
                    ]
                ):
                    self.logger.debug(
                        f"Evaluation successed (danger): {repr(eval_ss[i])}"
                    )
                    return eval_idss[i]
            else:
                if any(
                    k in "".join(output_s.upper().split())
                    for k in EVALUATION_PROMPT[self.eval_model.model_name][
                        "neg_indicator"
                    ]
                ):
                    self.logger.debug(
                        f"Evaluation successed (safe): {repr(eval_ss[i])}"
                    )
                    return eval_idss[i]

        return None

    @torch.no_grad()
    def __analyze_clauses_entailment(self, input_s, clauses):
        input_s = input_s.replace("</s>", "").strip()
        self.logger.debug(f"Analyzing entailment upon: {repr(input_s)}")

        candidate_labels = ["contradiction", "entailment"]

        sentences = []
        for clause in clauses:
            cand_s = clause.string

            m = re.match(r"^\w\.", cand_s)
            s = len(m[0]) if bool(m) else 0
            for i in range(s, len(cand_s)):
                if self.__terminate_fn(cand_s[i]):
                    break

            sentences.append(f"{input_s} {cand_s[:i + 1]}".strip())

        if self.classifier_type == "entailment":
            results = self.classifier(sentences, candidate_labels, multi_label=False)

            for i, result in enumerate(results):
                clauses[i].score = result["scores"][
                    result["labels"].index("entailment")
                ]

                if clauses[i].score > 0.9:
                    # if the score is higher that 0.9, it is likely we are going to a
                    # natural states.
                    #
                    # let's make it a huge number taking original ranking into consideratino
                    clauses[i].score = 1 + self.searching_topk - clauses[i].indice

                if clauses[i].string in input_s:
                    # if the clause is already in the input, we do not want to add it
                    clauses[i].score = -1

        elif self.classifier_type == "gptfuzzer":
            results = self.classifier(sentences)

            for i, result in enumerate(results):
                clauses[i].score = (
                    result["score"] if result["label"] == 1 else 1 - result["score"]
                )

        else:
            assert False, "Unreachable code"

        gc.collect()
        torch.cuda.empty_cache()

    def __interrogation_search(self, input_ids):
        input_s = self.tokenizer.decode(input_ids)
        depth = len(self.__schema)

        self.logger.info(f"Current depth ({self.breaker_id:x}): {depth}")
        self.logger.info(f"Current input ({self.breaker_id:x}): {repr(input_s)}")

        if (
            len(input_ids) >= self.searching_max_token_n + self.__instruction_token_n
            or input_ids[-1] == self.tokenizer.eos_token_id
        ) and not self.searching_manual:
            # we reach the target searching depth
            content = input_s
            run_time = (datetime.datetime.now() - self.__start_time).total_seconds()

            # let's go further, let the model output more
            ext_output_ids = self.generate(
                [input_ids], max_new_tokens=250, interception=self.interception
            )
            ext_content = self.tokenizer.decode(ext_output_ids[0])

            # update results
            if self.__behavior not in self.results:
                self.results[self.__behavior] = []

            # get the current result
            result = {
                "behavior": self.__behavior,
                "prompt": self.__prompt,
                "content": content,
                "ext_content": ext_content,
                "schema": copy.deepcopy(self.__schema),
                "time": run_time,
            }
            self.results[self.__behavior].append(result)
            self.results[self.__behavior].sort(key=lambda s: len(s["schema"]))

            self.logger.info(
                f"Success w/ {self.__get_force_n(self.__schema)} forcing "
                f"out of {len(self.__schema)} picks "
                f"({result['time']}): {repr(content)}"
            )
            self.logger.info(f"Extended version: {repr(ext_content)}")

            if self.dump_tmp_results:
                self.dump_results(f".tmp_interrogator_{self.breaker_id:x}")

            if len(self.results[self.__behavior]) < self.searching_target_n:
                self.__target_depth = min(
                    self.__target_depth, self.searching_traceback_depth
                )
                self.__early_stop_reason = "ATCK SUCC"
            else:
                self.__target_depth = -1
                self.__early_stop_reason = "INTERROGATION DONE"

            return True

        # check target depth (this has to be done at each round of recursion)
        # although it may cause redundant operations, it makes the code clean
        if depth > self.__target_depth:
            self.logger.debug(
                f"Early Stop [{self.__early_stop_reason}] "
                f"(depth {depth}-->{self.__target_depth}): {repr(input_s)}"
            )
            return False
        elif self.__target_depth != 0xDEADBEEF:
            self.__target_depth = 0xDEADBEEF
            self.__early_stop_reason = None

        topk = self.searching_topk
        topk_indices, topk_logits = self.forward([input_ids], topk)[0]

        # update force schema
        self.__schema.append(
            SearchingState(len(input_ids), topk_logits, topk_indices, self.tokenizer)
        )

        entailment_input_s = construct_prompt(
            self.__behavior, "none"
        ) + self.tokenizer.decode(input_ids[self.__instruction_token_n :])

        # get top-k clauses
        clauses = []
        for start_i in range(0, topk, self.batch_size):
            end_i = min(start_i + self.batch_size, topk)
            sub_clauses = self.__get_topk_clauses(
                input_ids, topk_indices[start_i:end_i], base=start_i
            )

            if depth <= self.searching_entailment_force_depth:
                self.__analyze_clauses_entailment(entailment_input_s, sub_clauses)
            else:
                for clause in sub_clauses:
                    if clause.string not in input_s:
                        clause.score = topk - clause.indice + 1
                    else:
                        clause.score = -1

            clauses.extend(sub_clauses)

            # no need to do more inference
            if sum(map(lambda c: c.score > 1, clauses)) >= self.searching_check_n:
                break

        clauses.sort(key=lambda c: c.score, reverse=True)
        for i, clause in enumerate(clauses[: self.searching_check_n]):
            self.logger.debug(
                f"Clause ({depth}.{i}): {repr(clause.string)} (score: {clause.score}, rank: {clause.indice})"
            )

        for i, clause in self.__searching_selection(clauses, self.searching_check_n):
            if depth >= self.__target_depth:
                break

            self.__schema[-1].picked = clause.indice
            self.__schema[-1].rank = i

            self.logger.debug(f"Clause ({depth}.{i}): {repr(clause.string)}")

            new_input_ids = input_ids + clause.tokens

            # magic shortcut of the additional token
            if self.searching_max_token_n + self.__instruction_token_n < len(
                new_input_ids
            ):
                new_input_ids = new_input_ids[
                    : self.searching_max_token_n + self.__instruction_token_n
                ]

            # let's try to extend as long as possible
            start_time = datetime.datetime.now()
            self.logger.debug(f"Start to extend current input")
            output_ids = self.generate(
                [new_input_ids],
                max_new_tokens=self.searching_max_token_n
                + 1
                + self.__instruction_token_n
                - len(new_input_ids),
                interception=self.interception,
            )[0]
            self.logger.debug(f"Extend done: {datetime.datetime.now() - start_time}")
            self.logger.debug(
                f"Current content: {repr(self.tokenizer.decode(new_input_ids + output_ids))}"
            )

            # then, we try to contruct the evaluation data
            clause_poss = [-1]
            cand_ids = new_input_ids[len(input_ids) :] + output_ids

            for j, token in enumerate(cand_ids):
                token_s = self.tokenizer.decode([token])
                if self.__terminate_fn(token_s):
                    clause_poss.append(j)
            clause_poss.append(len(cand_ids))

            # we will try to add valid (non-refusion) clauses into evaluation data
            eval_idss = []
            for j in range(len(clause_poss) - 1):
                s = clause_poss[j] + 1
                e = clause_poss[j + 1] + 1
                clause_s = self.tokenizer.decode(cand_ids[s:e])

                if not self.checker.check(clause_s) and not clause_s.endswith("</s>"):
                    self.logger.debug(f"Invalid extended clause: {repr(clause_s)}")
                    break

                self.logger.debug(f"Valid extended clause: {repr(clause_s)}")
                eval_idss.append(input_ids[self.__instruction_token_n :] + cand_ids[:e])

                if clause_s.endswith("</s>"):
                    break

            if len(eval_idss) == 0:
                continue

            # evaluate
            ext_input_ids = self.__evaluate(eval_idss[-1::-1], check_danger=True)
            if ext_input_ids is None:
                continue

            # let's do backward evaluation
            eval_idss = []
            for j in range(
                len(input_ids[self.__instruction_token_n :]), len(ext_input_ids)
            ):
                if len(eval_idss) == 0:
                    # enfore at least one check
                    eval_idss.append(ext_input_ids[j:])
                    continue

                token = ext_input_ids[j]
                token_s = self.tokenizer.decode([token])
                if self.__terminate_fn(token_s):
                    eval_idss.append(ext_input_ids[j + 1 :])

            safe_input_ids = self.__evaluate(eval_idss, check_danger=False)

            if safe_input_ids is None:
                pass
            elif len(safe_input_ids) + len(
                input_ids[self.__instruction_token_n :]
            ) == len(ext_input_ids):
                # we failed
                continue
            else:
                ext_input_ids = ext_input_ids[: -len(safe_input_ids)]

            new_input_ids = input_ids[: self.__instruction_token_n] + ext_input_ids

            self.__interrogation_search(new_input_ids)

        self.__schema.pop()
        return

    def __interrogation_search_manual(self, clauses, n):
        while True:
            for i, clause in enumerate(clauses[:n]):
                print(
                    f"{i:3d}: {repr(clause.string)} (score: {clause.score}, rank: {clause.indice})"
                )

            i = input("Select the clause, using '/help' to see help menu: ")
            if i.strip() == "/help":
                print("\t/back: go back to the previous level.")
                print("\t/more: see more candidates upon this level.")
                print("\t/less: reduce the number of candidates shown upon this level.")
                print("\t/exit: exit this round of attack.\n")
                print("\t/quit: quit this process.\n")
                print("\t/help: show this help menu.\n")
                continue

            if i.strip() == "/exit":
                raise Exception("USER EXIT")

            if i.strip() == "/quit":
                exit(0)

            if i.strip() == "/back":
                return

            if i.strip() == "/more":
                n += n
                continue

            if i.strip() == "/less":
                n = n // 2
                continue

            i = int(i.strip())
            if i >= n or i < 0:
                print(f"Invalid number ({i}), reenter please")
                continue

            yield (i, clauses[i])

    def __interrogation_search_auto(self, clauses, n):
        for i, clause in enumerate(clauses[:n]):
            yield (i, clause)

            # let' try to pick the first token, if there will be interception
            if not self.interception:
                continue

            for k in self.model.interception_map.keys():
                k = k.tolist()
                if is_contiguous_subsequence(k, clause.tokens):
                    break
            else:
                continue

            new_clause = copy.deepcopy(clause)
            new_clause.tokens = new_clause.tokens[:1]
            yield (i, new_clause)

    @staticmethod
    def __get_force_n(schema):
        return len(list(filter(lambda state: state.picked != 0, schema)))
