"""Model-free regressions: python -m unittest discover -s tests -v."""
import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

from evaluation.guji_paper_eval import summarize_dataset_paper_metrics
from evaluation.paper_metrics import evaluate_explanation_sample
from evaluation.paper_eval_runner import merge_weighted_metric_summaries
from evaluation.paper_eval_summary import collect_metric_rows
from util.sample_adapter import adapt_sample_for_method


class PromptVariantsTests(unittest.TestCase):
    def test_translation_preserves_instruction_source_and_target(self):
        sample = {
            "instruction": "将以下文本翻译成现代文:",
            "input": "退還白帝，連引荊州，徐還圖之，此下計也。",
            "output": "退回白帝城。",
        }
        original = deepcopy(sample)
        adapted = adapt_sample_for_method(sample, "Attribution")
        self.assertEqual(len(adapted["prompts"]), 3)
        self.assertEqual(adapted["prompts"][0], adapted["prompt"])
        for prompt in adapted["prompts"]:
            self.assertIn(sample["instruction"], prompt)
            self.assertIn(sample["input"], prompt)
        self.assertEqual(adapted["task_aware_target_tokens_full"], sample["output"])
        self.assertEqual(sample, original)

    def test_instruction_only_qa(self):
        for input_text in ("", " "):
            with self.subTest(input=input_text):
                sample = {"instruction": "第一个将甘草称为国老的人是谁？", "input": input_text}
                adapted = adapt_sample_for_method(sample, "test")
                self.assertEqual(len(set(adapted["prompts"])), 3)
                self.assertTrue(all(sample["instruction"] in p for p in adapted["prompts"]))

    def test_context_is_never_dropped(self):
        sample = {"question": "何人？", "context": "陶弘景称甘草为国老。", "source_text": "其他材料"}
        adapted = adapt_sample_for_method(sample, "test")
        self.assertEqual(len(adapted["prompts"]), 3)
        for prompt in adapted["prompts"]:
            self.assertIn(sample["context"], prompt)
            self.assertIn(sample["question"], prompt)

    def test_question_only(self):
        adapted = adapt_sample_for_method({"question": "何人？"}, "test")
        self.assertEqual(len(set(adapted["prompts"])), 3)
        self.assertTrue(all("何人？" in p for p in adapted["prompts"]))

    def test_opaque_prompts_are_not_reconstructed(self):
        for sample in (
            {"prompt": "自定义任务与材料"},
            {"prompt": "自定义任务与材料", "question": "部分问题", "context": "部分材料"},
            {"prompt": "自定义任务与材料", "source_text": "部分材料"},
            {"input": "仅有原文"},
        ):
            with self.subTest(sample=sample):
                adapted = adapt_sample_for_method(sample, "test")
                self.assertEqual(adapted["prompts"], [adapted["prompt"]])

    def test_repository_tasks_preserve_fields(self):
        root = Path(__file__).resolve().parents[1]
        paths = list((root / "data/0_shot").glob("*.json"))
        paths += list((root / "data/chain_of_thought").glob("*.json"))
        self.assertTrue(paths)
        for path in paths:
            rows = json.loads(path.read_text())
            for sample in rows:
                with self.subTest(task=path.name):
                    adapted = adapt_sample_for_method(sample, "Attribution")
                    self.assertEqual(len(set(adapted["prompts"])), 3)
                    for prompt in adapted["prompts"]:
                        self.assertIn(sample["instruction"].strip(), prompt)
                        self.assertIn(sample.get("input", "").strip(), prompt)


class StabilityTests(unittest.TestCase):
    def evaluate(self, sample, evidence=None, **kwargs):
        calls = []
        evidence = evidence or {}

        def diagnose_fn(sample, **unused):
            calls.append(deepcopy(sample))
            return {"evidence_spans": [{"text": text, "score": 1.0}
                                       for text in evidence.get(sample["prompt"], ["甲", "乙"])]}

        result = evaluate_explanation_sample(
            sample, "unused-model", "Attribution",
            scorer_fn=lambda **unused: 0.5, diagnose_fn=diagnose_fn, **kwargs,
        )
        return result, calls

    def test_no_variants_is_null_and_json_serializable(self):
        result, calls = self.evaluate({"prompt": "甲乙", "output": "答案"})
        self.assertIsNone(result["stability"]["span_iou"])
        self.assertEqual(result["diagnostics"]["stability_comparisons"], 0)
        self.assertEqual(len(calls), 1)
        self.assertIn('"span_iou": null', json.dumps(result, allow_nan=False))

    def test_real_perfect_overlap_is_one(self):
        result, calls = self.evaluate({"prompt": "甲乙", "prompts": ["甲乙", "甲乙变体"]})
        self.assertEqual(result["stability"]["span_iou"], 1.0)
        self.assertEqual(result["diagnostics"]["stability_comparisons"], 1)
        self.assertEqual(len(calls), 2)

    def test_multiple_variants_average_comparisons(self):
        sample = {"prompt": "甲乙", "output": "固定答案", "prompts": ["甲乙", "变体一", "变体二"]}
        original = deepcopy(sample)
        result, calls = self.evaluate(sample, {"变体一": ["甲", "丙"], "变体二": ["丁"]})
        self.assertEqual(result["stability"]["span_iou"], 0.166667)
        self.assertEqual(result["diagnostics"]["stability_comparisons"], 2)
        self.assertTrue(all(s["task_aware_target_tokens_full"] == "固定答案" for s in calls))
        self.assertEqual(sample, original)

    def test_duplicates_blanks_and_omitted_base(self):
        result, calls = self.evaluate({"prompt": "甲乙", "prompts": ["", " 甲乙 ", "甲乙", "变体一", "变体一", "变体二"]})
        self.assertEqual([s["prompt"] for s in calls], ["甲乙", "变体一", "变体二"])
        self.assertEqual(result["diagnostics"]["stability_comparisons"], 2)
        result, calls = self.evaluate({"prompt": "甲乙", "prompts": ["变体一"]})
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["diagnostics"]["stability_comparisons"], 1)

    def test_limit_includes_base(self):
        for limit in (0, 1, 2, 3):
            result, calls = self.evaluate({"instruction": "翻译", "input": "甲乙"}, variant_limit=limit)
            self.assertEqual(len(calls), max(1, limit))
            self.assertEqual(result["diagnostics"]["stability_comparisons"], max(1, limit) - 1)
            if limit <= 1:
                self.assertIsNone(result["stability"]["span_iou"])

    def test_injected_functions_do_not_import_model_dependencies(self):
        self.evaluate({"prompt": "甲乙"})
        self.assertNotIn("diagnose.diagnose", sys.modules)
        self.assertNotIn("util.runtime", sys.modules)
        self.assertNotIn("torch", sys.modules)


class SummaryTests(unittest.TestCase):
    def test_chunk_merge_weights_observed_stability_and_sums_counts(self):
        def summary(values):
            return {"test": summarize_dataset_paper_metrics([
                {"stability": {"span_iou": value}, "faithfulness": {"naopc": 0.4}}
                for value in values
            ])}

        chunks = [(3, summary([None, None, None])), (3, summary([None, None, 0.0])), (2, summary([1.0, 1.0]))]
        merged = merge_weighted_metric_summaries(chunks)["test"]
        self.assertEqual(merged["stability"]["span_iou"], 0.666667)
        self.assertEqual(merged["faithfulness"]["naopc"], 0.4)
        self.assertEqual(merged["diagnostics"], {
            "stability_evaluated_samples": 3, "stability_unevaluated_samples": 5, "total_samples": 8,
        })
        self.assertIsNone(merge_weighted_metric_summaries(chunks[:1])["test"]["stability"]["span_iou"])

    def test_legacy_chunk_weights_and_missing_keys(self):
        merged = merge_weighted_metric_summaries([
            (1, {"test": {"stability": {"span_iou": 0.0}}}),
            (3, {"test": {"stability": {"span_iou": 1.0}}, "other": {"stability": {"span_iou": None}}}),
        ])
        self.assertEqual(merged["test"]["stability"]["span_iou"], 0.75)
        self.assertIsNone(merged["other"]["stability"]["span_iou"])

    def test_export_retains_null_and_metric_name(self):
        summary = summarize_dataset_paper_metrics([{"stability": {"span_iou": None}}])
        payload = {"model_name": "test", "datasets": {"test": {"summary": {"test": summary}}}}
        rows = collect_metric_rows([json.loads(json.dumps(payload, allow_nan=False))])
        self.assertIsNone(rows[0]["stability.span_iou"])

    def test_mixed_dataset_uses_per_metric_denominators(self):
        rows = [
            {"faithfulness": {"naopc": 0.2}, "stability": {"span_iou": None}},
            {"faithfulness": {"naopc": 0.4}, "stability": {"span_iou": 0.0}, "target_alignment": {"tta_at_3": 0.2}},
            {"faithfulness": {"naopc": 0.6}, "stability": {"span_iou": 1.0}, "target_alignment": {"tta_at_3": 0.8}},
            {"faithfulness": {"naopc": None}},
        ]
        summary = summarize_dataset_paper_metrics(rows)
        self.assertEqual(summary["stability"]["span_iou"], 0.5)
        self.assertEqual(summary["faithfulness"]["naopc"], 0.4)
        self.assertEqual(summary["target_alignment"]["tta_at_3"], 0.5)
        self.assertEqual(summary["diagnostics"], {
            "stability_evaluated_samples": 2, "stability_unevaluated_samples": 2, "total_samples": 4,
        })
        self.assertEqual(summary, summarize_dataset_paper_metrics(list(reversed(rows))))

    def test_all_missing_or_invalid_stability_is_null(self):
        rows = [{"stability": {"span_iou": value}} for value in (None, float("nan"), float("inf"), True)]
        summary = summarize_dataset_paper_metrics(rows)
        self.assertIsNone(summary["stability"]["span_iou"])
        self.assertEqual(summary["diagnostics"]["stability_evaluated_samples"], 0)
        json.dumps(summary, allow_nan=False)

    def test_empty_dataset(self):
        self.assertEqual(summarize_dataset_paper_metrics([]), {})


if __name__ == "__main__":
    unittest.main()
