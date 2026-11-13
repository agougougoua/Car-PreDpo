# Evaluation

Export the base model plus the final Stage 2 adapter using the bundled
LLaMA-Factory CLI. Create `export.yaml` with your own absolute paths:

```yaml
model_name_or_path: /path/to/Qwen2.5-VL-3B-Instruct
adapter_name_or_path: /path/to/stage2
finetuning_type: lora
template: qwen2_vl
trust_remote_code: true
export_dir: /path/to/merged-model
export_size: 2
export_device: cpu
export_legacy_format: false
infer_dtype: bfloat16
```

```bash
llamafactory-cli export export.yaml
```

Use the merged model in the external benchmark harnesses:

| Benchmark | Harness used in the local project | Report |
|---|---|---|
| POPE | VLMEvalKit | Accuracy/F1 and the sampled setting |
| HallusionBench | VLMEvalKit | aAcc/fAcc/qAcc |
| ObjectHalBench | RLAIF-V | CHAIRs/CHAIRi and object recall |
| MMHal-Bench | RLAIF-V | Mean score and hallucination rate |

The local VLMEvalKit checkout records commit
`09163430240669be7171e419bdf2f390f938d059`. The copied RLAIF-V directory
has no Git metadata, so a source commit cannot be established from it.
Neither external harness is vendored in this compact release.

The local POPE and HallusionBench evaluation files contain 5,127 and 951
instances respectively; MMHal-Bench uses 96 questions. Check instance IDs
and exact dataset versions before comparing scores. Preserve generation
settings, benchmark subsets, judge model/version, judge prompts, and parser
failures. Different judge configurations must not be treated as identical.
API credentials must be supplied via the chosen harness's environment
configuration; no credentials or server proxy endpoints are included here.
Raw predictions and benchmark data must be obtained/generated separately.

Upstream source references: `https://github.com/open-compass/VLMEvalKit`
and `https://github.com/RLHF-V/RLAIF-V`. These are external dependencies,
not an assertion that their latest versions reproduce the local scores.
