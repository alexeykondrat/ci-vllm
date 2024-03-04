#!/usr/bin/env python3

"""
    For a given LLM sample solutions to the "human-eval" set of problems.
    Arguments: Beside typical vLLM arguments this script accepts
            --num-samples-per-task - an integer number of samples to extract
                                     per task
            --experiment-prefix -  a string to aggregate separate parallel
                                    experiments
            

    Output: two files   {experiment-prefix}_problems.jsonl
                        {experiment-prefix}_solutions.jsonl
            which are subsequently to be scored by the standard means of the
            "human-eval" dataset. E.g.:
            python evaluate_functional_correctness.py 
                {experiment-prefix}_solutions.jsonl 
                --problem_file={experiment-prefix}_problems.jsonl
"""

from human_eval.data import write_jsonl, read_problems
import json

import argparse
import datetime

from vllm import LLM, SamplingParams


def generate_prompt(input):
    INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{input}

### Response:"""
    return INSTRUCTION


def main(args: argparse.Namespace):

    print (f"### Initialising @ {datetime.datetime.now()}")
    
    print(args)

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        kv_cache_scales_path=args.kv_cache_scales_path if args.kv_cache_scales_path!='' else None,
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=0.0 if args.use_beam_search else args.temperature,
        top_p=1,
        use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)

    problems = read_problems()
    
    with open("./"+args.experiment_prefix+"_problems.jsonl", "w") as f:
        for task_id in problems:
            f.write(json.dumps(problems[task_id])+'\n')

    
    num_samples_per_task = args.num_samples_per_task
    
    print (f"### Starting generation @ {datetime.datetime.now()}")

    with open("./"+args.experiment_prefix+"_solutions.jsonl", "w") as f:
        for task_id in problems:
            one_completion = llm.generate(generate_prompt(problems[task_id]["prompt"]),sampling_params)[0]
            for i in range(args.n):
                myanswer = one_completion.outputs[i].text
                myanswer = myanswer.replace("\r", "")
                if '```python' in myanswer: 
                    def_line = myanswer.index('```python')
                    myanswer = myanswer[def_line:].strip()
                    myanswer = myanswer.replace('```python', '')
                    try:
                        next_line = myanswer.index('```')
                        myanswer = myanswer[:next_line].strip()
                    except:
                        pass

                if "__name__ == \"__main__\"" in myanswer:
                    next_line = myanswer.index('__name__ == "__main__"')
                    myanswer = myanswer[:next_line].strip()
                                                                
                if "# Example usage" in myanswer:
                    next_line = myanswer.index('# Example usage')
                    myanswer = myanswer[:next_line].strip()

                myanswer = '\n'.join([line for line in myanswer.splitlines() if not line.startswith("Here")])
                
                answer = dict(task_id=task_id, completion=myanswer)
                f.write(json.dumps(answer)+'\n')

print (f"### Done @ {datetime.datetime.now()}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--kv-cache-scales-path', type=str, default='')
    parser.add_argument('--num-samples-per-task', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--experiment-prefix',type=str, default='solution_samples')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'gptq', 'squeezellm', None],
                        default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--ppl-measurement', action='store_false')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters',
                        type=int,
                        default=3,
                        help='Number of iterations to run.')
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--enforce-eager',
                        action='store_true',
                        help='enforce eager mode and disable CUDA graph')
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=['auto', 'fp8_e5m2','fp8'],
        default='auto',
        help=
        'Data type for kv cache storage. If "auto", will use model data type.')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='profile the generation process of a single batch')
    parser.add_argument(
        '--profile-result-dir',
        type=str,
        default=None,
        help=('path to save the pytorch profiler output. Can be visualized '
              'with ui.perfetto.dev or Tensorboard.'))
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda"],
        help='device type for vLLM execution, supporting CUDA only currently.')
    args = parser.parse_args()
    main(args)

