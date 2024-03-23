- [Apple Silicon](#apple-silicon)
- [CPU Profiling](#cpu-profiling)
  - [Trace](#trace)
  - [Flame Graph](#flame-graph)
- [MPS Profiling](#mps-profiling)

# Apple Silicon

SEE: https://github.com/dougallj/applegpu

# CPU Profiling

SEE: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-profiler-to-analyze-execution-time

```python
with profile(
  activities=[
    ProfilerActivity.CPU,
  ],
) as prof:

  # Do the thing

print(prof.key_averages())
```

## Trace

SEE: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-tracing-functionality

```python
with profile(
  activities=[
    ProfilerActivity.CPU,
  ],
) as prof:

  # Do the thing

prof.export_chrome_trace('./trace.json')
```

```chrome
chrome://tracing
```

## Flame Graph

SEE: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#visualizing-data-as-a-flame-graph

```python
with profile(
  activities=[
    ProfilerActivity.CPU,
  ],
  with_stack = True,

  # NOTE: Needed due to https://github.com/pytorch/pytorch/issues/100253
  experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
) as prof:

  # Do the thing

prof.export_stacks('./stacks.txt', 'self_cpu_time_total')
```

```bash
brew install flamegraph
```

```bash
flamegraph.pl \
  --title 'CPU time' \
  --countname 'us.' \
  ./stacks.txt > stacks.svg
```

# MPS Profiling

SEE: https://github.com/pytorch/pytorch/pull/100635

```bash
PYTORCH_MPS_LOG_PROFILE_INFO=31 ./test.py
```
