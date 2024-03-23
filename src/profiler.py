import os
import torch
import torch.profiler as TorchProfiler

# torch.mps.profiler.start(
#   mode = 'interval,event',
#   wait_until_completed = True,
# )

# torch.mps.profiler.stop()

def profile(
  fn,
):

  # PROFILING
  if os.getenv('PERF', '0') == '1':

    print('[INFO] PERF: TRUE')

    # PROFILING MPS
    if os.getenv('MPS', '0') == '1':

        os.environ['PYTORCH_MPS_LOG_PROFILE_INFO'] = '31'

    print('\n')

    with TorchProfiler.profile(
        activities=[
          TorchProfiler.ProfilerActivity.CPU,
          # TorchProfiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory = True,
        with_stack = True,
        with_flops = True,

        # NOTE: Due to a bug stacks is empty without this line
        # SEE: https://github.com/pytorch/pytorch/issues/100253
        experimental_config = torch._C._profiler._ExperimentalConfig(verbose=True)
    ) as prof:

        with TorchProfiler.record_function("my_function"):

          fn()

    prof.export_chrome_trace('./trace.json')

    prof.export_stacks('./stacks.txt', 'self_cpu_time_total')

    print('\n')

    print(prof.key_averages().table())

    print('\n')

    # print(prof.key_averages(group_by_stack_n=5))
    print(prof.key_averages(group_by_input_shape=True))

    print('\n')

    print(prof.key_averages().table(top_level_events_only = True))

    # print(prof.key_averages()[0].cpu_children)
    # print(prof.key_averages()[1].cpu_parent)

  else:

    print('[INFO] PERF: FALSE')

    print('\n')

    fn()
