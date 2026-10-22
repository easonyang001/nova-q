"""Start vLLM when NVML is broken but CUDA works (active only with VLLM_NONVML=1).

After an NVIDIA driver update without a reboot, the userspace NVML (580) no longer matches the loaded kernel
module (565): nvidia-smi and pynvml fail with "Driver/library version mismatch", while CUDA itself still works
(torch sees both GPUs). vLLM detects its platform through NVML, so it stops with "Failed to infer device type"
(2026-10-11). vLLM already ships a non-NVML CUDA platform (used on Jetson); this hook makes its CUDA platform
check fall back to torch.cuda when NVML fails. It patches nothing on disk. Remove once the machine is rebooted.
"""
import os
import sys

if os.environ.get("VLLM_NONVML") == "1":
    import importlib.abc
    import importlib.util

    class _PatchPlatforms(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "vllm.platforms":
                return None
            sys.meta_path.remove(self)                       # one-shot, and avoid recursing into ourselves
            spec = importlib.util.find_spec(name)
            exec_module = spec.loader.exec_module

            def patched_exec(module):
                exec_module(module)
                original = module.cuda_platform_plugin

                def cuda_platform_plugin():
                    try:
                        found = original()
                    except Exception:
                        found = None
                    if found is None:
                        import torch
                        if torch.cuda.is_available():
                            return "vllm.platforms.cuda.CudaPlatform"   # resolves to NonNvmlCudaPlatform
                    return found

                module.cuda_platform_plugin = cuda_platform_plugin
                module.builtin_platform_plugins["cuda"] = cuda_platform_plugin

            spec.loader.exec_module = patched_exec
            return spec

    sys.meta_path.insert(0, _PatchPlatforms())
