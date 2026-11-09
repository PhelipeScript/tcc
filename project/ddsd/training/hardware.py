"""Hardware de treino: detecção CUDA → MPS → CPU, precisão e memória."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass

import torch

from ddsd.observability.logger import get_logger

log = get_logger("treino.hardware")

#: Capability mínima de CUDA com bfloat16 em hardware (Ampere).
_BF16_MIN_CAPABILITY = (8, 0)


@dataclass(frozen=True)
class DeviceInfo:
    backend: str  # "cuda" | "mps" | "cpu"
    name: str
    memory_gb: float
    bf16_supported: bool
    torch_version: str
    cuda_version: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _probe_mps_bf16() -> bool:
    """Testa de verdade o autocast bf16 no MPS (forward + backward), em vez de presumir suporte."""
    try:
        layer = torch.nn.Linear(16, 16).to("mps")
        with torch.autocast("mps", dtype=torch.bfloat16):
            out = layer(torch.randn(4, 16, device="mps"))
        out.float().sum().backward()
        torch.mps.synchronize()
        return out.dtype == torch.bfloat16 and layer.weight.grad is not None
    except Exception as exc:  # noqa: BLE001 - qualquer falha significa "sem bf16 aqui"
        log.warn("sondagem de bf16 no MPS falhou: %s", exc)
        return False


def describe_device() -> DeviceInfo:
    """Melhor acelerador disponível, com memória e suporte a bf16."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return DeviceInfo("cuda", props.name, round(props.total_memory / 2**30, 1),
                          torch.cuda.get_device_capability(0) >= _BF16_MIN_CAPABILITY,
                          torch.__version__, torch.version.cuda)
    memory = _sysctl("hw.memsize")
    memory_gb = round(int(memory) / 2**30, 1) if memory else 0.0
    if torch.backends.mps.is_available():
        chip = _sysctl("machdep.cpu.brand_string") or platform.processor()
        return DeviceInfo("mps", f"{chip} (GPU, memória unificada)", memory_gb, _probe_mps_bf16(),
                          torch.__version__, None)
    return DeviceInfo("cpu", platform.processor() or platform.machine(), memory_gb, False,
                      torch.__version__, None)


def require_accelerator(device: DeviceInfo, allow_cpu: bool) -> None:
    """Interrompe se não houver CUDA nem MPS, salvo em teste de fumaça (`allow_cpu`)."""
    if device.backend in {"cuda", "mps"}:
        return
    message = "nenhum acelerador (CUDA ou MPS) disponível; o treino completo em CPU levaria dias"
    if not allow_cpu:
        raise RuntimeError(message + " (use --allow-cpu só para teste de fumaça)")
    log.warn(message + "; seguindo em CPU por --allow-cpu")


def resolve_precision(requested: str, device: DeviceInfo) -> str:
    """Precisão efetiva: `auto` escolhe bf16 se sondado com sucesso, senão fp32."""
    if requested == "auto":
        return "bf16" if device.bf16_supported else "fp32"
    if requested == "bf16" and not device.bf16_supported:
        log.warn("%s não suporta bf16 (sondagem); usando fp32", device.name)
        return "fp32"
    if requested == "fp16" and device.backend == "cpu":
        log.warn("fp16 em CPU não é suportado; usando fp32")
        return "fp32"
    return requested


def configure_backend(device: DeviceInfo) -> None:
    """Ajustes específicos do dispositivo (TF32 só existe em CUDA)."""
    if device.backend == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def allocated_memory_gb(device: DeviceInfo) -> float:
    """Memória alocada pelo PyTorch no dispositivo, em GiB (amostrada durante o treino)."""
    if device.backend == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30
    if device.backend == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    return 0.0
