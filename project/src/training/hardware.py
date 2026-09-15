"""
Descrição:
    Diagnóstico e configuração do dispositivo de treino.

    O treino do TCC é feito em uma RTX A2000 de 12 GB (Ampere, capability 8.6),
    portanto o caminho principal é CUDA. Este módulo exige CUDA por padrão e
    falha com mensagem acionável quando ela não está disponível, em vez de cair
    silenciosamente para CPU — um treino de 185 mil exemplos em CPU levaria dias
    e o usuário só descobriria depois.

    Também resolve a precisão numérica contra a capacidade real da placa:
    bfloat16 é preferido a float16 em Ampere porque tem a mesma vazão nos tensor
    cores e dispensa o `GradScaler`, eliminando a classe de falha em que o
    treino pula passos por overflow de gradiente.

Uso:
    from src.training import hardware
    info = hardware.describe_device()
    hardware.require_cuda(allow_cpu=False)
    precision = hardware.resolve_precision("bf16", info)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from src.data_prep import console

#: Capability mínima de CUDA com suporte de hardware a bfloat16 (Ampere).
_BF16_MIN_CAPABILITY: tuple[int, int] = (8, 0)


@dataclass(frozen=True)
class DeviceInfo:
    """
    Descrição:
        Retrato do dispositivo disponível no momento da execução.

    Atributos:
        available: Se há GPU CUDA disponível.
        name: Nome da GPU, ou `"cpu"`.
        total_vram_gb: VRAM total da GPU, em GiB. Zero em CPU.
        capability: Compute capability da GPU, ex.: `(8, 6)` para a A2000.
        bf16_supported: Se a GPU tem suporte de hardware a bfloat16.
        torch_version: Versão do PyTorch instalada.
        cuda_version: Versão de CUDA com que o PyTorch foi compilado.
        n_gpus: Número de GPUs visíveis.
    """

    available: bool
    name: str
    total_vram_gb: float
    capability: tuple[int, int]
    bf16_supported: bool
    torch_version: str
    cuda_version: str | None
    n_gpus: int

    def to_dict(self) -> dict[str, object]:
        """
        Descrição:
            Serializa o retrato do dispositivo, para gravar no manifest da
            execução.

        Retorna:
            Dicionário com todos os campos.
        """
        return {
            "available": self.available,
            "name": self.name,
            "total_vram_gb": round(self.total_vram_gb, 2),
            "capability": list(self.capability),
            "bf16_supported": self.bf16_supported,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "n_gpus": self.n_gpus,
        }


def describe_device() -> DeviceInfo:
    """
    Descrição:
        Coleta as informações do dispositivo de treino ativo.

    Retorna:
        `DeviceInfo` com os dados da GPU, ou um retrato de CPU se CUDA não
        estiver disponível.
    """
    if not torch.cuda.is_available():
        return DeviceInfo(
            available=False,
            name="cpu",
            total_vram_gb=0.0,
            capability=(0, 0),
            bf16_supported=False,
            torch_version=torch.__version__,
            cuda_version=None,
            n_gpus=0,
        )

    properties = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    return DeviceInfo(
        available=True,
        name=properties.name,
        total_vram_gb=properties.total_memory / 1024**3,
        capability=capability,
        bf16_supported=capability >= _BF16_MIN_CAPABILITY,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        n_gpus=torch.cuda.device_count(),
    )


def require_cuda(allow_cpu: bool = False) -> None:
    """
    Descrição:
        Verifica que há GPU CUDA disponível e interrompe a execução se não
        houver.

    Args:
        allow_cpu: Se `True`, apenas emite aviso em vez de falhar. Destinado a
            testes de fumaça, nunca ao treino real.

    Levanta:
        RuntimeError: Se CUDA não estiver disponível e `allow_cpu` for `False`.
    """
    if torch.cuda.is_available():
        return

    message = (
        "CUDA não está disponível — o treino foi configurado para GPU NVIDIA.\n"
        f"  torch instalado: {torch.__version__} (CUDA de compilação: {torch.version.cuda})\n"
        "  Verifique, na máquina de treino:\n"
        "    1. `nvidia-smi` responde e mostra a GPU\n"
        "    2. o wheel do torch é a build com CUDA (o do PyPI para Windows é CPU-only)\n"
        "       -> `uv sync` deve resolver o torch pelo índice `pytorch-cu126`\n"
        "    3. `python -c \"import torch; print(torch.cuda.is_available())\"` retorna True\n"
        "  Para um teste de fumaça em CPU, use `--allow-cpu` (não serve para o treino real)."
    )
    if not allow_cpu:
        raise RuntimeError(message)
    console.warn("executando sem CUDA (--allow-cpu): use apenas para teste de fumaça")


def resolve_precision(
    requested: Literal["bf16", "fp16", "fp32"], info: DeviceInfo
) -> Literal["bf16", "fp16", "fp32"]:
    """
    Descrição:
        Ajusta a precisão pedida à capacidade real do dispositivo, rebaixando
        quando necessário: `bf16` -> `fp16` -> `fp32`.

    Args:
        requested: Precisão desejada.
        info: Retrato do dispositivo.

    Retorna:
        A precisão efetivamente utilizável.
    """
    if not info.available:
        if requested != "fp32":
            console.warn(f"sem GPU: precisão {requested} rebaixada para fp32")
        return "fp32"

    if requested == "bf16" and not info.bf16_supported:
        console.warn(
            f"GPU {info.name} (capability {info.capability[0]}.{info.capability[1]}) "
            "não suporta bfloat16; usando fp16"
        )
        return "fp16"

    return requested


def enable_tf32(enabled: bool = True) -> None:
    """
    Descrição:
        Liga o TF32 nas multiplicações de matriz e no cuDNN.

        Em GPUs Ampere é ganho de velocidade praticamente gratuito, com perda de
        precisão irrelevante para fine-tuning de classificação.

    Args:
        enabled: Se `True`, ativa o TF32.
    """
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def reset_peak_memory() -> None:
    """
    Descrição:
        Zera o contador de pico de memória da GPU, para medir o consumo de uma
        fase específica do treino.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb() -> float:
    """
    Descrição:
        Consulta o pico de VRAM alocado desde o último `reset_peak_memory`.

    Retorna:
        Pico em GiB, ou 0.0 se não houver GPU.
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def log_device(info: DeviceInfo) -> None:
    """
    Descrição:
        Imprime o bloco de diagnóstico do dispositivo no início da execução.

    Args:
        info: Retrato do dispositivo.
    """
    console.header("Dispositivo")
    if not info.available:
        console.warn(f"nenhuma GPU CUDA — executando em CPU (torch {info.torch_version})")
        return
    console.info(f"GPU: {info.name}  ({info.n_gpus} visível(is))")
    console.info(f"VRAM: {info.total_vram_gb:.1f} GiB")
    console.info(
        f"capability: {info.capability[0]}.{info.capability[1]}  "
        f"bfloat16: {'sim' if info.bf16_supported else 'não'}"
    )
    console.info(f"torch {info.torch_version} (CUDA {info.cuda_version})")
