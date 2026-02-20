# Hardware Profiles

This document records hardware configurations used for benchmarking.

## Profile Template

When running benchmarks, document your hardware:

```markdown
## Profile: [Name/Date]

### CPU
- Model: [e.g., AMD Ryzen 9 5900X]
- Cores: [e.g., 12 cores / 24 threads]
- Base Clock: [e.g., 3.7 GHz]
- Boost Clock: [e.g., 4.8 GHz]
- L3 Cache: [e.g., 64 MB]

### Memory
- Total: [e.g., 64 GB]
- Type: [e.g., DDR4-3200]
- Channels: [e.g., Dual channel]

### Storage
- Type: [e.g., NVMe SSD]
- Model: [e.g., Samsung 980 Pro]

### Operating System
- OS: [e.g., Ubuntu 22.04.3 LTS]
- Kernel: [e.g., 6.2.0-39-generic]

### Software Versions
- Rust: [e.g., 1.75.0]
- R: [e.g., 4.3.2]
- Python: [e.g., 3.11.5]
```

---

## Collecting Hardware Info

### Linux

```bash
# Save to hardware_info.json
{
  echo "{"
  echo "  \"timestamp\": \"$(date -Iseconds)\","
  echo "  \"cpu\": {"
  lscpu | grep -E "^(Model name|CPU\(s\)|Thread|Core|MHz|L[123])" | sed 's/^/    "/; s/: */": "/; s/$/",/'
  echo "  },"
  echo "  \"memory\": {"
  free -h | grep Mem | awk '{print "    \"total\": \"" $2 "\","}'
  echo "  },"
  echo "  \"os\": {"
  echo "    \"name\": \"$(lsb_release -ds 2>/dev/null || cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2)\","
  echo "    \"kernel\": \"$(uname -r)\""
  echo "  },"
  echo "  \"versions\": {"
  echo "    \"rust\": \"$(rustc --version | cut -d' ' -f2)\","
  echo "    \"r\": \"$(R --version | head -1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+')\","
  echo "    \"python\": \"$(python3 --version | cut -d' ' -f2)\""
  echo "  }"
  echo "}"
} > hardware_info.json
```

### macOS

```bash
system_profiler SPHardwareDataType SPSoftwareDataType
```

### Windows

```powershell
Get-ComputerInfo | Select-Object CsProcessors, CsPhyicallyInstalledMemory, WindowsVersion
```

---

## Reference Profiles

### NVIDIA DGX Spark (GPU Benchmarks)

Profile used for GPU vs CPU benchmarks (2026-02-20).

#### CPU
- Model: ARM Cortex-X925 (10 cores, up to 4.0 GHz) + Cortex-A725 (10 cores)
- Total CPUs: 20 (no SMT)
- Architecture: aarch64

#### GPU
- Model: NVIDIA GB10 (Grace Blackwell)
- VRAM: 12 GB
- Compute Capability: 12.1
- Memory: Unified (CPU+GPU shared address space)

#### Memory
- Total: 120 GB unified

#### Operating System
- OS: Linux (NVIDIA DGX Spark)
- Kernel: 6.14.0-1015-nvidia

#### Software Versions
- Rust: 1.93.0
- CUDA: via cudarc 0.12 (cuBLAS + cuSOLVER)

#### Notes
- Unified memory eliminates explicit host-device transfers
- ARM OpenBLAS is highly optimized for Cortex-X925 NEON/SVE
- GPU crossover thresholds are calibrated for this specific hardware

---

### Development Machine (Primary)

*To be filled in when benchmarks are run on non-GPU hardware.*

---

## Performance Normalization

When comparing across different hardware, consider normalization:

1. **Single-thread Geekbench score**: Good general baseline
2. **LINPACK benchmark**: Relevant for linear algebra
3. **Relative comparison**: Express as ratio to reference machine

Example:
```
If Reference Machine = 100%, and benchmark shows:
- Development Machine: 85% (slower)
- Cloud Instance: 120% (faster)

Normalize results by dividing by hardware factor.
```

## Cloud Instance Profiles

For reproducible benchmarks, use standard cloud instances:

| Provider | Instance | vCPU | Memory | Notes |
|----------|----------|------|--------|-------|
| AWS | c6i.xlarge | 4 | 8 GB | Compute optimized |
| GCP | c2-standard-4 | 4 | 16 GB | Compute optimized |
| Azure | Standard_F4s_v2 | 4 | 8 GB | Compute optimized |

Using a consistent cloud instance allows reproducibility across researchers.
