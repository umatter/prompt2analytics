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

### Development Machine (Primary)

*To be filled in when benchmarks are run*

```markdown
## Profile: Development-Primary (YYYY-MM-DD)

### CPU
- Model:
- Cores:
- Base Clock:
- L3 Cache:

### Memory
- Total:
- Type:

### Operating System
- OS:
- Kernel:

### Software Versions
- Rust:
- R:
- Python:
```

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
