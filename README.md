# redp

## Lightning-fast Emon Data Processor implemented in 100% Rust

### Pre-requisite
Rust compiler and Cargo installed, if not
`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

### Quick Guide
```
$ git clone https://github.com/intel-sandbox/redp.git
# Build the binary
$ cargo build --release
$ ./target/release/redp parse /path/to/emon.dat --xml /path/to/platform.xml -sctn
# Run ./target/release/redp help for more usages
```

### Performance BKM (Following methods can be applied to improve the performance of `redp`)
#### 1. If `redp` is running on Linux, enable THP (Transparent Hugepage)
```
# Recommend to configure THP via grub

# grub configurations
$ sudo vim /etc/default/grub

# modify the line of "GRUB_CMDLINE_LINUX_DEFAULT"
# to GRUB_CMDLINE_LINUX_DEFAULT="transparent_hugepage=always"

# make it effective and restart the system
$ sudo update-grub
$ sudo reboot

# Verify
$ cat /sys/kernel/mm/transparent_hugepage/enabled
-> [always] madvise never
```

| **Platform - Duration** | **Size on disk (emon.dat)** | **w/o details** | **w/o details** | **Speedup** | **w/ details** | **w/ details** | **Speedup** |
| :---------------------- | --------------------------: | --------------: | --------------: | ----------: | -------------: | -------------: | ----------: |
|                         |                             |      REDP (sec) |     PyEDP (sec) |             |     REDP (sec) |    PyEDP (sec) |             |
| GNR – 30s               |                      3.8 MB |            0.98 |           13.34 |   **13.6x** |           2.35 |          37.13 |   **15.8x** |
| GNR – 10m               |                       74 MB |            1.92 |           44.69 |   **23.3x** |           5.72 |         212.71 |   **37.2x** |
| GNR – 1h                |                      461 MB |            4.60 |           72.46 |   **15.8x** |          15.23 |         963.25 |   **63.2x** |
| GNR – 3h                |                      1.4 GB |           10.83 |          163.42 |   **15.1x** |          39.15 |        2833.52 |   **72.4x** |
| GNR – 12h               |                      5.4 GB |           37.93 |          602.02 |   **15.9x** |        *169.48 |            Err |      **NA** |
| SRF – 30s               |                      4.2 MB |            0.62 |           13.36 |   **21.6x** |           1.46 |          29.88 |   **20.5x** |
| SRF – 10m               |                       84 MB |            1.56 |           31.57 |   **20.2x** |           4.76 |         276.90 |   **58.2x** |
| SRF – 1h                |                      503 MB |            4.47 |           76.96 |   **17.2x** |          17.45 |        1475.49 |   **84.6x** |
| SRF – 3h                |                      1.5 GB |           10.82 |          188.36 |   **17.4x** |          47.48 |        4508.89 |   **95.0x** |
| SRF – 12h               |                      5.9 GB |           38.60 |          690.77 |   **17.9x** |        *415.88 |            Err |      **NA** |
| **Avg.**                |                             |                 |                 |     17.8x   |                |                |     55.9x   |

## Contribution
Your contribution is highly appreciated. Do not hesitate to open an issue or a pull request. Note that any contribution submitted for inclusion in the project will be licensed according to the terms given in [LICENSE](LICENSE).