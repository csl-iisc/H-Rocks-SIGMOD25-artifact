# H-Rocks (SIGMOD Artifact Evaluation)

## H-Rocks: CPU-GPU accelerated Heterogeneous RocksDB on Persistent Memory

H-Rocks extends the popular key-value store RocksDB by Meta [1] to leverage both GPU and CPU.
H-Rocks significantly improves the throughput of RocksDB.
This repository provides the source code for H-RocksDB, designed to accelerate a wide range of RocksDB operations by selectively offloading them to the GPU. 

This README provides a peek into the key-value store and a high-level view of source code organization.

For full details, refer to our paper:
<pre>
<b>H-Rocks: CPU-GPU accelerated Heterogeneous RocksDB on Persistent Memory</b>
Shweta Pandey and Arkaprava Basu
<i>Proceedings of the ACM on Management of Data, Volume 3, Issue 1 (SIGMOD), 2025</i>
DOI: https://doi.org/10.1145/3709694
</pre>

---

## 0. Artifact at a Glance

- **Goal:** Build H-Rocks and baselines; run bundled experiments; reproduce **Figures 8–12** from the paper.
- **What's provided:** End-to-end scripts to **compile → run → parse → plot**, plus per-figure convenience scripts.
- **AE scope:** Functional and results-reproduced for the included figures (see §8, §11).
- **Contact:** Please contact on hotcrp for any queries or reach out to shwetapandey@iisc.ac.in. 

---

## 1. Hardware & Software Requirements

H-Rocks is built on top of pmem-rocksdb and uses NVIDIA GPUs.

- **GPU:** NVIDIA SM compute capability **≥ 7.5 and ≤ 8.6**
- **CPU:** x86_64 
- **OS:** Linux **5.4.0-169-generic** (or compatible)
- **GCC:** **≥ 5.3.0**
- **CUDA Toolkit:** **≥ 8.0 and ≤ 12.1**
- **NVIDIA Driver:** **≥ 530.xx**
- **Persistent Memory:** Intel Optane (or compatible) in **App-Direct** mode with **interleaving**

---

## 2. Artifact Layout

After extracting the artifact, you should see:

```csharp
artifact-root/
├─ pmem-rocksdb/
│ └─ h-rocks/ # H-Rocks lives here
│ ├─ src/ # (hrocksdb.h, hrocksdb.cu, …)
│ ├─ benchmarks/ # (test_puts, test_gets, …)
│ ├─ Makefile / scripts …
│ └─ compile.sh / run.sh / parse.sh 
├─ viper/
│ ├─ compile.sh / run.sh / parse.sh 
├─ plush/
│ ├─ compile.sh / run.sh / parse.sh 
├─ utree/
│ ├─ compile.sh / run.sh / parse.sh 
├─ gpkvs/
│ ├─ compile.sh / run.sh / parse.sh 
├─ scripts/
│ ├─ dependencies.sh
│ ├─ pmem-setup/
│ │ ├─ teardown.bashrc
│ │ ├─ preboot.bashrc
│ │ └─ config-setup.bashrc
│ ├─ compile_all.sh
│ ├─ run_all.sh
│ ├─ parse_all.sh
│ ├─ plot_all.sh
│ ├─ end_to_end.sh
│ ├─ run_figure8a.sh … run_figure13d.sh
└─ out/ # created by runs: logs/, parsed/, plots/
```

Each top-level system dir (`pmem-rocksdb/h-rocks`, `viper/`, `plush/`, `utree/`, `gpkvs/`) contains its own `compile.sh`, `run.sh`, `parse.sh`, `plot.sh`. Top-level `*_all.sh` iterate into each system directory.

---

## 3. Quick Start (Smoke Test)

If you just want to verify the pipeline works end to end:

```bash
# From artifact root

# 1) Install dependencies (PMDK, etc.)
chmod +x scripts/dependencies.sh
sudo ./scripts/dependencies.sh

# 2) (Optional but recommended) PMEM setup (requires reboot; see §4)
# sudo ./scripts/pmem-setup/teardown.bashrc
# sudo ./scripts/pmem-setup/preboot.bashrc
# sudo su
# ./scripts/pmem-setup/config-setup.bashrc
# exit

# 3) Build pmem-rocksdb
cd pmem-rocksdb
make ROCKSDB_ON_DCPMM=1 install-static -j
cd ..

# 4) Build H-Rocks (release)
cd pmem-rocksdb/h-rocks
make                     # builds libgpu.a and all test binaries
# or: make lib DEBUG=1   # build only libgpu.a with debug flags

# 5) Sanity tests (microbenchmarks)
make bin/test_puts
./bin/test_puts -n 100000 -k 8 -v 8

make bin/test_gets
./bin/test_gets -p 100000 -n 100000 -k 8 -v 8
```
If these succeed and produce output logs, you're ready for figure scripts.

## 4. PMEM Setup (Bare-Metal)

Danger: This reconfigures namespaces and reboots. Do this only on a dedicated AE machine.

1) Install PMEM deps
```bash
chmod +x scripts/dependencies.sh
sudo ./scripts/dependencies.sh
```

2) Tear down old configs
```bash
sudo ./scripts/pmem-setup/teardown.bashrc
```

3) Pre-boot cleanup (destroys namespaces + reboots)
```bash
sudo ./scripts/pmem-setup/preboot.bashrc
```

4) Configure App-Direct + interleaving (as root)
```bash
sudo su
./scripts/pmem-setup/config-setup.bashrc
exit
```

This prepares an interleaved namespace and (by default) a /pmem/... path used by scripts.
If your mount differs, export PMEM_DIR and/or adjust scripts accordingly.

## 5. CUDA & NVIDIA Drivers

### Installing CUDA Toolkit 12.1

For Ubuntu/Debian systems:

```bash
# Download CUDA 12.1 installer
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run

# Run installer
sudo sh cuda_12.1.0_530.30.02_linux.run

# Follow the prompts:
# - Accept the license
# - Install CUDA Toolkit 12.1
# - Install NVIDIA Driver 530.30.02 (if not already installed)
```

For other distributions or manual download, visit:
https://developer.nvidia.com/cuda-12-1-0-download-archive

### Setting up CUDA PATH

**Important:** CUDA binaries are in two locations that both need to be in your PATH:
- `/usr/local/cuda-12.1/bin` (main CUDA tools: nvcc, etc.)
- `/usr/local/cuda-12.1/nvvm/bin` (includes cicc compiler)

#### Option 1: Add to `.bashrc` (Permanent - Recommended)

```bash
# Add CUDA 12.1 to your PATH permanently
echo 'export PATH=/usr/local/cuda-12.1/bin:/usr/local/cuda-12.1/nvvm/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc

# Reload your shell configuration
source ~/.bashrc
```

#### Option 2: Export for Current Session Only

```bash
export PATH=/usr/local/cuda-12.1/bin:/usr/local/cuda-12.1/nvvm/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.1
```

### Verify Installation

```bash
# Check CUDA compiler version
nvcc --version

# Check NVIDIA driver
nvidia-smi

# Verify cicc is accessible
which cicc
# Should output: /usr/local/cuda-12.1/nvvm/bin/cicc
```

**Expected output for `nvcc --version`:**
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
```

### Troubleshooting CUDA Issues

**Problem: `cicc: not found` error during compilation**

Solution:
1. Verify CUDA 12.1 installation: `ls /usr/local/cuda-12.1/nvvm/bin/cicc`
2. Ensure `/usr/local/cuda-12.1/nvvm/bin` is in your PATH
3. Check that you've sourced your `.bashrc`: `source ~/.bashrc`

**Problem: Multiple CUDA versions installed**

If you have multiple CUDA versions on your system:
```bash
# Find all installed CUDA versions
ls -d /usr/local/cuda*

# Find where cicc is located
find /usr/local/cuda* -name cicc 2>/dev/null

# Make sure CUDA 12.1 comes first in your PATH
echo $PATH | grep cuda-12.1
```

To switch to CUDA 12.1, ensure it appears first in your PATH before other CUDA versions.

## 6. Building H-Rocks (Manual)

From pmem-rocksdb/h-rocks:

```bash
make hrocksdb                  # release
# or
make hrocksdb -DENABLE_DEBUG   # debug with extra logs
```

## 7. Running H-Rocks Manually (Optional)

From pmem-rocksdb/h-rocks:

```bash
make bin/test_puts
./bin/test_puts -n <num_keys> -k <key_size> -v <value_size>

make bin/test_gets
./bin/test_gets -p <prefill_keys> -n <num_keys> -k <key_size> -v <value_size>
```

Other tests follow the same pattern.

## 8. End-to-End: Compile → Run → Parse → Plot

One shot end to end: 

```bash
cd scripts/
./end_to_end.sh    # compile + run + parse + plot everything
```
That one command will:
Compile all systems (via scripts/compile_all.sh if present, else per-system compile.sh),
Run figure scripts (all the ones currently included in scripts/),
Parse and plot, then
Collect every plot PDF in a single folder.

### Prerequisites

Python 3 + matplotlib for plotting:

```bash 
python3 -m pip install --user matplotlib
```

The plotting helper exists and is executable: scripts/plot_lines_from_csvs.py

Make scripts executable (once):

```bash
chmod +x scripts/*.sh
chmod +x scripts/plot_lines_from_csvs.py
```

### Environment knobs (optional)

Limit the data sizes used by parsers/plots, or choose a GET value size for systems that expose it:

export SIZES="10000 100000 1000000"   # only these sizes will be parsed/plotted
export VAL_SIZES=8                    # e.g., GET value size selector (used by some GET parsers)
bash scripts/run_end_to_end.sh

### Where do results appear?
Per-figure outputs
Each figure runner writes its CSVs (one per system) and a combined PDF to a dedicated folder under out/.
Actual names depend on your figure scripts; examples:

```csharp
out/
  fig8a/                 # example name; yours may differ
    hrocks_puts.csv
    pmem_puts.csv
    viper_puts.csv
    plush_puts.csv
    fig8a_puts.pdf
  fig9b/
    ...
  fig13c/
    ...
```
CSV format is consistently: size,throughput_ops_per_s

### Final "all plots" folder

At the end, every PDF discovered under out/fig*/ is copied into:

out/plots_all/
  <figure-folder>_<original-pdf-name>.pdf
  plots_manifest.txt   # index of all PDFs collected

## 9. Reproducing Individual Paper Figures

We provide per-figure convenience scripts (from scripts/):

```bash
# Figure 8
./run_figure8a.sh
./run_figure8b.sh
./run_figure8c.sh
./run_figure8d.sh

# Figure 9
./run_figure9a.sh
./run_figure9b.sh

# Figure 10
./run_figure10a.sh
./run_figure10b.sh

# Figure 12
./run_figure12a.sh
./run_figure12b.sh
./run_figure12c.sh
./run_figure12d.sh
```

Each script performs the minimal compile → run → parse → plot for that figure.
To reproduce Figure 11, we need to make manual changes. This requires setting putsWithValues = true; in the file pmem-rocksdb/h-rocks/batch.cu.

## 11. H-Rocks Source & API Overview (for reference)

Key files: pmem-rocksdb/h-rocks/src/hrocksdb.h, pmem-rocksdb/h-rocks/src/hrocksdb.cu.

Main APIs


```cpp
// Ctor / Dtor

HRocksDB(Config config);

~HRocksDB();

// KVS setup ops

void HOpen(std::string fileLocation);

void Close();

// KVS ops

void Put(const std::string& key, const std::string& value);

void Get(const std::string& key);

void Delete(const std::string& key);

void Range(const std::string& startKey, const std::string& endKey);

void Merge(const std::string& key);

// Config knobs

setMemtableSize(uint64_t size);

setNumMemtables(int num);

setBatchSize(uint64_t size);
```

## 12. Docker (Alternative)

If you prefer a containerized setup:

```bash
# Install Docker (Ubuntu)
sudo apt install docker.io

# Build image
docker build . -t hrocks-ae:v1

# Enable GPU access via NVIDIA Container Toolkit (host must have it).
# Then run:
docker run -it --runtime=nvidia --gpus all --privileged \
  -v /pmem:/pmem \
  -v "$(pwd)":/workspace \
  -w /workspace \
  hrocks-ae:v1 bash
```

We mount /pmem so the container can access host PMEM. Adjust if your PMEM path differs.

## 13. Common Pitfalls & Troubleshooting

1. PMEM path confusion (/pmem/...)
Some scripts auto-prefix /pmem. If you also pass a path including /pmem, you may end up with a non-existent double prefix like /pmem/pmem/....
Use a relative or non-/pmem path if the script auto-prefixes; or export PMEM_DIR explicitly.
H-Rocks runners now honor `HR_PMEM_DIR`. By default they fall back to `pmem-rocksdb/h-rocks/tmp/{pmem,shm}` so sandboxed runs "just work". Set `HR_PMEM_DIR=/pmem` on bare metal to target a real DAX mount.

2. Permissions
PMEM setup requires root; experiments do not (except external profilers you might use).

3. CUDA/Driver mismatch
Ensure nvidia-smi driver is compatible with CUDA Toolkit (Toolkit ≤ 12.1, Driver ≥ 530.xx).

4. Viper errors
Viper requires that the database it is trying to create must not already exist. Please clear /pmem/ before running viper experiments. Our scripts take care of this.

5. Plush errors
Plush requires plush_table in /pmem/. Please create the folder before executing Plush. Our scripts take care of it. 

6. Docker & PMEM
Use --privileged (or grant required caps) and mount the PMEM path.

## 14. License & Citation

Please cite our paper:

Shweta Pandey and Arkaprava Basu. H-Rocks: CPU-GPU accelerated Heterogeneous RocksDB on Persistent Memory. Proc. ACM on Management of Data (SIGMOD), 2025. DOI: 10.1145/3709694.

## References
[1] RocksDB (Code) — https://github.com/facebook/rocksdb

[2] pmem-rocksDB (Code) — https://github.com/pmem/pmem-rocksdb
