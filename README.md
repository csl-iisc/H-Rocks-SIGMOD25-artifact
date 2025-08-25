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
- **What’s provided:** End-to-end scripts to **compile → run → parse → plot**, plus per-figure convenience scripts.
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

artifact-root/
├─ pmem-rocksdb/
│ └─ h-rocks/ # H-Rocks lives here
│ ├─ src/ # (hrocksdb.h, hrocksdb.cu, …)
│ ├─ benchmarks/ # (test_puts, test_gets, …)
│ ├─ Makefile / scripts …
│ └─ compile.sh / run.sh / parse.sh / plot.sh
├─ viper/
│ ├─ compile.sh / run.sh / parse.sh / plot.sh
│ └─ (sources / helpers)
├─ plush/
│ ├─ compile.sh / run.sh / parse.sh / plot.sh
│ └─ (sources / helpers)
├─ utree/
│ ├─ compile.sh / run.sh / parse.sh / plot.sh
│ └─ (sources / helpers)
├─ gpkvs/
│ ├─ compile.sh / run.sh / parse.sh / plot.sh
│ └─ (sources / helpers)
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
│ ├─ run_figure8a.sh … run_figure12d.sh
│ └─ shared helpers
└─ out/ # created by runs: logs/, parsed/, plots/


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
make hrocksdb            # produces hrocksdb.a
# or: make hrocksdb -DENABLE_DEBUG

# 5) Sanity tests (microbenchmarks)
make bin/test_puts
./bin/test_puts -n 100000 -k 8 -v 8

make bin/test_gets
./bin/test_gets -p 100000 -n 100000 -k 8 -v 8
```
If these succeed and produce output logs, you’re ready for figure scripts.

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

Install per NVIDIA guidance:

CUDA Toolkit ≤ 12.1

NVIDIA Driver ≥ 530.xx

Verify:

nvcc --version
nvidia-smi

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

Top-level orchestrators (from scripts/):

```bash
./compile_all.sh   # builds all systems (h-rocks, viper, plush, utree, gpkvs)
./run_all.sh       # runs all experiments across systems
./parse_all.sh     # parses logs -> CSVs
./plot_all.sh      # generates all plots
```

One-shot pipeline:

```bash
./end_to_end.sh    # compile + run + parse + plot everything
```

Per-system (inside each system dir):

```bash 
./compile.sh
./run.sh
./parse.sh
./plot.sh
```

Most scripts accept env overrides (e.g., CUDA_VISIBLE_DEVICES, PMEM_DIR, dataset sizes). See script headers.

## 9. Reproducing Paper Figures

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

# Figure 11
./run_figure11a.sh
./run_figure11b.sh

# Figure 12
./run_figure12a.sh
./run_figure12b.sh
./run_figure12c.sh
./run_figure12d.sh
```

Each script performs the minimal compile → run → parse → plot for that figure.

## 10. Outputs & Verification

All outputs are placed under out/:

```bash 
out/
  logs/                   # raw logs from runs
  parsed/                 # CSVs after parsing
  plots/
    fig8a.pdf
    fig8b.pdf
    ...
    fig12d.pdf
```

Verification checklist

Per-figure script prints a final OK/summary line.

Expected plot(s) appear in out/plots/.

CSVs have sensible non-zero metrics.

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

### PMEM path confusion (/pmem/...)
Some scripts auto-prefix /pmem. If you also pass a path including /pmem, you may end up with a non-existent double prefix like /pmem/pmem/....
✔️ Use a relative or non-/pmem path if the script auto-prefixes; or export PMEM_DIR explicitly.

### Permissions
PMEM setup requires root; experiments do not (except external profilers you might use).

### CUDA/Driver mismatch
Ensure nvidia-smi driver is compatible with CUDA Toolkit (Toolkit ≤ 12.1, Driver ≥ 530.xx).

### Viper errors
Viper requires that the database it is trying to create must not already exist. Please clear /pmem/ before running viper experiments. Our scripts take care of this.

### Plush errors
Plush requires plush_table in /pmem/. Please create the folder before executing Plush. Our scripts take care of it. 

### Docker & PMEM
Use --privileged (or grant required caps) and mount the PMEM path.

## 14. License & Citation

Please cite our paper:

Shweta Pandey and Arkaprava Basu. H-Rocks: CPU-GPU accelerated Heterogeneous RocksDB on Persistent Memory. Proc. ACM on Management of Data (SIGMOD), 2025. DOI: 10.1145/3709694.

## References
[1] RocksDB (Code) — https://github.com/facebook/rocksdb

[2] pmem-rocksDB (Code) — https://github.com/pmem/pmem-rocksdb

