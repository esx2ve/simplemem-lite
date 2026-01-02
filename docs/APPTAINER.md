# Apptainer Setup for HPC Clusters

This guide covers installing and configuring Apptainer (formerly Singularity) on HPC clusters without root access, optimized for running FalkorDB and other containers.

## Why Apptainer?

- **No Docker daemon required** - runs as unprivileged user
- **No sudo needed** - install with `--without-suid` flag
- **HPC compatible** - works with job schedulers (SLURM, PBS)
- **SIF format** - single file containers, easy to manage

## Prerequisites

- Linux system (tested on Ubuntu 20.04)
- Go 1.21+ (for building Apptainer)
- Access to a large storage partition (home directories often have quotas)

---

## 1. Install Go (if not available)

```bash
# Download Go
wget https://go.dev/dl/go1.23.4.linux-amd64.tar.gz

# Extract to user space
mkdir -p ~/.local
tar -xzf go1.23.4.linux-amd64.tar.gz -C ~/.local/

# Add to PATH (add to ~/.bashrc for persistence)
export PATH=~/.local/go/bin:$PATH
export GOPATH=~/.local/go

# Verify
go version
```

---

## 2. Build and Install Apptainer

```bash
# Clone repository
git clone https://github.com/apptainer/apptainer.git ~/apptainer-src
cd ~/apptainer-src

# Configure for user-space installation (NO SUDO)
./mconfig --without-suid --prefix=~/.local/apptainer

# Build (use multiple cores)
cd builddir
make -j$(nproc)

# Install
make install

# Verify installation
~/.local/apptainer/bin/apptainer --version
```

### Key Flag: `--without-suid`

This enables rootless installation. Without it, Apptainer requires root privileges for the `starter-suid` binary.

---

## 3. Configure Storage Locations

**CRITICAL:** HPC home directories often have strict quotas (e.g., 10GB). Container images and caches can easily exceed this.

### Create directories on large storage

```bash
# Replace /weka/home-username with your large storage path
mkdir -p /weka/home-username/apptainer/{cache,tmp,images}
```

### Set environment variables

Add to `~/.bashrc`:

```bash
# Apptainer storage configuration
export APPTAINER_CACHEDIR=/weka/home-username/apptainer/cache
export APPTAINER_TMPDIR=/weka/home-username/apptainer/tmp

# Add Apptainer to PATH
export PATH=~/.local/apptainer/bin:$PATH
```

### Environment Variables Reference

| Variable | Purpose | Default |
|----------|---------|---------|
| `APPTAINER_CACHEDIR` | OCI blobs, layers, build cache | `~/.apptainer/cache` |
| `APPTAINER_TMPDIR` | Temporary files during build/pull | System temp |
| `APPTAINER_BIND` | Default bind mounts | None |

---

## 4. Install squashfuse (CRITICAL for Performance)

Without `squashfuse`, Apptainer must extract SIF files to a temporary sandbox on every run. This can take **2+ minutes** for large images.

With `squashfuse`, containers start in **~0.1 seconds**.

### Download and extract packages

```bash
cd /tmp

# Download squashfuse and its library (no sudo needed)
apt-get download squashfuse libsquashfuse0

# Extract to user space
dpkg-deb -x libsquashfuse0_*.deb ~/.local
dpkg-deb -x squashfuse_*.deb ~/.local

# Verify extraction
ls ~/.local/usr/bin/squashfuse
ls ~/.local/usr/lib/x86_64-linux-gnu/libsquashfuse.so.0
```

### Create wrapper script

Apptainer runs squashfuse in a namespace where `$HOME` may change. The wrapper must use **absolute paths**:

```bash
cat > ~/.local/bin/squashfuse << 'EOF'
#!/bin/bash
# IMPORTANT: Use absolute paths, not $HOME (namespace changes $HOME)
export LD_LIBRARY_PATH=/admin/home-username/.local/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
exec /admin/home-username/.local/usr/bin/squashfuse "$@"
EOF

chmod +x ~/.local/bin/squashfuse
```

**Replace `/admin/home-username` with your actual home directory path** (use `echo $HOME` to find it).

### Symlink starter binary

Apptainer may look for the starter in a different location:

```bash
mkdir -p ~/.local/libexec/apptainer/bin
ln -sf ~/.local/apptainer/libexec/apptainer/bin/starter \
       ~/.local/libexec/apptainer/bin/starter
```

### Update PATH

Add to `~/.bashrc`:

```bash
# squashfuse for fast SIF mounting
export PATH=~/.local/bin:$PATH
```

### Verify squashfuse works

```bash
source ~/.bashrc
squashfuse --help  # Should show usage

# Test with a container (should be fast, no "Converting SIF" message)
apptainer exec docker://hello-world echo "Hello"
```

---

## 5. Pull and Run Containers

### Pull a Docker image

```bash
cd /weka/home-username/apptainer/images

# Pull FalkorDB
apptainer pull docker://falkordb/falkordb:latest

# Pull other images
apptainer pull docker://python:3.11-slim
```

### Run a container

```bash
# One-off execution
apptainer exec falkordb_latest.sif echo "Hello from container"

# Interactive shell
apptainer shell falkordb_latest.sif

# Run default entrypoint
apptainer run falkordb_latest.sif
```

### Run as background instance

```bash
# Start instance
apptainer instance start \
  --bind /weka/home-username/data:/data \
  falkordb_latest.sif \
  my-instance

# Execute commands in instance
apptainer exec instance://my-instance redis-cli PING

# List running instances
apptainer instance list

# Stop instance
apptainer instance stop my-instance
```

---

## 6. FalkorDB Specific Setup

### Start FalkorDB with graph module

```bash
# Start instance with data directory bound
apptainer instance start \
  --bind /weka/home-username/apptainer/falkordb-data:/data \
  /weka/home-username/apptainer/images/falkordb_latest.sif \
  falkordb

# Start redis-server WITH the FalkorDB module
# (default entrypoint may not load the module correctly)
apptainer exec instance://falkordb \
  redis-server --port 6379 \
  --loadmodule /var/lib/falkordb/bin/falkordb.so \
  --daemonize yes

# Verify graph module is loaded
apptainer exec instance://falkordb redis-cli MODULE LIST
# Should show: name=graph, ver=41411

# Test a graph query
apptainer exec instance://falkordb \
  redis-cli GRAPH.QUERY test "CREATE (n:Person {name:'Alice'}) RETURN n"
```

### Connect from Python

```python
from falkordb import FalkorDB

# Connect to the containerized FalkorDB
db = FalkorDB(host='localhost', port=6379)
graph = db.select_graph('simplemem')

# Run a query
result = graph.query("MATCH (n) RETURN n LIMIT 5")
```

### Cleanup

```bash
# Stop the instance
apptainer instance stop falkordb
```

---

## 7. Common Issues and Solutions

### Issue: "Converting SIF file to temporary sandbox" (slow startup)

**Cause:** squashfuse not found or not working.

**Solution:** Install squashfuse as described in Section 4.

### Issue: "libsquashfuse.so.0: cannot open shared object file"

**Cause:** Library path not set correctly in squashfuse wrapper.

**Solution:** Ensure the wrapper script uses absolute paths, not `$HOME`:

```bash
# Check your actual home path
echo $HOME  # e.g., /admin/home-username

# Update wrapper with correct path
cat ~/.local/bin/squashfuse  # Verify paths are absolute
```

### Issue: "starter not found"

**Cause:** Apptainer looking in wrong location for starter binary.

**Solution:** Create symlink:

```bash
mkdir -p ~/.local/libexec/apptainer/bin
ln -sf ~/.local/apptainer/libexec/apptainer/bin/starter \
       ~/.local/libexec/apptainer/bin/starter
```

### Issue: "No space left on device" during pull

**Cause:** Cache/tmp directories on quota-limited filesystem.

**Solution:** Set `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` to large storage partition.

### Issue: FalkorDB MODULE LIST only shows "vectorset"

**Cause:** redis-server started without loading the graph module.

**Solution:** Explicitly load the module:

```bash
redis-server --loadmodule /var/lib/falkordb/bin/falkordb.so
```

### Warning: "Memory overcommit must be enabled"

**Cause:** Redis warning about `vm.overcommit_memory` setting.

**Impact:** Safe to ignore for development. For production, ask sysadmin to set `vm.overcommit_memory = 1`.

### Warning: "fuse2fs not found" / "gocryptfs not found"

**Impact:** Safe to ignore. These are only needed for:
- `fuse2fs`: EXT3 filesystem images (not SIF)
- `gocryptfs`: Encrypted containers

---

## 8. Complete ~/.bashrc Configuration

```bash
# ============================================
# Apptainer Configuration for HPC
# ============================================

# Go (if installed locally)
export PATH=~/.local/go/bin:$PATH
export GOPATH=~/.local/go

# Apptainer binary
export PATH=~/.local/apptainer/bin:$PATH

# squashfuse for fast SIF mounting
export PATH=~/.local/bin:$PATH

# Storage on large partition (CRITICAL - avoid home quota)
export APPTAINER_CACHEDIR=/weka/home-username/apptainer/cache
export APPTAINER_TMPDIR=/weka/home-username/apptainer/tmp

# Optional: Default bind mounts
# export APPTAINER_BIND="/data,/scratch"
```

---

## 9. Quick Reference

### Pull image
```bash
apptainer pull docker://image:tag
```

### Run command
```bash
apptainer exec container.sif command args
```

### Start background instance
```bash
apptainer instance start [--bind src:dst] container.sif name
```

### Execute in instance
```bash
apptainer exec instance://name command
```

### Stop instance
```bash
apptainer instance stop name
```

### List instances
```bash
apptainer instance list
```

### Clear cache
```bash
apptainer cache clean
```

---

## 10. Performance Comparison

| Configuration | Container Startup Time |
|---------------|----------------------|
| Without squashfuse | ~120 seconds (extracts to sandbox) |
| With squashfuse | ~0.1 seconds (direct FUSE mount) |

| Operation | Time |
|-----------|------|
| Pull FalkorDB image | ~60 seconds |
| Start instance | ~0.1 seconds |
| FalkorDB graph query | ~4 ms |

---

## References

- [Apptainer Documentation](https://apptainer.org/docs/)
- [Apptainer Admin Guide - Config Files](https://apptainer.org/docs/admin/main/configfiles.html)
- [squashfuse GitHub](https://github.com/vasi/squashfuse)
- [FalkorDB Documentation](https://docs.falkordb.com/)
