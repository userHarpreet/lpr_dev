# Dependency Conflict Analysis and Resolution

## 🚨 Primary Conflict Identified

**Error:**
```
ERROR: Cannot install -r requirements.txt (line 55) and mpmath==1.4.0a4 because these package versions have conflicting dependencies.

The conflict is caused by:
    The user requested mpmath==1.4.0a4
    sympy 1.13.3 depends on mpmath<1.4 and >=1.1.0
```

## 📊 Root Cause Analysis

### Main Issues:

1. **Version Pinning Too Restrictive**: The original `requirements.txt` pins exact versions, which creates rigid dependency chains that can't be resolved.

2. **Alpha/Beta Versions**: Using `mpmath==1.4.0a4` (alpha version) conflicts with stable dependency requirements from other packages.

3. **Outdated Package Combinations**: Some pinned versions are from different time periods and weren't tested together.

### Specific Conflicts Found:

| Package | Original Version | Conflict With | Required By | Solution |
|---------|------------------|---------------|-------------|----------|
| `mpmath` | `1.4.0a4` | `sympy 1.13.3` | sympy needs `<1.4,>=1.1.0` | Use `mpmath>=1.3.0,<1.4.0` |
| `numpy` | `2.1.1` | Various ML packages | Some may not support 2.x yet | Use `numpy>=1.24.0,<2.0.0` |
| `typing_extensions` | `4.13.0rc1` | Core packages | RC versions can be unstable | Use stable `>=4.8.0,<5.0.0` |
| `shapely` | `2.1.0rc1` | Geospatial packages | RC version conflicts | Remove if not needed |

## 🔧 Solutions Implemented

### 1. **requirements.txt (Updated)**
- **Strategy**: Flexible version ranges with upper bounds
- **Benefits**: Allows pip to resolve compatible versions
- **Use Case**: Production deployments where you want control

### 2. **requirements-minimal.txt (New)**
- **Strategy**: Core packages only, let pip resolve all dependencies
- **Benefits**: Simplest, most flexible approach
- **Use Case**: Development, testing, or when you trust pip's resolver

## 📦 Package Categories and Recommendations

### Essential for LPR Functionality:
```
paddleocr          # OCR engine
paddlepaddle       # Backend for PaddleOCR  
ultralytics        # YOLO models
opencv-python      # Computer vision
torch/torchvision  # ML framework
numpy              # Numerical computing
```

### Supporting Packages:
```
pillow             # Image processing
requests           # HTTP requests
PyYAML             # Configuration files
psutil             # System monitoring
schedule           # Task scheduling
```

### Optional/Removable:
```
matplotlib         # Only needed for debugging plots
seaborn           # Statistical plots
scipy             # Advanced math functions
pandas            # Data manipulation
sympy             # Symbolic math
tensorboard-*     # ML training visualization
```

## 🏗️ Build Strategy Options

### Option 1: Use Minimal Requirements (Recommended)
```dockerfile
COPY requirements-minimal.txt .
RUN pip3 install --no-cache-dir -r requirements-minimal.txt
```

**Pros:**
- Fastest build time
- Fewest conflicts
- Automatic dependency resolution

**Cons:**
- Less control over exact versions
- Might install newer versions than tested

### Option 2: Use Updated Requirements with Ranges
```dockerfile
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
```

**Pros:**
- More control over versions
- Predictable builds
- Better for production

**Cons:**
- Longer to resolve
- May need periodic updates

### Option 3: Multi-stage Build with Conflict Resolution
```dockerfile
# Build stage - install and resolve dependencies
FROM ubuntu:22.04 as builder
COPY requirements-minimal.txt .
RUN pip3 install --user -r requirements-minimal.txt

# Production stage - copy resolved packages
FROM ubuntu:22.04
COPY --from=builder /root/.local /usr/local
```

## 🔍 Testing Your Dependencies

### 1. Local Testing:
```bash
# Test minimal requirements
pip install -r requirements-minimal.txt

# Test updated requirements  
pip install -r requirements.txt

# Check for conflicts
pip check
```

### 2. Virtual Environment Testing:
```bash
python -m venv test_env
source test_env/bin/activate  # Linux/Mac
# or
test_env\Scripts\activate     # Windows

pip install -r requirements-minimal.txt
python -c "import paddleocr, ultralytics, cv2; print('All imports successful')"
```

### 3. Docker Testing:
```bash
# Test the minimal build
docker build -f Dockerfile.minimal -t lpr-minimal .
docker run --rm lpr-minimal python3 -c "import paddleocr; print('OK')"
```

## 📋 Recommended Next Steps

### Immediate Actions:
1. **Try Minimal Requirements First**:
   - Update Dockerfile to use `requirements-minimal.txt`
   - Test the build: `.\build.ps1 build`

2. **Test Core Functionality**:
   - Verify OCR works: `docker run --rm lpr-app python3 -c "from paddleocr import PaddleOCR; print('OCR OK')"`
   - Verify YOLO works: `docker run --rm lpr-app python3 -c "from ultralytics import YOLO; print('YOLO OK')"`

3. **Add Missing Dependencies**:
   - If application fails, add specific missing packages to minimal requirements
   - Avoid version pinning unless absolutely necessary

### Long-term Strategy:
1. **Regular Updates**: Update dependencies monthly
2. **Testing Pipeline**: Set up automated testing of dependency changes
3. **Version Monitoring**: Use tools like `pip-audit` for security updates
4. **Documentation**: Document which specific versions are tested and working

## 🐛 Common Issues and Solutions

### Issue: "Package not found"
**Solution**: Check package name spelling, some packages have changed names (e.g., `Pillow` vs `PIL`)

### Issue: "Version conflict during runtime"
**Solution**: Use `pip freeze > working-requirements.txt` after successful installation to lock working versions

### Issue: "Import errors in container"
**Solution**: 
- Check system dependencies in Dockerfile
- Verify Python path in container
- Use `pip show <package>` to verify installation

### Issue: "Out of memory during build"
**Solution**:
- Use multi-stage builds
- Increase Docker memory limits
- Build on machine with more RAM

## 📝 Template Dockerfile for Minimal Build

```dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Use minimal requirements for faster, more reliable builds
COPY requirements-minimal.txt .
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements-minimal.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p output_dir debug_plates

# Security: non-root user
RUN useradd -m lpruser && chown -R lpruser:lpruser /app
USER lpruser

CMD ["python3", "main.py"]
```

This approach prioritizes working functionality over exact version control, which is often the best strategy for containerized applications.
