# 1. Base Image
# Use a Python version that matches .python-version and pyproject.toml
FROM python:3.10-slim

# 2. Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# PORT is usually set by Hugging Face Spaces environment. Default to 7860 if not set.
ENV PORT 7860

# 3. Install System Dependencies
# - git: for some pip packages or potential model fetching
# - build-essential: for packages that might need compilation
# - libgl1-mesa-glx, libglib2.0-0: for headless OpenCV (dependency of deepface)
# - curl: to download uv
RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  build-essential \
  libgl1-mesa-glx \
  libglib2.0-0 \
  curl

RUN rm -rf /var/lib/apt/lists/*

# 4. Install uv (for package management, as uv.lock is present)
# Pin uv version for reproducibility, e.g. 0.2.30 or use latest
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# 5. Set Working Directory
WORKDIR /app

# 6. Copy Dependency Files
# Copy pyproject.toml and uv.lock to leverage Docker cache for dependencies
COPY pyproject.toml uv.lock ./

# 7. Install Python Dependencies using uv
# This will use pyproject.toml and uv.lock to install the project and its dependencies
# --system flag tells uv to install into the system Python environment
# --no-cache to avoid caching issues, uv manages its own cache differently from pip.
RUN uv pip install --system --no-cache .

# 8. Copy Application Code
# Copy the rest of the application files
# This includes main.py, run_hf.py, templates/, etc.
COPY . .

# 9. Create Upload Folder (main.py also does this, but good for Docker layer & permissions)
# Ensure the app user (default root in this image) can write to it.
RUN mkdir -p /app/temp_uploads && chmod -R 777 /app/temp_uploads

# 10. Expose Port
# The run_hf.py script will use the PORT environment variable, defaulting to 7860.
EXPOSE ${PORT}

# 11. Command to Run Application
# Use run_hf.py as it's designed for HF Spaces and handles port binding correctly.
CMD ["python", "run_hf.py"]
