# Base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Application Config
ENV SHOW_LIVE=False
ENV PLATE_CONF_MIN=0.65
ENV VEHICLE_CONF_MIN=0.5
ENV VIDEO_SOURCE=requirements/input_5.mkv
ENV RESIZE_FACTOR=1.5
ENV TIME_FORMAT=%Y%m%d_%H%M%S.%f
ENV OUTPUT_DIR=output_dir
ENV FRONTEND_DIR=frontend_dir
ENV VEHICLE_CLASSES=2,3,5,7
ENV WATCHDOG_INTERVAL=5
ENV HTML_HEADERS="S. No.,Object ID,Time Stamp,HSRP Detected,HSRP File,Processed File,Confidence,HSRP Output"
ENV VEHICLE_MODEL_PATH=requirements/yolo12n.pt
ENV PLATE_MODEL_PATH=requirements/best28.pt
ENV SENDER_EMAIL=systemalert@mail.dccmail.in
# Note: TO_EMAILS has multiple values, handle carefully if logic splits by comma
ENV TO_EMAILS=user.harpreetsingh@gmail.com
ENV CC_EMAILS=""
ENV PASSWORD=ZUqp2215@%%
ENV SUBJECT="License Plate detected - Incoming traffic - Gate 6"
ENV BODY1="Please check http://192.168.150.57/lpr.html for all results and http://192.168.150.57/output_dir/"
ENV BODY2=" for yesterday's results. This email confirms that yesterday's file has been successfully uploaded to the server. Kindly do not reply to this email, as it is system-generated."
ENV SMTP_HOST=mail.dccmail.in
ENV SMTP_PORT=587
ENV JOB_TIME=14:08

# Set working directory
WORKDIR /app

# Install system dependencies
# libgl1 and libglib2.0-0 are often required by OpenCV and other CV libs, even headless sometimes.
# libgomp1 is required by PyTorch (used by Ultralytics) and PaddlePaddle.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create output and logs directories if they don't exist
RUN mkdir -p output_dir logs data

# Command to run the application
CMD ["python", "main.py"]
