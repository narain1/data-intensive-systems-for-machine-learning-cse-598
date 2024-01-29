# Use the TensorFlow 2.5.1 base image
FROM tensorflow/tensorflow:2.5.1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app/

# Command to run your Python script
CMD ["python", "a1b.py"]

