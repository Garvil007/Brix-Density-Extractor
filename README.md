# Brix Density Extraction System

This repository provides a Python-based system for extracting Brix density readings from images using computer vision and generative AI. It processes images of refractometers, detects key lines, and performs regression to extract accurate density values.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [main.py](#mainpy)
---

## Overview

The system leverages OpenCV for image processing and integrates Google's Generative AI for analyzing extracted regions of interest (ROI). It calculates the intersection between a red horizontal line and a vertical scale to determine the Brix density reading.

---

## Features

- Processes images of refractometers to extract key scale readings.
- Uses computer vision for line detection and clustering.
- Performs linear regression for precise scale interpretation.
- Integrates with Google's Generative AI for enhanced accuracy.
- Outputs both intermediate images and final Brix density readings.

---

## Project Structure

```plaintext
.
├── main.py                # Main script for processing images
├── requirements.txt       # Python dependencies
├── intermediate_images/   # Directory for saving intermediate results
├── README.md              # Project documentation

---

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Scikit-learn
- Pillow
- PDF2Image
- Google's Generative AI (Gemini)

---

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/brix-density-extraction.git
   cd brix-density-extraction
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**

   Create a `.env` file with your API key (if using Roboflow OCR):

   ```env
   GOOGLE_KEY=your_google_api_key
   ```

---

## Usage

### main.py

1. Place the image you want to analyze in the appropriate directory.

2. Update the file path in the __main__ section of main.py and run the script:


```bash
image_path = "path_to_your_image.jpg"
python combine.py
```

Press `q` to exit the webcam feed.


3. The processed images and results will be saved in the intermediate_images directory, and the Brix density reading will be printed to the console.

---

