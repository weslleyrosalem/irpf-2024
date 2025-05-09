# IRPF 2024 / 2025

This repository contains tools and scripts for the automated processing of PDFs related to the **2024 Personal Income Tax (IRPF)** in Brazil. The main goal is to facilitate data extraction and conversion from these PDFs using OCR techniques and data processing pipelines.

## 📁 Repository Structure

Below is a description of the main directories and files in the repository.

### 📂 Directories

- **`milvus/`** - Contains scripts and configuration files for integration with the **Milvus vector database**, used for efficient storage and retrieval of feature vectors extracted from documents.
- **`pdfs/`** - Directory for storing the **IRPF PDF files** to be processed.
- **`runtimes/`** - Includes scripts and configuration files related to the **execution environments** required for the processing pipelines.
- **`vector-db/`** - Contains scripts and settings for configuring and managing the **vector database** used in this project.

### 📄 Files

- **`18_01782176543_2023_2024.pdf`** - Sample IRPF PDF file for the **2023–2024** period, used for testing and development purposes.
- **`Dockerfile`** - Configuration file used to build a **Docker image** that defines the required environment to run the project’s scripts and pipelines.
- **`README.md`** - This file, providing an overview of the repository and usage instructions.
- **`app.py`** - Main application script that **coordinates the PDF processing flow**, including upload, data extraction, and storage steps.
- **`entrypoint.sh`** - Entry script used to **initialize the environment** and run the necessary services or scripts when the Docker container starts.
- **`kafka_consumer.py`** - Script responsible for consuming messages from a **Kafka topic**, handling events related to PDF uploads and triggering the appropriate pipelines.
- **`rhoai-pdf-to-xml.py`** - Script that defines the **PDF-to-XML conversion pipeline**, leveraging OCR tools and other text processing techniques.
- **`rhoai-pdf-to-xml.yaml`** - **YAML configuration file** describing the PDF-to-XML pipeline, including its steps, parameters, and dependencies.
- **`sqs_consumer.py`** - Script that consumes messages from an **Amazon SQS queue**. It monitors the queue for PDF upload events and, upon receiving a message, processes the event and triggers the corresponding PDF-to-XML conversion pipeline. This script serves as an alternative to `kafka_consumer.py`, tailored for environments using **Amazon SQS** instead of **Kafka**.
- **`tesseract-saturno.ipynb`** - Jupyter Notebook that explores the use of **Tesseract OCR** for text extraction from IRPF PDFs, possibly including experiments and parameter tuning to improve **character recognition accuracy**.

## 🚀 How to Use

### 1️⃣ Environment Setup

Use the `Dockerfile` to build the **Docker image** containing all required dependencies:

```bash
docker build -t irpf-2024 .
