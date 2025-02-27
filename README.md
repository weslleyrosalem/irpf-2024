# IRPF 2024

Este repositório contém ferramentas e scripts para o processamento automatizado de PDFs relacionados ao **Imposto de Renda Pessoa Física (IRPF) de 2024**. O objetivo principal é facilitar a extração e conversão de dados desses PDFs utilizando técnicas de OCR e pipelines de processamento de dados.

## 📁 Estrutura do Repositório

Abaixo está a descrição dos principais diretórios e arquivos presentes no repositório.

### 📂 Diretórios

- **`milvus/`** - Contém scripts e configurações para integração com o banco de dados vetorial **Milvus**, utilizado para armazenamento e busca eficiente de vetores de características extraídas dos documentos.
- **`pdfs/`** - Diretório destinado ao armazenamento dos arquivos **PDF do IRPF** que serão processados.
- **`runtimes/`** - Inclui scripts e configurações relacionadas aos **ambientes de execução** necessários para os pipelines de processamento.
- **`vector-db/`** - Contém scripts e configurações para a configuração e gerenciamento do **banco de dados vetorial** utilizado no projeto.

### 📄 Arquivos

- **`18_01782176543_2023_2024.pdf`** - Exemplo de um arquivo PDF do IRPF referente ao período de **2023-2024**, utilizado para testes e desenvolvimento.
- **`Dockerfile`** - Arquivo de configuração para a criação de uma **imagem Docker** que define o ambiente necessário para a execução dos scripts e pipelines do projeto.
- **`README.md`** - Este arquivo, fornecendo uma visão geral do repositório e instruções de uso.
- **`app.py`** - Script principal da aplicação que **coordena o fluxo de processamento dos PDFs**, incluindo etapas de upload, extração de dados e armazenamento.
- **`entrypoint.sh`** - Script de entrada utilizado para **inicializar o ambiente** e executar os serviços ou scripts necessários quando o contêiner Docker é iniciado.
- **`kafka_consumer.py`** - Script responsável por consumir mensagens de um **tópico Kafka**, processando eventos relacionados ao upload de PDFs e iniciando os pipelines correspondentes.
- **`rhoai-pdf-to-xml.py`** - Script que define o pipeline de **conversão de PDFs para XML**, utilizando ferramentas de OCR e outras técnicas de processamento de texto.
- **`rhoai-pdf-to-xml.yaml`** - Arquivo de configuração em **YAML** que descreve o pipeline de conversão de PDFs para XML, incluindo as etapas, parâmetros e dependências necessárias.
- **`sqs_consumer.py`** - Script responsável por consumir mensagens de uma **fila do Amazon SQS**. Ele monitora a fila para detectar eventos relacionados ao upload de PDFs e, ao receber uma mensagem, processa o evento e inicia o pipeline correspondente para conversão do PDF em XML. Este script é uma alternativa ao `kafka_consumer.py`, adaptado para ambientes que utilizam o **Amazon SQS** em vez do **Kafka**.
- **`tesseract-saturno.ipynb`** - Notebook Jupyter que explora o uso do **Tesseract OCR** para extração de texto dos PDFs do IRPF, possivelmente incluindo experimentações e ajustes de parâmetros para melhorar a **acurácia do reconhecimento de caracteres**.

## 🚀 Como Utilizar

### 1️⃣ Configuração do Ambiente

Utilize o `Dockerfile` para construir a **imagem Docker** que contém todas as dependências necessárias:

```bash
docker build -t irpf-2024 .
