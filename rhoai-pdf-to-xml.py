import os
import subprocess
import sys
import uuid

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function to install dependencies
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary dependencies
dependencies = [
    "sentence-transformers", "pymilvus", "openai==0.28", "langchain", "langchain_community", "minio",
    "pymupdf", "Pillow", "pytesseract", "pandas", "langchain-milvus", "opencv-python", "pdf2image", "vllm"
]

for dep in dependencies:
    install(dep)
    
import fitz  # PyMuPDF
import re
# Remove import openai
from minio import Minio
from minio.error import S3Error
from sentence_transformers import SentenceTransformer
#from langchain_milvus import Milvus
from langchain_community.vectorstores import Milvus
import pytesseract
from pytesseract import Output
import pandas as pd
import logging
import urllib3
from pdf2image import convert_from_path

# Import necessary modules from langchain
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# MinIO configuration
AWS_S3_ENDPOINT = "minio-api-minio.apps.cluster-lqsm2.lqsm2.sandbox441.opentlc.com"
AWS_ACCESS_KEY_ID = "minio"
AWS_SECRET_ACCESS_KEY = "minio123"
AWS_S3_BUCKET = "irpf-files"

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create an HTTP client with timeouts
http_client = urllib3.PoolManager(
    timeout=urllib3.Timeout(connect=5.0, read=10.0)
)

# Initialize the MinIO client with the custom HTTP client
client = Minio(
    AWS_S3_ENDPOINT,
    access_key=AWS_ACCESS_KEY_ID,
    secret_key=AWS_SECRET_ACCESS_KEY,
    secure=True,
    http_client=http_client
)

# Get file identifier from environment variable
#file_identifier = os.getenv('file_identifier')
file_identifier = "18_01782176543_2023_2024"
if not file_identifier:
    print("Error: file_identifier environment variable is not set.")
    sys.exit(1)

# Generate a UUID for the current execution
execution_id = str(uuid.uuid4())

# Download the PDF file from MinIO
object_name = f"{file_identifier}.pdf"
file_path = f"./{file_identifier}.pdf"
output_pdf_path = f"./{file_identifier}_no_watermark.pdf"

try:
    client.fget_object(AWS_S3_BUCKET, object_name, file_path)
    print(f"'{object_name}' successfully downloaded to '{file_path}'.")
except S3Error as e:
    print("Error occurred: ", e)
    sys.exit(1)

# Function to remove the watermark from the PDF
def remove_watermark_advanced(pdf_path, output_path):
    doc = fitz.open(pdf_path)
    for page in doc:
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            page.delete_image(xref)

        annots = page.annots()
        if annots:
            for annot in annots:
                annot_info = annot.info
                if "Watermark" in annot_info.get("title", ""):
                    annot.set_flags(fitz.ANNOT_HIDDEN)

        page.apply_redactions()

    doc.save(output_path)
    print(f"Watermark removed: {output_path}")

# Remove the watermark from the PDF
remove_watermark_advanced(file_path, output_pdf_path)

# Function to extract text from PDF
def extract_text_new(pdf_path):
    extracted_text = ''
    if os.path.exists(pdf_path):
        try:
            paginas_imagens = convert_from_path(pdf_path)
            for i, imagem_processada in enumerate(paginas_imagens):
                print(f"Processing page {i+1} of the PDF...")

                ocr_config = r'-c preserve_interword_spaces=1 --oem 1 --psm 1'
                ocr_data = pytesseract.image_to_data(imagem_processada, lang='por', config=ocr_config, output_type=Output.DICT)
                ocr_dataframe = pd.DataFrame(ocr_data)
                cleaned_df = ocr_dataframe[(ocr_dataframe.conf != '-1') & (ocr_dataframe.text != ' ') & (ocr_dataframe.text != '')]

                sorted_block_numbers = cleaned_df.groupby('block_num').first().sort_values('top').index.tolist()
                for block_num in sorted_block_numbers:
                    current_block = cleaned_df[cleaned_df['block_num'] == block_num]
                    filtered_text = current_block[current_block.text.str.len() > 3]
                    avg_char_width = (filtered_text.width / filtered_text.text.str.len()).mean()
                    prev_paragraph, prev_line, prev_left_margin = 0, 0, 0
                    
                    for idx, line_data in current_block.iterrows():
                        if prev_paragraph != line_data['par_num']:
                            extracted_text += '\n'
                            prev_paragraph = line_data['par_num']
                            prev_line = line_data['line_num']
                            prev_left_margin = 0
                        elif prev_line != line_data['line_num']:
                            extracted_text += '\n'
                            prev_line = line_data['line_num']
                            prev_left_margin = 0

                        spaces_to_add = 0
                        if line_data['left'] / avg_char_width > prev_left_margin + 1:
                            spaces_to_add = int((line_data['left']) / avg_char_width) - prev_left_margin
                            extracted_text += ' ' * spaces_to_add
                        extracted_text += line_data['text'] + ' '
                        prev_left_margin += len(line_data['text']) + spaces_to_add + 1

                    extracted_text += '\n'

            return extracted_text
        except Exception as e:
            print(f"Error processing the PDF: {e}")
            return ''
    else:
        print(f"The file {pdf_path} was not found.")
        return ''

# Milvus configurations
MILVUS_HOST = "vectordb-milvus.milvus.svc.cluster.local"
MILVUS_PORT = 19530
MILVUS_USERNAME = "root"
MILVUS_PASSWORD = "Milvus"
MILVUS_COLLECTION = "irpf"

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class EmbeddingFunctionWrapper:
    def __init__(self, model):
        self.model = model

    def embed_query(self, query):
        return self.model.encode(query)

    def embed_documents(self, documents):
        return self.model.encode(documents)

embedding_function = EmbeddingFunctionWrapper(embedding_model)

# Function to split the text
def split_text(text, max_length=60000):
    words = text.split()
    parts = []
    current_part = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 <= max_length:
            current_part.append(word)
            current_length += len(word) + 1
        else:
            parts.append(" ".join(current_part))
            current_part = [word]
            current_length = len(word) + 1
    if current_part:
        parts.append(" ".join(current_part))
    return parts

# Initialize Milvus
store = Milvus(
    embedding_function=embedding_function,
    connection_args={
        "host": MILVUS_HOST, 
        "port": MILVUS_PORT,
        "user": MILVUS_USERNAME,
        "password": MILVUS_PASSWORD,
        "timeout": 30
    },
    collection_name=MILVUS_COLLECTION,
    metadata_field="metadata",
    text_field="page_content",
    drop_old=False,
    auto_id=True
)

# Store text parts in Milvus with metadatas
def store_text_parts_in_milvus(text_parts, pdf_file, execution_id):
    for i, part in enumerate(text_parts):
        metadata = {"source": pdf_file, "part": i, "execution_id": execution_id}
        store.add_texts([part], metadatas=[metadata])

# Process the PDF and store the extracted data in Milvus
pdf_path = output_pdf_path
text = extract_text_new(pdf_path)

text_parts = split_text(text)
store_text_parts_in_milvus(text_parts, os.path.basename(pdf_path), execution_id)

print("PDF processed and stored in Milvus successfully.")

# LLM Inference Server URL
inference_server_url = "http://vllm.vllm.svc.cluster.local:8000"

# LLM definition
llm = ChatOpenAI(
    openai_api_key="EMPTY",
    openai_api_base=f"{inference_server_url}/v1",
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    top_p=0.92,
    temperature=0.01,
    max_tokens=512,
    presence_penalty=1.03,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Function to perform a query in Milvus and generate the XML
def query_information(query, execution_id, file_identifier):
    documents = store.search(
        query=query, 
        k=1, 
        search_type="similarity", 
        filter={"execution_id": execution_id, "source": f"{file_identifier}.pdf"}
    )

    context_parts = [doc.page_content for doc in documents]
    context_combined = "\n".join(context_parts)

    messages = [
        SystemMessage(content="Você é um assistente financeiro avançado, experiente com profundo conhecimento em declaração de imposto de renda. Seu único objetivo é extrair informações do contexto fornecido e gerar respostas no formato XML. NUNCA interrompa uma resposta devido ao seu tamanho. Crie o XML com TODAS as informações pertinentes à sessão de bens e direitos, respeitando TODOS seus atributos e todos seus detalhes."),
        HumanMessage(content=f"""{context_combined}\n\nQuais são todos os bens e direitos declarados no arquivo {file_identifier}? O resultado deve ser apresentado exclusivamente em XML com TODAS as características e detalhes de cada um dos bens, conforme exemplos abaixo:\n\n
<?xml version=\"1.0\" ?>\n
<SECTION Name=\"DECLARACAO DE BENS E DIREITOS\">\n
    <TABLE>\n
        <ROW No=\"1\">\n
            <Field Name=\"GRUPO\" Value=\"01\"/>\n
            <Field Name=\"CODIGO\" Value=\"01\"/>\n
            <Field Name=\"DISCRIMINACAO\" Value=\"UT QUIS ALIQUAM LEO. DONEC ALIQUA\"/>\n
            <Field Name=\"SITUACAOANTERIOR\" Value=\"23.445,00\"/>\n
            <Field Name=\"SITUACAOATUAL\" Value=\"342.342,00\"/>\n
            <Field Name=\"InscricaoMunicipal(IPTU)\" Value=\"23423424\"/>\n
            <Field Name=\"Logradouro\" Value=\"RUA QUALQUER\"/>\n
            <Field Name=\"Numero\" Value=\"89\"/>\n
            <Field Name=\"Complemento\" Value=\"COMPLEM 2\"/>\n
            <Field Name=\"Bairro\" Value=\"BRASILIA\"/>\n
            <Field Name=\"Municipio\" Value=\"BRASÍLIA\"/>\n
            <Field Name=\"UF\" Value=\"DF\"/>\n
            <Field Name=\"CEP\" Value=\"1321587\"/>\n
            <Field Name=\"AreaTotal\" Value=\"345,0 m²\"/>\n
            <Field Name=\"DatadeAquisicao\" Value=\"12/12/1993\"/>\n
            <Field Name=\"RegistradonoCartorio\" Value=\"Sim\"/>\n
            <Field Name=\"NomeCartorio\" Value=\"CARTORIO DE SÇNJJKLCDF ASLK SAKÇK SAÇKLJ SAÇLKS\"/>\n
            <Field Name=\"Matricula\" Value=\"2344234\"/>\n
        </ROW>\n
    </TABLE>\n
</SECTION>\n
Certifique-se de que todos os bens e direitos, com suas respectivas caracteristicas, estejam presentes no XML gerado, incluindo todos os valores, tais como SITUACAOANTERIOR e SITUACAOATUAL. Se uma informacao nao estiver disponivel no contexto, deixe o campo vazio. Use esses exemplo fornecidoscomo referencia ao processar as informacoes fornecidas via contexto, os próximos itens devem conter a mesma estrutra de xml, porem podem possuir campos diferentes, cada conjunto de GRUPO e CODIGO representam uma categoria de item diferente, com diferentes caracteristicas. TODAS AS CARACTERISTICAS SAO IMPORTANTES E DEVEM ESTAR PRESENTES NO XML, alguns exemplos incluem mas nao se limitam a: RENAVAM, RegistrodeEmbarcacao, RegistrodeAeronave, CPF, CNPJ, NegociadosemBolsa, Bemoudireitopertencenteao, Titular, Conta, Agencia, Banco, e etc. alguns desses itens podem estar presentes, ou não estar presentes de acordo com sua categoria, determinada pelo grupo e codigo, porem todos os itens devem possuir SITUACAOANTERIOR e SITUACAOATUAL informados no formato de data, conforme exemplo SITUACAO EM 31/12/2021 , SITUACAO EM 31/12/2022. a data da SITUACAOANTERIOR sera sempre anterior a data da SITUACAOATUAL, o que nao significa ser a data de hoje, porem mais recente que a data anterior.
"""
        )
    ]

    response = llm.predict_messages(messages)

    return response.content

# Main function to generate XML
def main():
    query = f"Quais são todos os bens e direitos suas informações, seus atributos e detalhes, declarados no arquivo {file_identifier}? Não interrompa a resposta devido ao seu tamanho, forneça a resposta com todo o contexto que tiver. Valores monetarios devem estar em portugues do brasil, moeda real brl."
    result = query_information(query, execution_id, file_identifier)

    if result.startswith("```xml"):
        result = result[6:]  # Remove "```xml"
    if result.endswith("```"):
        result = result[:-3]  # Remove "```"

    result = result.strip()

    file_name = f'./{file_identifier}.xml'
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(result)

    print(f"File {file_name} has been successfully saved.")

    # Upload XML to MinIO
    object_name = f"{file_identifier}.xml"
    bucket_name = 'irpf-xml'
    try:
        client.fput_object(bucket_name, object_name, file_name)
        print(f"'{object_name}' successfully uploaded to '{bucket_name}'.")
    except S3Error as e:
        print("Error occurred: ", e)

# Execute the script
if __name__ == "__main__":
    main()
import os
import subprocess
import sys
import uuid

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function to install dependencies
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary dependencies
dependencies = [
    "sentence-transformers", "pymilvus", "openai==0.28", "langchain", "langchain_community", "minio",
    "pymupdf", "Pillow", "pytesseract", "pandas", "langchain-milvus", "opencv-python", "pdf2image", "vllm"
]

for dep in dependencies:
    install(dep)
    
import fitz  # PyMuPDF
import re
# Remove import openai
from minio import Minio
from minio.error import S3Error
from sentence_transformers import SentenceTransformer
#from langchain_milvus import Milvus
from langchain_community.vectorstores import Milvus
import pytesseract
from pytesseract import Output
import pandas as pd
import logging
import urllib3
from pdf2image import convert_from_path

# Import necessary modules from langchain
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# MinIO configuration
AWS_S3_ENDPOINT = "minio-api-minio.apps.cluster-lqsm2.lqsm2.sandbox441.opentlc.com"
AWS_ACCESS_KEY_ID = "minio"
AWS_SECRET_ACCESS_KEY = "minio123"
AWS_S3_BUCKET = "irpf-files"

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create an HTTP client with timeouts
http_client = urllib3.PoolManager(
    timeout=urllib3.Timeout(connect=5.0, read=10.0)
)

# Initialize the MinIO client with the custom HTTP client
client = Minio(
    AWS_S3_ENDPOINT,
    access_key=AWS_ACCESS_KEY_ID,
    secret_key=AWS_SECRET_ACCESS_KEY,
    secure=True,
    http_client=http_client
)

# Get file identifier from environment variable
#file_identifier = os.getenv('file_identifier')
file_identifier = "18_01782176543_2023_2024"
if not file_identifier:
    print("Error: file_identifier environment variable is not set.")
    sys.exit(1)

# Generate a UUID for the current execution
execution_id = str(uuid.uuid4())

# Download the PDF file from MinIO
object_name = f"{file_identifier}.pdf"
file_path = f"./{file_identifier}.pdf"
output_pdf_path = f"./{file_identifier}_no_watermark.pdf"

try:
    client.fget_object(AWS_S3_BUCKET, object_name, file_path)
    print(f"'{object_name}' successfully downloaded to '{file_path}'.")
except S3Error as e:
    print("Error occurred: ", e)
    sys.exit(1)

# Function to remove the watermark from the PDF
def remove_watermark_advanced(pdf_path, output_path):
    doc = fitz.open(pdf_path)
    for page in doc:
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            page.delete_image(xref)

        annots = page.annots()
        if annots:
            for annot in annots:
                annot_info = annot.info
                if "Watermark" in annot_info.get("title", ""):
                    annot.set_flags(fitz.ANNOT_HIDDEN)

        page.apply_redactions()

    doc.save(output_path)
    print(f"Watermark removed: {output_path}")

# Remove the watermark from the PDF
remove_watermark_advanced(file_path, output_pdf_path)

# Function to extract text from PDF
def extract_text_new(pdf_path):
    extracted_text = ''
    if os.path.exists(pdf_path):
        try:
            paginas_imagens = convert_from_path(pdf_path)
            for i, imagem_processada in enumerate(paginas_imagens):
                print(f"Processing page {i+1} of the PDF...")

                ocr_config = r'-c preserve_interword_spaces=1 --oem 1 --psm 1'
                ocr_data = pytesseract.image_to_data(imagem_processada, lang='por', config=ocr_config, output_type=Output.DICT)
                ocr_dataframe = pd.DataFrame(ocr_data)
                cleaned_df = ocr_dataframe[(ocr_dataframe.conf != '-1') & (ocr_dataframe.text != ' ') & (ocr_dataframe.text != '')]

                sorted_block_numbers = cleaned_df.groupby('block_num').first().sort_values('top').index.tolist()
                for block_num in sorted_block_numbers:
                    current_block = cleaned_df[cleaned_df['block_num'] == block_num]
                    filtered_text = current_block[current_block.text.str.len() > 3]
                    avg_char_width = (filtered_text.width / filtered_text.text.str.len()).mean()
                    prev_paragraph, prev_line, prev_left_margin = 0, 0, 0
                    
                    for idx, line_data in current_block.iterrows():
                        if prev_paragraph != line_data['par_num']:
                            extracted_text += '\n'
                            prev_paragraph = line_data['par_num']
                            prev_line = line_data['line_num']
                            prev_left_margin = 0
                        elif prev_line != line_data['line_num']:
                            extracted_text += '\n'
                            prev_line = line_data['line_num']
                            prev_left_margin = 0

                        spaces_to_add = 0
                        if line_data['left'] / avg_char_width > prev_left_margin + 1:
                            spaces_to_add = int((line_data['left']) / avg_char_width) - prev_left_margin
                            extracted_text += ' ' * spaces_to_add
                        extracted_text += line_data['text'] + ' '
                        prev_left_margin += len(line_data['text']) + spaces_to_add + 1

                    extracted_text += '\n'

            return extracted_text
        except Exception as e:
            print(f"Error processing the PDF: {e}")
            return ''
    else:
        print(f"The file {pdf_path} was not found.")
        return ''

# Milvus configurations
MILVUS_HOST = "vectordb-milvus.milvus.svc.cluster.local"
MILVUS_PORT = 19530
MILVUS_USERNAME = "root"
MILVUS_PASSWORD = "Milvus"
MILVUS_COLLECTION = "irpf"

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class EmbeddingFunctionWrapper:
    def __init__(self, model):
        self.model = model

    def embed_query(self, query):
        return self.model.encode(query)

    def embed_documents(self, documents):
        return self.model.encode(documents)

embedding_function = EmbeddingFunctionWrapper(embedding_model)

# Function to split the text
def split_text(text, max_length=60000):
    words = text.split()
    parts = []
    current_part = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 <= max_length:
            current_part.append(word)
            current_length += len(word) + 1
        else:
            parts.append(" ".join(current_part))
            current_part = [word]
            current_length = len(word) + 1
    if current_part:
        parts.append(" ".join(current_part))
    return parts

# Initialize Milvus
store = Milvus(
    embedding_function=embedding_function,
    connection_args={
        "host": MILVUS_HOST, 
        "port": MILVUS_PORT,
        "user": MILVUS_USERNAME,
        "password": MILVUS_PASSWORD,
        "timeout": 30
    },
    collection_name=MILVUS_COLLECTION,
    metadata_field="metadata",
    text_field="page_content",
    drop_old=False,
    auto_id=True
)

# Store text parts in Milvus with metadatas
def store_text_parts_in_milvus(text_parts, pdf_file, execution_id):
    for i, part in enumerate(text_parts):
        metadata = {"source": pdf_file, "part": i, "execution_id": execution_id}
        store.add_texts([part], metadatas=[metadata])

# Process the PDF and store the extracted data in Milvus
pdf_path = output_pdf_path
text = extract_text_new(pdf_path)

text_parts = split_text(text)
store_text_parts_in_milvus(text_parts, os.path.basename(pdf_path), execution_id)

print("PDF processed and stored in Milvus successfully.")

# LLM Inference Server URL
inference_server_url = "http://vllm.vllm.svc.cluster.local:8000"

# LLM definition
llm = ChatOpenAI(
    openai_api_key="EMPTY",
    openai_api_base=f"{inference_server_url}/v1",
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    top_p=0.92,
    temperature=0.01,
    max_tokens=512,
    presence_penalty=1.03,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Function to perform a query in Milvus and generate the XML
def query_information(query, execution_id, file_identifier):
    documents = store.search(
        query=query, 
        k=1, 
        search_type="similarity", 
        filter={"execution_id": execution_id, "source": f"{file_identifier}.pdf"}
    )

    context_parts = [doc.page_content for doc in documents]
    context_combined = "\n".join(context_parts)

    messages = [
        SystemMessage(content="Você é um assistente financeiro avançado, experiente com profundo conhecimento em declaração de imposto de renda. Seu único objetivo é extrair informações do contexto fornecido e gerar respostas no formato XML. NUNCA interrompa uma resposta devido ao seu tamanho. Crie o XML com TODAS as informações pertinentes à sessão de bens e direitos, respeitando TODOS seus atributos e todos seus detalhes."),
        HumanMessage(content=f"""{context_combined}\n\nQuais são todos os bens e direitos declarados no arquivo {file_identifier}? O resultado deve ser apresentado exclusivamente em XML com TODAS as características e detalhes de cada um dos bens, conforme exemplos abaixo:\n\n
<?xml version=\"1.0\" ?>\n
<SECTION Name=\"DECLARACAO DE BENS E DIREITOS\">\n
    <TABLE>\n
        <ROW No=\"1\">\n
            <Field Name=\"GRUPO\" Value=\"01\"/>\n
            <Field Name=\"CODIGO\" Value=\"01\"/>\n
            <Field Name=\"DISCRIMINACAO\" Value=\"UT QUIS ALIQUAM LEO. DONEC ALIQUA\"/>\n
            <Field Name=\"SITUACAOANTERIOR\" Value=\"23.445,00\"/>\n
            <Field Name=\"SITUACAOATUAL\" Value=\"342.342,00\"/>\n
            <Field Name=\"InscricaoMunicipal(IPTU)\" Value=\"23423424\"/>\n
            <Field Name=\"Logradouro\" Value=\"RUA QUALQUER\"/>\n
            <Field Name=\"Numero\" Value=\"89\"/>\n
            <Field Name=\"Complemento\" Value=\"COMPLEM 2\"/>\n
            <Field Name=\"Bairro\" Value=\"BRASILIA\"/>\n
            <Field Name=\"Municipio\" Value=\"BRASÍLIA\"/>\n
            <Field Name=\"UF\" Value=\"DF\"/>\n
            <Field Name=\"CEP\" Value=\"1321587\"/>\n
            <Field Name=\"AreaTotal\" Value=\"345,0 m²\"/>\n
            <Field Name=\"DatadeAquisicao\" Value=\"12/12/1993\"/>\n
            <Field Name=\"RegistradonoCartorio\" Value=\"Sim\"/>\n
            <Field Name=\"NomeCartorio\" Value=\"CARTORIO DE SÇNJJKLCDF ASLK SAKÇK SAÇKLJ SAÇLKS\"/>\n
            <Field Name=\"Matricula\" Value=\"2344234\"/>\n
        </ROW>\n
    </TABLE>\n
</SECTION>\n
Certifique-se de que todos os bens e direitos, com suas respectivas caracteristicas, estejam presentes no XML gerado, incluindo todos os valores, tais como SITUACAOANTERIOR e SITUACAOATUAL. Se uma informacao nao estiver disponivel no contexto, deixe o campo vazio. Use esses exemplo fornecidoscomo referencia ao processar as informacoes fornecidas via contexto, os próximos itens devem conter a mesma estrutra de xml, porem podem possuir campos diferentes, cada conjunto de GRUPO e CODIGO representam uma categoria de item diferente, com diferentes caracteristicas. TODAS AS CARACTERISTICAS SAO IMPORTANTES E DEVEM ESTAR PRESENTES NO XML, alguns exemplos incluem mas nao se limitam a: RENAVAM, RegistrodeEmbarcacao, RegistrodeAeronave, CPF, CNPJ, NegociadosemBolsa, Bemoudireitopertencenteao, Titular, Conta, Agencia, Banco, e etc. alguns desses itens podem estar presentes, ou não estar presentes de acordo com sua categoria, determinada pelo grupo e codigo, porem todos os itens devem possuir SITUACAOANTERIOR e SITUACAOATUAL informados no formato de data, conforme exemplo SITUACAO EM 31/12/2021 , SITUACAO EM 31/12/2022. a data da SITUACAOANTERIOR sera sempre anterior a data da SITUACAOATUAL, o que nao significa ser a data de hoje, porem mais recente que a data anterior.
"""
        )
    ]

    response = llm.predict_messages(messages)

    return response.content

# Main function to generate XML
def main():
    query = f"Quais são todos os bens e direitos suas informações, seus atributos e detalhes, declarados no arquivo {file_identifier}? Não interrompa a resposta devido ao seu tamanho, forneça a resposta com todo o contexto que tiver. Valores monetarios devem estar em portugues do brasil, moeda real brl."
    result = query_information(query, execution_id, file_identifier)

    if result.startswith("```xml"):
        result = result[6:]  # Remove "```xml"
    if result.endswith("```"):
        result = result[:-3]  # Remove "```"

    result = result.strip()

    file_name = f'./{file_identifier}.xml'
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(result)

    print(f"File {file_name} has been successfully saved.")

    # Upload XML to MinIO
    object_name = f"{file_identifier}.xml"
    bucket_name = 'irpf-xml'
    try:
        client.fput_object(bucket_name, object_name, file_name)
        print(f"'{object_name}' successfully uploaded to '{bucket_name}'.")
    except S3Error as e:
        print("Error occurred: ", e)

# Execute the script
if __name__ == "__main__":
    main()