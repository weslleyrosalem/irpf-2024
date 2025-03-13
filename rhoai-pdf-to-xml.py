import os
import subprocess
import sys
import uuid

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------
# 1) Instalação das dependências
# ---------------------------------------------
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Descomente se precisar forçar a instalação
# dependencies = [
#     "sentence-transformers", "pymilvus", "openai==0.28", "langchain",
#     "langchain_community", "minio", "pymupdf", "Pillow",
#     "pytesseract", "pandas", "langchain-milvus",
#     "opencv-python", "pdf2image", "vllm"
# ]
# for dep in dependencies:
#     install(dep)

import fitz  # PyMuPDF
import re
from minio import Minio
from minio.error import S3Error
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Milvus
import pytesseract
from pytesseract import Output
import pandas as pd
import logging
import urllib3
from pdf2image import convert_from_path

# Import necessário do langchain
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# ---------------------------------------------
# 2) Configurações do MinIO
# ---------------------------------------------
AWS_S3_ENDPOINT = "s3.sa-east-1.amazonaws.com"
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_S3_BUCKET = "ai-bucket-rosa-hml"

logging.basicConfig(level=logging.DEBUG)

# Criação do http client com timeouts
http_client = urllib3.PoolManager(
    timeout=urllib3.Timeout(connect=5.0, read=10.0)
)

# Inicializa o cliente MinIO
client = Minio(
    AWS_S3_ENDPOINT,
    access_key=AWS_ACCESS_KEY_ID,
    secret_key=AWS_SECRET_ACCESS_KEY,
    secure=True,
    http_client=http_client
)

file_identifier = "joao-2023"
if not file_identifier:
    print("Error: file_identifier environment variable is not set.")
    sys.exit(1)

execution_id = str(uuid.uuid4())

object_name = f"{file_identifier}.pdf"
file_path = f"./{file_identifier}.pdf"
output_pdf_path = f"./{file_identifier}_no_watermark.pdf"

# ---------------------------------------------
# 3) Download do PDF no local
# ---------------------------------------------
try:
    client.fget_object(AWS_S3_BUCKET, object_name, file_path)
    print(f"'{object_name}' successfully downloaded to '{file_path}'.")
except S3Error as e:
    print("Error occurred: ", e)
    sys.exit(1)

# ---------------------------------------------
# 4) Função de remoção de watermark
# ---------------------------------------------
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

remove_watermark_advanced(file_path, output_pdf_path)

# ---------------------------------------------
# 5) Extração básica de texto via PyMuPDF + OCR
# ---------------------------------------------
def extract_text_new(pdf_path):
    extracted_text = ''
    if os.path.exists(pdf_path):
        try:
            paginas_imagens = convert_from_path(pdf_path)
            for i, imagem_processada in enumerate(paginas_imagens):
                print(f"Processing page {i+1} of the PDF...")

                # Pode ajustar a config do OCR caso queira
                ocr_config = r'-c preserve_interword_spaces=1 --oem 1 --psm 1'
                ocr_data = pytesseract.image_to_data(imagem_processada, lang='por',
                                                     config=ocr_config,
                                                     output_type=Output.DICT)
                ocr_dataframe = pd.DataFrame(ocr_data)
                cleaned_df = ocr_dataframe[
                    (ocr_dataframe.conf != '-1') &
                    (ocr_dataframe.text != ' ') &
                    (ocr_dataframe.text != '')
                ]

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

text = extract_text_new(output_pdf_path)

# ---------------------------------------------------------------
# 6) (Opcional) Armazenar no Milvus, mas iremos filtrar manualmente
# ---------------------------------------------------------------
MILVUS_HOST = "vectordb-milvus.milvus.svc.cluster.local"
MILVUS_PORT = 19530
MILVUS_USERNAME = "root"
MILVUS_PASSWORD = "Milvus"
MILVUS_COLLECTION = "irpf"

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class EmbeddingFunctionWrapper:
    def __init__(self, model):
        self.model = model

    def embed_query(self, query):
        return self.model.encode(query)

    def embed_documents(self, documents):
        return self.model.encode(documents)

embedding_function = EmbeddingFunctionWrapper(embedding_model)

# Inicializa Milvus
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

# Exemplo de split e store (caso mantenha Milvus)
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

def store_text_parts_in_milvus(text_parts, pdf_file, execution_id):
    for i, part in enumerate(text_parts):
        metadata = {"source": pdf_file, "part": i, "execution_id": execution_id}
        store.add_texts([part], metadatas=[metadata])

text_parts = split_text(text)
store_text_parts_in_milvus(text_parts, os.path.basename(output_pdf_path), execution_id)

print("PDF processed and stored in Milvus successfully.")

# ----------------------------------------------------------------
# 7) LLM com streaming
# ----------------------------------------------------------------
inference_server_url = "http://llama-33-70b.vllm.svc.cluster.local:8080"

llm = ChatOpenAI(
    openai_api_key="EMPTY",
    openai_api_base=f"{inference_server_url}/v1",
    model_name="llama-33-70b",
    top_p=0.92,
    temperature=0.01,
    max_tokens=32768,
    presence_penalty=1.03,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# ----------------------------------------------------------------
# 8) Filtra somente a parte de Bens e Direitos do texto
# ----------------------------------------------------------------
# A ideia: localiza no texto extraído a seção "DECLARAÇÃO DE BENS E DIREITOS"
# e as linhas subsequentes, até chegar em outra seção não relacionada
# Este regex é um exemplo, ajuste conforme a formatação do PDF

def extract_bens_e_direitos(full_text: str) -> str:
    # Regex que pega desde "DECLARAÇÃO DE BENS E DIREITOS" até "DÍVIDAS E ÔNUS REAIS"
    # ou "DEMONSTRATIVO DE" ou algo do tipo. Ajuste se necessário.
    pattern = r"(DECLARAÇÃO DE BENS E DIREITOS.*?)(?:DÍVIDAS E ÔNUS REAIS|DEMOSNTRATIVO|PÁGINA \d+ de \d+|TOTAL\b|DOAÇÕES A PARTIDOS|FIM_DA_SECAO)"
    match = re.search(pattern, full_text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

filtered_bens_direitos_text = extract_bens_e_direitos(text)

# Se ficar vazio, significa que não achou. Em muitos PDFs da Receita,
# "Declaração de Bens e Direitos" se repete várias vezes, ou só tem "BENS E DIREITOS"
# Ajuste acima conforme o texto real.

# ----------------------------------------------------------------
# 9) Consulta de fato: mas agora mandamos só a parte relevante
# ----------------------------------------------------------------
def query_information(bens_text, file_identifier):
    if not bens_text.strip():
        # se está vazio, devolve aviso
        return "Não localizei seção de Bens e Direitos no PDF."

    # Montamos as mensagens
    system_content = (
        "Você é um assistente financeiro avançado, experiente com profundo "
        "conhecimento em declaração de imposto de renda. Seu único objetivo é "
        "extrair informações do texto fornecido e gerar respostas no formato XML. "
        "NUNCA interrompa a resposta devido ao seu tamanho. Crie o XML com TODAS "
        "as informações pertinentes à sessão de bens e direitos, respeitando TODOS "
        "seus atributos e detalhes."
    )

    user_prompt = (
        f"""{bens_text}\n\n
Quais são todos os bens e direitos declarados no arquivo {file_identifier}? 
O resultado deve ser apresentado exclusivamente em XML com TODAS as características 
e detalhes de cada um dos bens, conforme exemplos abaixo:

<?xml version="1.0" ?>
<SECTION Name="DECLARACAO DE BENS E DIREITOS">
    <TABLE>
        <ROW No="1">
            <Field Name="GRUPO" Value="01"/>
            <Field Name="CODIGO" Value="01"/>
            <Field Name="DISCRIMINACAO" Value="UT QUIS ALIQUAM LEO. DONEC ALIQUA"/>
            <Field Name="SITUACAOANTERIOR" Value="23.445,00"/>
            <Field Name="SITUACAOATUAL" Value="342.342,00"/>
            <Field Name="InscricaoMunicipal(IPTU)" Value="23423424"/>
            <Field Name="Logradouro" Value="RUA QUALQUER"/>
            <Field Name="Numero" Value="89"/>
            <Field Name="Complemento" Value="COMPLEM 2"/>
            <Field Name="Bairro" Value="BRASILIA"/>
            <Field Name="Municipio" Value="BRASÍLIA"/>
            <Field Name="UF" Value="DF"/>
            <Field Name="CEP" Value="1321587"/>
            <Field Name="AreaTotal" Value="345,0 m²"/>
            <Field Name="DatadeAquisicao" Value="12/12/1993"/>
            <Field Name="RegistradonoCartorio" Value="Sim"/>
            <Field Name="NomeCartorio" Value="CARTORIO DE SÇNJJKLCDF ASLK SAKÇK SAÇKLJ SAÇLKS"/>
            <Field Name="Matricula" Value="2344234"/>
        </ROW>
    </TABLE>
</SECTION>

Certifique-se de que todos os bens e direitos, com suas respectivas caracteristicas, estejam presentes no XML, 
incluindo SITUACAOANTERIOR e SITUACAOATUAL no formato SITUACAO EM DD/MM/AAAA. 
Se a informação não estiver disponível no contexto, deixe o campo vazio.
Cada conjunto de (GRUPO, CODIGO) é uma categoria distinta. 
Inclua também campos como RENAVAM, CPF, CNPJ, Banco, Conta, Agência, NegociadosemBolsa etc., se constarem no texto.
"""
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_prompt),
    ]

    response = llm.predict_messages(messages)
    return response.content

def main():
    bens_direitos_text = filtered_bens_direitos_text
    result = query_information(bens_direitos_text, file_identifier)

    # Ajuste do response caso precise remover algo de prefixo
    result = result.strip()

    file_name = f'./{file_identifier}.xml'
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(result)

    print(f"File {file_name} has been successfully saved.")

    # Upload do XML de volta ao MinIO
    upload_object_name = f"{file_identifier}.xml"
    bucket_name = "ai-bucket-rosa-hml"
    try:
        client.fput_object(bucket_name, upload_object_name, xml_file_name)
        print(f"'{upload_object_name}' successfully uploaded to bucket '{bucket_name}'.")
    except S3Error as e:
        print("Error occurred while uploading XML:", e)

if __name__ == "__main__":
    main()