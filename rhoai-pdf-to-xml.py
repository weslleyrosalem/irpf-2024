import os
import sys
import uuid
import logging
import re

logging.basicConfig(level=logging.DEBUG)

import fitz
import pytesseract
from pytesseract import Output
import pandas as pd
import urllib3
from pdf2image import convert_from_path

from minio import Minio
from minio.error import S3Error

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Milvus
from langchain.llms import OpenAI

import requests

from transformers import AutoTokenizer

# ------------------------------------------------------------------------------
# CONFIGURAÇÕES BÁSICAS
# ------------------------------------------------------------------------------
AWS_S3_ENDPOINT = "minio-api-minio.apps.cluster-mx7zs.mx7zs.sandbox2523.opentlc.com"
AWS_ACCESS_KEY_ID = "minio"
AWS_SECRET_ACCESS_KEY = "minio123"
AWS_S3_BUCKET = "irpf-files"

MILVUS_HOST = "vectordb-milvus.milvus.svc.cluster.local"
MILVUS_PORT = 19530
MILVUS_USERNAME = "root"
MILVUS_PASSWORD = "Milvus"
MILVUS_COLLECTION = "irpf"

INFERENCE_SERVER_URL = "http://falcon-40b-instruct-service.kserve-demo.svc.cluster.local:8080"
LLM_MODEL_NAME = "falcon-40b-instruct"
FALCON_CHECKPOINT = "tiiuae/falcon-40b-instruct"

http_client = urllib3.PoolManager(timeout=urllib3.Timeout(connect=5.0, read=10.0))
client = Minio(
    AWS_S3_ENDPOINT,
    access_key=AWS_ACCESS_KEY_ID,
    secret_key=AWS_SECRET_ACCESS_KEY,
    secure=True,
    http_client=http_client
)

file_identifier = os.getenv('file_identifier', "joao-2023")
execution_id = str(uuid.uuid4())

object_name = f"{file_identifier}.pdf"
file_path = f"./{file_identifier}.pdf"
output_pdf_path = f"./{file_identifier}_no_watermark.pdf"

# ------------------------------------------------------------------------------
# DOWNLOAD DO ARQUIVO PDF DO MINIO
# ------------------------------------------------------------------------------
try:
    client.fget_object(AWS_S3_BUCKET, object_name, file_path)
    print(f"Download do arquivo '{object_name}' concluído, salvo em: '{file_path}'.")
except S3Error as e:
    print("Houve um problema ao baixar o PDF:", e)
    sys.exit(1)

# ------------------------------------------------------------------------------
# FUNÇÃO PARA REMOVER MARCA D'ÁGUA E IMAGENS DO PDF
# ------------------------------------------------------------------------------
def remove_watermark_advanced(pdf_path, output_path):
    doc = fitz.open(pdf_path)
    for page in doc:
        # Apagar imagens
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            page.delete_image(xref)

        # Esconder anotações "Watermark"
        annots = page.annots()
        if annots:
            for annot in annots:
                info = annot.info
                if "Watermark" in info.get("title", ""):
                    annot.set_flags(fitz.ANNOT_HIDDEN)

        page.apply_redactions()

    doc.save(output_path)
    print(f"Arquivo PDF pós-remoção de marca d'água gerado em: {output_path}")

remove_watermark_advanced(file_path, output_pdf_path)

# ------------------------------------------------------------------------------
# OCR
# ------------------------------------------------------------------------------
def extract_text_ocr(pdf_path):
    extracted_text = ""
    if not os.path.exists(pdf_path):
        print(f"O arquivo {pdf_path} não existe.")
        return ""

    try:
        pages = convert_from_path(pdf_path)
        for i, page_img in enumerate(pages):
            ocr_config = r'-c preserve_interword_spaces=1 --oem 1 --psm 3'
            ocr_data = pytesseract.image_to_data(
                page_img,
                lang='por',
                config=ocr_config,
                output_type=Output.DICT
            )
            df = pd.DataFrame(ocr_data)
            df = df[(df.conf != '-1') & (df.text.str.strip() != '') & (df.conf.astype(float) > 50)]
            if df.empty:
                continue

            block_nums = df.groupby('block_num').first().sort_values('top').index.tolist()
            for block_num in block_nums:
                block_df = df[df['block_num'] == block_num]
                avg_char_width = (block_df.width / block_df.text.str.len()).mean() if not block_df.empty else 0
                prev_par, prev_line, prev_left = 0, 0, 0

                for _, line_data in block_df.iterrows():
                    if prev_par != line_data['par_num']:
                        extracted_text += '\n'
                        prev_par = line_data['par_num']
                        prev_line = line_data['line_num']
                        prev_left = 0
                    elif prev_line != line_data['line_num']:
                        extracted_text += '\n'
                        prev_line = line_data['line_num']
                        prev_left = 0

                    spaces_to_add = 0
                    if avg_char_width:
                        left_margin_chars = int(line_data['left'] / avg_char_width)
                    else:
                        left_margin_chars = 0

                    if left_margin_chars > prev_left + 1:
                        spaces_to_add = left_margin_chars - prev_left
                        extracted_text += ' ' * spaces_to_add

                    extracted_text += line_data['text'] + ' '
                    prev_left += len(line_data['text']) + spaces_to_add + 1

                extracted_text += '\n'
        return extracted_text
    except Exception as e:
        print(f"Falha durante a execução do OCR: {e}")
        return ""

# ------------------------------------------------------------------------------
# Indexação no Milvus (opcional)
# ------------------------------------------------------------------------------
class EmbeddingFunctionWrapper:
    def __init__(self, model):
        self.model = model

    def embed_query(self, query):
        emb = self.model.encode(query)
        return emb

    def embed_documents(self, docs):
        emb = self.model.encode(docs)
        return emb

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_function = EmbeddingFunctionWrapper(embedding_model)

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

def split_text(text, max_length=60000):
    words = text.split()
    parts = []
    current_part = []
    current_len = 0

    for w in words:
        if current_len + len(w) + 1 <= max_length:
            current_part.append(w)
            current_len += len(w) + 1
        else:
            parts.append(" ".join(current_part))
            current_part = [w]
            current_len = len(w) + 1

    if current_part:
        parts.append(" ".join(current_part))
    return parts

def store_text_parts_in_milvus(text_parts, pdf_file, execution_id):
    if not text_parts:
        print("Não há blocos de texto para inserir no Milvus.")
        return

    print(f"Número de blocos de texto: {len(text_parts)}")
    for i, part in enumerate(text_parts):
        metadata = {"source": pdf_file, "part": i, "execution_id": execution_id}
        store.add_texts([part], metadatas=[metadata])

# ----------------------------------------------------------------------------
# Tokenizer do Falcon para contagem
# ----------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(FALCON_CHECKPOINT, trust_remote_code=True)

def count_tokens(text: str) -> int:
    # Só para referência de chunking
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

# ----------------------------------------------------------------------------
# Ajustar o LLM (passo importante: max_tokens!)
# ----------------------------------------------------------------------------
llm = OpenAI(
    openai_api_key="EMPTY",  
    openai_api_base=f"{INFERENCE_SERVER_URL}/v1",
    model_name=LLM_MODEL_NAME,
    streaming=False,
    temperature=0.01,
    top_p=0.92,
    presence_penalty=1.03,
    # AQUI definimos quantos tokens de RESPOSTA (max_tokens) esperamos.
    # Ajuste conforme o endpoint do Falcon permitir (ex: 2048, 4096...).
    max_tokens=1000
)

# ----------------------------------------------------------------------------
# Dividir texto em chunks p/ LLM
# ----------------------------------------------------------------------------
def chunk_text_for_falcon(text, max_prompt_tokens=600):
    """
    Divide o texto em blocos tais que, ao somar o texto e o prompt,
    não corra risco de truncamento. max_prompt_tokens aqui é
    só do CHUNK; o prompt principal também gasta tokens.
    """
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        tentative = " ".join(current_chunk + [word])
        if count_tokens(tentative) <= max_prompt_tokens:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# ----------------------------------------------------------------------------
# Função para gerar snippet XML a partir de um chunk
# ----------------------------------------------------------------------------
def generate_xml_snippet(llm, chunk, file_identifier):
    system_prompt = (
        "Você é um assistente especializado em declaração de Imposto de Renda, "
        "capaz de extrair dados de bens e direitos a partir do texto fornecido. "
        "Retorne SOMENTE o XML, sem explicações adicionais. "
        "Se faltarem dados, deixe o campo vazio. NÃO interrompa o XML."
    )

    user_prompt = (
        f"Trecho do documento:\n{chunk}\n\n"
        "Gere um XML que represente os bens encontrados. Use a estrutura abaixo como base:\n\n"
        "<?xml version=\"1.0\"?>\n"
        "<SECTION Name=\"DECLARACAO DE BENS E DIREITOS\">\n"
        "    <TABLE>\n"
        "        <ROW No=\"1\">\n"
        "            <Field Name=\"GRUPO\" Value=\"\"/>\n"
        "            <Field Name=\"CODIGO\" Value=\"\"/>\n"
        "            <Field Name=\"DISCRIMINACAO\" Value=\"\"/>\n"
        "            <Field Name=\"SITUACAOANTERIOR\" Value=\"\"/>\n"
        "            <Field Name=\"SITUACAOATUAL\" Value=\"\"/>\n"
        "            <!-- Adicione mais fields se necessário -->\n"
        "        </ROW>\n"
        "    </TABLE>\n"
        "</SECTION>\n\n"
        "Incorpore TODOS os bens do trecho no XML. Caso algum dado esteja ausente, deixe vazio. "
        "Não retorne qualquer texto além do XML final."
    )

    full_prompt = system_prompt + "\n\n" + user_prompt

    # >>> Se quisermos, podemos remover este check ou aumentar o limite <<<
    # total_tokens = count_tokens(full_prompt)
    # if total_tokens > 2048:
    #     raise ValueError(f"Prompt estourou 2048 tokens (={total_tokens}). Ajuste a chunkagem!")

    snippet = llm(full_prompt).strip()

    # Remoção de cercas de Markdown
    if snippet.startswith("```"):
        snippet = snippet.lstrip("```").strip()
    if snippet.endswith("```"):
        snippet = snippet.rstrip("```").strip()

    return snippet

# ----------------------------------------------------------------------------
# Limpeza e união dos fragmentos de XML
# ----------------------------------------------------------------------------
def clean_snippet_for_merge(snippet_xml: str) -> str:
    # Remove possíveis declarações e tags de SECTION para unificar
    snippet_xml = re.sub(r'<\?xml.*?\?>', '', snippet_xml, flags=re.IGNORECASE)
    snippet_xml = re.sub(r'<SECTION.*?>', '', snippet_xml, flags=re.IGNORECASE)
    snippet_xml = re.sub(r'</SECTION>', '', snippet_xml, flags=re.IGNORECASE)
    return snippet_xml.strip()

def merge_snippets(snippets):
    merged_rows = []
    for snip in snippets:
        if snip.strip():
            merged_rows.append(clean_snippet_for_merge(snip))

    joined_rows = "\n".join(merged_rows)
    final_xml = (
        "<?xml version=\"1.0\" ?>\n"
        "<SECTION Name=\"DECLARACAO DE BENS E DIREITOS\">\n"
        + joined_rows + "\n"
        "</SECTION>\n"
    )
    return final_xml

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    # 1) OCR
    text = extract_text_ocr(output_pdf_path)
    print(f"\nTexto extraído tem {len(text)} caracteres.\n")

    # 2) Indexar no Milvus
    text_parts = split_text(text)
    store_text_parts_in_milvus(text_parts, os.path.basename(output_pdf_path), execution_id)
    print("Inserção no Milvus concluída.")

    # 3) Dividir em chunks para LLM
    chunks_for_llm = chunk_text_for_falcon(text, max_prompt_tokens=600)
    print(f"Número de blocos para o LLM: {len(chunks_for_llm)}\n")

    # 4) Gera XML parcial para cada bloco
    xml_snippets = []
    for i, chunk in enumerate(chunks_for_llm, start=1):
        token_count = count_tokens(chunk)
        print(f"Processando bloco {i}/{len(chunks_for_llm)}. Tokens no chunk: {token_count}")
        snippet = generate_xml_snippet(llm, chunk, file_identifier)
        xml_snippets.append(snippet)

    # 5) Une tudo
    final_xml = merge_snippets(xml_snippets)

    # 6) Salva local
    xml_file_name = f"./{file_identifier}.xml"
    with open(xml_file_name, "w", encoding="utf-8") as f:
        f.write(final_xml)

    print(f"XML gerado e salvo em: {xml_file_name}")

    # 7) Upload ao MinIO (bucket 'irpf-xml')
    upload_object_name = f"{file_identifier}.xml"
    bucket_name = "irpf-xml"
    try:
        client.fput_object(bucket_name, upload_object_name, xml_file_name)
        print(f"Arquivo '{upload_object_name}' enviado para o bucket '{bucket_name}'.")
    except S3Error as e:
        print("Falha ao enviar o arquivo XML:", e)

if __name__ == "__main__":
    main()