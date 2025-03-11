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

# Usando a classe OpenAI do LangChain para conversar com /v1/completions
from langchain.llms import OpenAI

import requests

# ------------------------------------------------------------------------------
# CONFIGURAÇÕES BÁSICAS DO SISTEMA
# ------------------------------------------------------------------------------
AWS_S3_ENDPOINT = "minio-api-minio.apps.cluster-lqsm2.lqsm2.sandbox441.opentlc.com"
AWS_ACCESS_KEY_ID = "minio"
AWS_SECRET_ACCESS_KEY = "minio123"
AWS_S3_BUCKET = "irpf-files"

MILVUS_HOST = "vectordb-milvus.milvus.svc.cluster.local"
MILVUS_PORT = 19530
MILVUS_USERNAME = "root"
MILVUS_PASSWORD = "Milvus"
MILVUS_COLLECTION = "irpf"

# Informações do endpoint de inferência, que aceita chamadas em /v1/completions
INFERENCE_SERVER_URL = "http://falcon-40b-instruct-service.vllm.svc.cluster.local:8080"
LLM_MODEL_NAME = "falcon-40b-instruct"

# Usado para tokenização (cálculo de tokens)
FALCON_CHECKPOINT = "tiiuae/falcon-40b-instruct"

# Configuração para conexão com o MinIO
http_client = urllib3.PoolManager(timeout=urllib3.Timeout(connect=5.0, read=10.0))
client = Minio(
    AWS_S3_ENDPOINT,
    access_key=AWS_ACCESS_KEY_ID,
    secret_key=AWS_SECRET_ACCESS_KEY,
    secure=True,
    http_client=http_client
)

# Se uma variável de ambiente "file_identifier" estiver definida, uso ela; caso contrário, uso "joao-2023"
file_identifier = os.getenv('file_identifier', "joao-2023")
if not file_identifier:
    print("Erro: a variável de ambiente file_identifier não está configurada.")
    sys.exit(1)

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
    """
    Aqui removo todas as imagens de cada página e oculto qualquer anotação
    que tenha 'Watermark' como título. Use com cuidado, pois o PDF pode ser
    todo em imagem.
    """
    doc = fitz.open(pdf_path)
    for page in doc:
        # Apago todas as imagens da página
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            page.delete_image(xref)

        # Se a anotação tiver "Watermark" no título, ela é ocultada
        annots = page.annots()
        if annots:
            for annot in annots:
                info = annot.info
                if "Watermark" in info.get("title", ""):
                    annot.set_flags(fitz.ANNOT_HIDDEN)

        page.apply_redactions()

    doc.save(output_path)
    print(f"Foram removidas marcas d'água e imagens. Novo arquivo: {output_path}")

# Remover as imagens e possíveis marcas d'água
remove_watermark_advanced(file_path, output_pdf_path)

# ------------------------------------------------------------------------------
# APLICAÇÃO DE OCR NO PDF
# ------------------------------------------------------------------------------
def extract_text_ocr(pdf_path):
    """
    Converte cada página em imagem e extrai texto via Tesseract.
    Se o PDF já tiver texto embutido, talvez seja melhor extrair via fitz.
    """
    extracted_text = ""
    if not os.path.exists(pdf_path):
        print(f"O arquivo {pdf_path} não existe.")
        return ""

    try:
        pages = convert_from_path(pdf_path)
        for i, page_img in enumerate(pages):
            print(f"Realizando OCR da página {i+1}/{len(pages)}...")

            ocr_config = r'-c preserve_interword_spaces=1 --oem 1 --psm 1'
            ocr_data = pytesseract.image_to_data(
                page_img,
                lang='por',
                config=ocr_config,
                output_type=Output.DICT
            )
            df = pd.DataFrame(ocr_data)

            # Filtra qualquer linha de texto inválida
            df = df[(df.conf != '-1') & (df.text.str.strip() != '')]
            if df.empty:
                continue

            # Agrupo por bloco para manter a ordem original
            block_nums = df.groupby('block_num').first().sort_values('top').index.tolist()
            for block_num in block_nums:
                block_df = df[df['block_num'] == block_num]
                filtered = block_df[block_df.text.str.len() > 2]
                if filtered.empty:
                    continue

                avg_char_width = (filtered.width / filtered.text.str.len()).mean()
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
# PREPARAÇÃO PARA O MILVUS
# ------------------------------------------------------------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class EmbeddingFunctionWrapper:
    def __init__(self, model):
        self.model = model

    def embed_query(self, query):
        emb = self.model.encode(query)
        logging.debug(f"embed_query shape: {emb.shape}")
        return emb

    def embed_documents(self, docs):
        emb = self.model.encode(docs)
        logging.debug(f"embed_documents shape: {emb.shape}")
        return emb

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
    """
    Aqui vou dividir o texto em pedaços para não sobrecarregar o Milvus.
    """
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
    """
    Envio dos blocos de texto para o Milvus, mantendo um metadado simples.
    """
    if not text_parts:
        print("Não há blocos de texto para inserir no Milvus.")
        return

    print(f"Número de blocos de texto: {len(text_parts)}")
    for i, part in enumerate(text_parts):
        metadata = {"source": pdf_file, "part": i, "execution_id": execution_id}
        print(f"Inserindo bloco #{i}, com tamanho de {len(part)} caracteres.")
        store.add_texts([part], metadatas=[metadata])

# ------------------------------------------------------------------------------
# USO DO TOKENIZER DO FALCON PARA CALCULAR QUANTIDADE DE TOKENS
# ------------------------------------------------------------------------------
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(FALCON_CHECKPOINT, trust_remote_code=True)

def count_tokens(text: str) -> int:
    """
    Uso o tokenizer do Falcon para saber quantos tokens existem no texto.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

# ------------------------------------------------------------------------------
# CONFIGURAÇÃO DO MODELO (USANDO OPENAI PARA /v1/completions)
# ------------------------------------------------------------------------------
llm = OpenAI(
    openai_api_key="EMPTY",  # Aqui pode ser algo fictício, caso o endpoint não exija key
    openai_api_base=f"{INFERENCE_SERVER_URL}/v1",
    model_name=LLM_MODEL_NAME,
    streaming=False,  # Ajuste se o seu endpoint der suporte a streaming
    temperature=0.01,
    top_p=0.92,
    presence_penalty=1.03
)

# ------------------------------------------------------------------------------
# DIVISÃO DO TEXTO EM PARTES MENORES PARA O LLM
# ------------------------------------------------------------------------------
def chunk_text_for_falcon(text, max_prompt_tokens=1800):
    """
    Quebro o texto em sub-blocos para não exceder limite de contexto.
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

# ------------------------------------------------------------------------------
# FUNÇÃO PARA GERAR TRECHO DE XML PARA CADA SUB-BLOCO
# ------------------------------------------------------------------------------
def generate_xml_snippet(llm, chunk, file_identifier):
    """
    Fazemos um único prompt, já que estamos usando /v1/completions (OpenAI).
    """
    system_prompt = (
        "Você é um assistente de impostos que cria XML de bens/direitos a partir do texto dado.\n"
        "Use somente o trecho que vou fornecer para criar esse XML.\n\n"
    )
    user_prompt = (
        f"Trecho:\n{chunk}\n\n"
        f"Por favor, gere o XML que corresponde aos bens encontrados aqui, "
        f"baseado no padrão do arquivo {file_identifier}, sem incluir explicações fora das tags."
    )

    full_prompt = system_prompt + user_prompt
    total_tokens = count_tokens(full_prompt)
    if total_tokens > 2048:
        raise ValueError(f"O prompt tem {total_tokens} tokens, excedendo o limite de 2048. "
                         "Diminua o tamanho no chunk_text_for_falcon.")

    snippet = llm(full_prompt).strip()

    # Remoção de possíveis cercas do Markdown
    if snippet.startswith("```"):
        snippet = snippet.lstrip("```").strip()
    if snippet.endswith("```"):
        snippet = snippet.rstrip("```").strip()

    return snippet

# ------------------------------------------------------------------------------
# FUNÇÕES PARA LIMPEZA E UNIÃO DOS FRAGMENTOS DE XML
# ------------------------------------------------------------------------------
def clean_snippet_for_merge(snippet_xml: str) -> str:
    """
    Vou remover declarações de XML e tags <SECTION> para unir tudo num só bloco depois.
    """
    snippet_xml = re.sub(r'<\?xml.*?\?>', '', snippet_xml, flags=re.IGNORECASE)
    snippet_xml = re.sub(r'<SECTION.*?>', '', snippet_xml, flags=re.IGNORECASE)
    snippet_xml = re.sub(r'</SECTION>', '', snippet_xml, flags=re.IGNORECASE)
    return snippet_xml.strip()

def merge_snippets(snippets):
    """
    Uno todos os pedaços de XML em um único <SECTION>.
    """
    merged_rows = []
    for snip in snippets:
        inner = clean_snippet_for_merge(snip)
        merged_rows.append(inner)

    final_xml = """<?xml version="1.0" ?>
<SECTION Name="DECLARACAO DE BENS E DIREITOS">
{rows}
</SECTION>
""".format(rows="\n".join(merged_rows))

    return final_xml

# ------------------------------------------------------------------------------
# FLUXO PRINCIPAL DO SCRIPT
# ------------------------------------------------------------------------------
def main():
    # 1) Passo: extrair texto do PDF (após remoção das marcas d'água)
    text = extract_text_ocr(output_pdf_path)
    print(f"\nTexto extraído tem {len(text)} caracteres.\n")

    # 2) (Opcional) quebro o texto em partes grandes e indexo no Milvus
    text_parts = split_text(text)
    store_text_parts_in_milvus(text_parts, os.path.basename(output_pdf_path), execution_id)
    print("Inserção no Milvus concluída.")

    # 3) Divido o texto em blocos que caibam no contexto do modelo
    chunks_for_llm = chunk_text_for_falcon(text, max_prompt_tokens=1800)
    print(f"Número de blocos para o LLM: {len(chunks_for_llm)}\n")

    # 4) Para cada bloco, gero um snippet de XML
    xml_snippets = []
    for i, chunk in enumerate(chunks_for_llm, start=1):
        token_count = count_tokens(chunk)
        print(f"Processando bloco {i}/{len(chunks_for_llm)}. Tokens: {token_count}")
        snippet = generate_xml_snippet(llm, chunk, file_identifier)
        xml_snippets.append(snippet)

    # 5) Combino todos os trechos de XML num só arquivo final
    final_xml = merge_snippets(xml_snippets)

    # 6) Gravo o XML em um arquivo
    xml_file_name = f"./{file_identifier}.xml"
    with open(xml_file_name, "w", encoding="utf-8") as f:
        f.write(final_xml)
    print(f"XML salvo em: {xml_file_name}")

    # 7) Envio o arquivo para o bucket 'irpf-xml' no MinIO
    upload_object_name = f"{file_identifier}.xml"
    bucket_name = "irpf-xml"
    try:
        client.fput_object(bucket_name, upload_object_name, xml_file_name)
        print(f"Arquivo '{upload_object_name}' enviado para o bucket '{bucket_name}'.")
    except S3Error as e:
        print("Falha ao enviar o arquivo XML:", e)

if __name__ == "__main__":
    main()
