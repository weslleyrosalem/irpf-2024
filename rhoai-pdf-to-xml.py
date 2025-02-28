import os
import sys
import uuid
import logging
import re

# Caso precise instalar pacotes de forma automática, é só descomentar e ajustar:
# import subprocess
# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# dependencias = [
#     "sentence-transformers", "pymilvus", "openai==0.28", "langchain", 
#     "langchain_community", "minio", "pymupdf", "Pillow", "pytesseract", 
#     "pandas", "langchain-milvus", "opencv-python", "pdf2image", "vllm",
#     "transformers"
# ]
# for dep in dependencias:
#     install(dep)

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

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# --------------------------------------------------------------------------------
# CONFIGURAÇÕES PRINCIPAIS
# --------------------------------------------------------------------------------
AWS_S3_ENDPOINT = "minio-api-minio.apps.cluster-lqsm2.lqsm2.sandbox441.opentlc.com"
AWS_ACCESS_KEY_ID = "minio"
AWS_SECRET_ACCESS_KEY = "minio123"
AWS_S3_BUCKET = "irpf-files"

MILVUS_HOST = "vectordb-milvus.milvus.svc.cluster.local"
MILVUS_PORT = 19530
MILVUS_USERNAME = "root"
MILVUS_PASSWORD = "Milvus"
MILVUS_COLLECTION = "irpf"

# Endereço do servidor e modelo LLM que será utilizado
INFERENCE_SERVER_URL = "http://vllm.vllm.svc.cluster.local:8000"
LLM_MODEL_NAME = "tiiuae/falcon-40b-instruct"

# Modelo usado para tokenização do Falcon
FALCON_CHECKPOINT = "tiiuae/falcon-40b-instruct"

http_client = urllib3.PoolManager(
    timeout=urllib3.Timeout(connect=5.0, read=10.0)
)

# Configura o cliente MinIO
client = Minio(
    AWS_S3_ENDPOINT,
    access_key=AWS_ACCESS_KEY_ID,
    secret_key=AWS_SECRET_ACCESS_KEY,
    secure=True,
    http_client=http_client
)

# Captura o nome do arquivo a partir de variável de ambiente, se existir
file_identifier = os.getenv('file_identifier', "joao-2023")
if not file_identifier:
    print("Erro: variável de ambiente file_identifier não definida.")
    sys.exit(1)

execution_id = str(uuid.uuid4())

object_name = f"{file_identifier}.pdf"
file_path = f"./{file_identifier}.pdf"
output_pdf_path = f"./{file_identifier}_no_watermark.pdf"

# --------------------------------------------------------------------------------
# BAIXA O PDF DO MINIO
# --------------------------------------------------------------------------------
try:
    client.fget_object(AWS_S3_BUCKET, object_name, file_path)
    print(f"Arquivo '{object_name}' baixado com sucesso em '{file_path}'.")
except S3Error as e:
    print("Ocorreu um erro ao baixar o PDF:", e)
    sys.exit(1)

# --------------------------------------------------------------------------------
# FUNÇÃO PARA REMOVER MARCA D'ÁGUA E/OU IMAGENS
# --------------------------------------------------------------------------------
def remove_watermark_advanced(pdf_path, output_path):
    """
    Remove imagens de cada página e oculta anotações que contenham 'Watermark' no título.
    Pode apagar conteúdos importantes se o PDF for todo em imagem, então revise o uso.
    """
    doc = fitz.open(pdf_path)
    for page in doc:
        # Deleta todas as imagens
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            page.delete_image(xref)

        # Verifica se há anotações com título "Watermark" e as oculta
        annots = page.annots()
        if annots:
            for annot in annots:
                info = annot.info
                if "Watermark" in info.get("title", ""):
                    annot.set_flags(fitz.ANNOT_HIDDEN)

        page.apply_redactions()

    doc.save(output_path)
    print(f"Marca d'água removida (ou imagens deletadas): {output_path}")

# Caso necessário, comente essa linha se não quiser remover todas as imagens
remove_watermark_advanced(file_path, output_pdf_path)

# --------------------------------------------------------------------------------
# EXTRAÇÃO DE TEXTO VIA OCR
# --------------------------------------------------------------------------------
def extract_text_ocr(pdf_path):
    """
    Converte páginas para imagens e aciona o Tesseract. 
    Em PDFs que já tenham texto interno, a extração direta via fitz pode ser melhor.
    """
    extracted_text = ""
    if not os.path.exists(pdf_path):
        print(f"Arquivo não encontrado: {pdf_path}")
        return ""

    try:
        pages = convert_from_path(pdf_path)
        for i, page_img in enumerate(pages):
            print(f"OCR na página {i+1}/{len(pages)}...")

            ocr_config = r'-c preserve_interword_spaces=1 --oem 1 --psm 1'
            ocr_data = pytesseract.image_to_data(
                page_img, 
                lang='por', 
                config=ocr_config, 
                output_type=Output.DICT
            )
            df = pd.DataFrame(ocr_data)
            # Filtra texto vazio ou com confiança -1
            df = df[(df.conf != '-1') & (df.text.str.strip() != '')]
            if df.empty:
                continue

            # Agrupa por blocos para manter a sequência de leitura
            block_nums = df.groupby('block_num').first().sort_values('top').index.tolist()

            for block_num in block_nums:
                block_df = df[df['block_num'] == block_num]
                # Se quiser ignorar trechos muito curtos, pode filtrar aqui
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
        print(f"Erro durante o OCR: {e}")
        return ""

# --------------------------------------------------------------------------------
# CONFIGURAÇÃO E INSERÇÃO NO MILVUS
# --------------------------------------------------------------------------------
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
    Separa o texto em partes, evitando blocos muito grandes para indexação.
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
    if not text_parts:
        print("Nenhum bloco de texto encontrado, ignorando inserção no Milvus.")
        return

    print(f"Quantidade de blocos de texto: {len(text_parts)}")
    for i, part in enumerate(text_parts):
        metadata = {"source": pdf_file, "part": i, "execution_id": execution_id}
        print(f"Inserindo chunk #{i}, tamanho {len(part)} caracteres")
        store.add_texts([part], metadatas=[metadata])

# --------------------------------------------------------------------------------
# CARREGA TOKENIZER DO FALCON PARA CONTROLAR NÚMERO DE TOKENS
# --------------------------------------------------------------------------------
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(FALCON_CHECKPOINT, trust_remote_code=True)

def count_tokens(text: str) -> int:
    """
    Calcula quantos tokens são gerados a partir de uma string com o tokenizer do Falcon.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

# --------------------------------------------------------------------------------
# CONFIGURAÇÃO DO MODELO LLM
# --------------------------------------------------------------------------------
llm = ChatOpenAI(
    openai_api_key="EMPTY",
    openai_api_base=f"{INFERENCE_SERVER_URL}/v1",
    model_name=LLM_MODEL_NAME,
    top_p=0.92,
    temperature=0.01,
    presence_penalty=1.03,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# --------------------------------------------------------------------------------
# FUNÇÃO PARA QUEBRAR O TEXTO EM PARTES SEM ESTOURAR A JANELA DE CONTEXTO
# --------------------------------------------------------------------------------
def chunk_text_for_falcon(text, max_prompt_tokens=1800):
    """
    Fragmenta o texto para assegurar que, junto das instruções fixas,
    o total não exceda 2048 tokens do Falcon. Ajuste se necessário.
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

# --------------------------------------------------------------------------------
# GERA UM TRECHO DE XML PARA CADA BLOCO DE TEXTO
# --------------------------------------------------------------------------------
def generate_xml_snippet(llm, chunk, file_identifier):
    """
    Solicita ao LLM um pedaço de XML referente apenas ao conteúdo do chunk.
    """
    system_prompt = (
        "Você é um assistente de impostos capaz de gerar XML de bens/direitos com base no texto fornecido. "
        "Analise somente o trecho que será passado e gere a parte do XML correspondente."
    )

    user_prompt = (
        f"Trecho:\n{chunk}\n\n"
        f"Crie o XML para os bens identificados nesse pedaço, seguindo o padrão usado em {file_identifier}, "
        "sem adicionar explicações fora das tags."
    )

    total_prompt = system_prompt + user_prompt
    total_tokens = count_tokens(total_prompt)
    if total_tokens > 2048:
        raise ValueError(f"O prompt atual ultrapassa 2048 tokens (tem {total_tokens}). Ajuste o tamanho do chunk.")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    response = llm.predict_messages(messages)
    snippet = response.content.strip()

    # Remove possíveis fences de markdown
    if snippet.startswith("```"):
        snippet = snippet.lstrip("```").strip()
    if snippet.endswith("```"):
        snippet = snippet.rstrip("```").strip()

    return snippet

# --------------------------------------------------------------------------------
# FUNÇÕES PARA LIMPEZA E UNIÃO DOS FRAGMENTOS DE XML
# --------------------------------------------------------------------------------
def clean_snippet_for_merge(snippet_xml: str) -> str:
    """
    Tira declarações de XML e tags <SECTION> para facilitar a fusão posterior.
    """
    snippet_xml = re.sub(r'<\?xml.*?\?>', '', snippet_xml, flags=re.IGNORECASE)
    snippet_xml = re.sub(r'<SECTION.*?>', '', snippet_xml, flags=re.IGNORECASE)
    snippet_xml = re.sub(r'</SECTION>', '', snippet_xml, flags=re.IGNORECASE)
    return snippet_xml.strip()

def merge_snippets(snippets):
    """
    Concatena vários trechos de XML em uma só <SECTION>, evitando duplicações.
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

# --------------------------------------------------------------------------------
# FLUXO PRINCIPAL
# --------------------------------------------------------------------------------
def main():
    # 1) Executa OCR
    text = extract_text_ocr(output_pdf_path)
    print(f"\nTamanho do texto extraído: {len(text)} caracteres.\n")

    # 2) Cria blocos grandes para indexação no Milvus (opcional)
    text_parts = split_text(text)
    store_text_parts_in_milvus(text_parts, os.path.basename(output_pdf_path), execution_id)
    print("\nPDF processado e inserido no Milvus.")

    # 3) Divide o texto para garantir que o LLM não exceda o limite de contexto
    chunks_for_llm = chunk_text_for_falcon(text, max_prompt_tokens=1800)
    print(f"Quantidade de partes a serem processadas pelo LLM: {len(chunks_for_llm)}\n")

    # 4) Processa cada trecho e gera um snippet de XML
    xml_snippets = []
    for i, chunk in enumerate(chunks_for_llm, start=1):
        print(f"[Trecho {i}/{len(chunks_for_llm)}] tokens: {count_tokens(chunk)}")
        snippet = generate_xml_snippet(llm, chunk, file_identifier)
        xml_snippets.append(snippet)

    # 5) Combina todos os fragmentos num XML final
    final_xml = merge_snippets(xml_snippets)

    # 6) Salva o arquivo XML localmente
    xml_file_name = f"./{file_identifier}.xml"
    with open(xml_file_name, "w", encoding="utf-8") as f:
        f.write(final_xml)
    print(f"Arquivo {xml_file_name} criado com sucesso.")

    # 7) Sobe o XML no MinIO
    upload_object_name = f"{file_identifier}.xml"
    bucket_name = "irpf-xml"
    try:
        client.fput_object(bucket_name, upload_object_name, xml_file_name)
        print(f"'{upload_object_name}' enviado para o bucket '{bucket_name}'.")
    except S3Error as e:
        print("Erro ao subir o XML:", e)

if __name__ == "__main__":
    main()
