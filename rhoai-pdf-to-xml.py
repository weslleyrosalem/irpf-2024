#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import logging
import uuid
import fitz  # PyMuPDF
import pandas as pd
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
from minio import Minio
from minio.error import S3Error
import urllib3

# LangChain e SentenceTransformers
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Milvus
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# Configurações de logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)

# ==============================
# Configurações Gerais
# ==============================
AWS_S3_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio-api-minio.apps.cluster-mx7zs.mx7zs.sandbox2523.opentlc.com")
AWS_ACCESS_KEY_ID = os.getenv("MINIO_ACCESS_KEY", "minio")
AWS_SECRET_ACCESS_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
AWS_S3_BUCKET = os.getenv("MINIO_BUCKET", "irpf-files")

# Milvus
MILVUS_HOST = os.getenv("MILVUS_HOST", "vectordb-milvus.milvus.svc.cluster.local")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_USERNAME = os.getenv("MILVUS_USERNAME", "root")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "Milvus")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "irpf")

# LLM e inference server
INFERENCE_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://vllm.vllm.svc.cluster.local:8000")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

# Parâmetros do LLM
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.01"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.92"))
LLM_PRESENCE_PENALTY = float(os.getenv("LLM_PRESENCE_PENALTY", "1.03"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "20000"))  # Ajuste conforme o real suporte do seu modelo

# Campo obtido do environment, representando o arquivo
file_identifier = os.getenv('file_identifier')
if not file_identifier:
    logging.error("Error: file_identifier environment variable is not set.")
    sys.exit(1)

# Nome do PDF
object_name = f"{file_identifier}.pdf"
file_path = f"./{file_identifier}.pdf"
output_pdf_path = f"./{file_identifier}_no_watermark.pdf"

# Gera um ID de execução
execution_id = str(uuid.uuid4())

# ==============================
# Inicializa MinIO com PoolManager customizado
# ==============================
http_client = urllib3.PoolManager(timeout=urllib3.Timeout(connect=5.0, read=10.0))
minio_client = Minio(
    AWS_S3_ENDPOINT,
    access_key=AWS_ACCESS_KEY_ID,
    secret_key=AWS_SECRET_ACCESS_KEY,
    secure=True,
    http_client=http_client
)

# ==============================
# Funções Utilitárias
# ==============================
def download_file_from_minio(bucket, obj_name, local_path):
    """Baixa um objeto (PDF) do MinIO para o caminho local local_path."""
    try:
        minio_client.fget_object(bucket, obj_name, local_path)
        logging.info(f"Arquivo '{obj_name}' baixado de '{bucket}' para '{local_path}' com sucesso.")
    except S3Error as e:
        logging.error(f"Erro ao baixar '{obj_name}' do bucket '{bucket}': {e}")
        sys.exit(1)

def upload_file_to_minio(bucket, obj_name, local_path):
    """Faz upload de um arquivo local para o MinIO no bucket e objeto especificados."""
    try:
        minio_client.fput_object(bucket, obj_name, local_path)
        logging.info(f"Arquivo '{obj_name}' enviado para '{bucket}' com sucesso.")
    except S3Error as e:
        logging.error(f"Erro ao enviar '{obj_name}' para '{bucket}': {e}")

def remove_watermark_advanced(pdf_input, pdf_output):
    """
    Remove possíveis marca-d'água de PDF, apagando imagens e anotações do tipo Watermark.
    Se necessário, refine a detecção (p.ex. layers).
    """
    try:
        doc = fitz.open(pdf_input)
        for page in doc:
            # Remove as imagens
            for img in page.get_images(full=True):
                xref = img[0]
                page.delete_image(xref)

            # Remove anotações marcadas como Watermark
            annots = page.annots()
            if annots:
                for annot in annots:
                    annot_info = annot.info
                    if "Watermark" in annot_info.get("title", ""):
                        annot.set_flags(fitz.ANNOT_HIDDEN)

            # Aplica qualquer redaction pendente
            page.apply_redactions()

        doc.save(pdf_output)
        logging.info(f"Marca-d'água (se existente) removida do PDF e salvo em: {pdf_output}")
    except Exception as e:
        logging.warning(f"Erro ao tentar remover marca-d'água do PDF: {e}")

def extract_text_ocr(pdf_path):
    """
    Converte cada página do PDF para imagem e executa OCR via Tesseract.
    Retorna uma string com o texto extraído.
    """
    extracted_text = []
    if not os.path.exists(pdf_path):
        logging.error(f"Arquivo {pdf_path} não encontrado para OCR.")
        return ""

    try:
        pages = convert_from_path(pdf_path)
    except Exception as e:
        logging.error(f"Erro ao converter PDF em imagens: {e}")
        return ""

    for i, page_image in enumerate(pages, start=1):
        logging.info(f"Processando OCR na página {i} do PDF...")
        # Ajuste de --psm 
        ocr_config = r'-c preserve_interword_spaces=1 --oem 1 --psm 1'
        ocr_data = pytesseract.image_to_data(page_image, lang='por', config=ocr_config, output_type=Output.DICT)
        df_ocr = pd.DataFrame(ocr_data)

        # Filtra linhas válidas
        df_clean = df_ocr[(df_ocr.conf != '-1') & (df_ocr.text.str.strip() != "")]
        if df_clean.empty:
            continue

        # Calcula média de largura de caracteres para heurística de espaçamento
        df_text_len = df_clean[df_clean.text.str.len() > 3]
        if not df_text_len.empty:
            avg_char_width = (df_text_len.width / df_text_len.text.str.len()).mean()
        else:
            avg_char_width = 5.0  # fallback

        # Ordena por blocos e linhas
        block_nums_sorted = df_clean.groupby('block_num').first().sort_values('top').index.tolist()
        page_text = []

        for block_num in block_nums_sorted:
            block_data = df_clean[df_clean['block_num'] == block_num]
            # Agrupa por linha
            line_nums_sorted = block_data.groupby('line_num').first().sort_values('top').index.tolist()
            for line_num in line_nums_sorted:
                line_data = block_data[block_data['line_num'] == line_num].sort_values('left')
                
                line_str_parts = []
                prev_right = 0
                for _, token in line_data.iterrows():
                    left_pos = token['left']
                    gap = left_pos - prev_right
                    # Heurística: se gap grande, adiciona espaço
                    if gap > avg_char_width:
                        line_str_parts.append(" ")
                    line_str_parts.append(token['text'])
                    prev_right = left_pos + token['width']  # atualiza final do token

                line_str = " ".join(line_str_parts)
                page_text.append(line_str.strip())
            page_text.append("")  # quebra de bloco

        extracted_text.append("\n".join(page_text))

    return "\n".join(extracted_text)

def split_text_in_chunks(text, max_length=60000):
    """
    Divide o texto em partes de até max_length caracteres
    para evitar problemas de inserção no Milvus.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_len = 0

    for w in words:
        w_len = len(w) + 1  # +1 para espaço
        if current_len + w_len <= max_length:
            current_chunk.append(w)
            current_len += w_len
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [w]
            current_len = w_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# ==============================
# Inicializa e Configura Embeddings e Milvus
# ==============================
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class EmbeddingFunctionWrapper:
    """Wrapper para compatibilidade com o vectorstore."""
    def __init__(self, model):
        self.model = model

    def embed_query(self, query: str):
        return self.model.encode(query)

    def embed_documents(self, documents: list[str]):
        return self.model.encode(documents)

embedding_function = EmbeddingFunctionWrapper(embedding_model)

# Cria (ou reusa) a collection no Milvus
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

def store_texts_in_milvus(text_chunks, pdf_file_name, exec_id):
    """
    Armazena cada chunk de texto no Milvus com metadados indicando
    de qual PDF e qual parte do texto veio.
    """
    for i, chunk in enumerate(text_chunks):
        metadata = {"source": pdf_file_name, "part": i, "execution_id": exec_id}
        store.add_texts([chunk], metadatas=[metadata])

# ==============================
# LLM Configuração (streaming desativado)
# ==============================
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=f"{INFERENCE_SERVER_URL}/v1",
    model_name=LLM_MODEL_NAME,
    top_p=LLM_TOP_P,
    temperature=LLM_TEMPERATURE,
    presence_penalty=LLM_PRESENCE_PENALTY,
    streaming=False,  # streaming esta como false para evitar truncamento do arquivo XML final.
    max_tokens=LLM_MAX_TOKENS,
)

def generate_xml_from_context(query, exec_id, pdf_identifier):
    """
    Faz a busca no Milvus para o pdf_identifier filtrando por exec_id,
    envia o contexto para o LLM e retorna a resposta em formato XML.
    """
    # Busca até k=1 chunk mais similar - ou k>1
    documents = store.search(
        query=query,
        k=1,
        search_type="similarity",
        filter={
            "execution_id": exec_id,
            "source": f"{pdf_identifier}.pdf"
        }
    )
    context = "\n".join([doc.page_content for doc in documents])

    messages = [
        SystemMessage(
            content=(
                "Você é um assistente financeiro avançado, experiente com profundo conhecimento em "
                "declaração de imposto de renda. Seu único objetivo é extrair informações do contexto fornecido "
                "e gerar respostas no formato XML. NUNCA interrompa uma resposta devido ao seu tamanho. "
                "Crie o XML com TODAS as informações pertinentes à sessão de bens e direitos, respeitando "
                "TODOS seus atributos e detalhes."
            )
        ),
        HumanMessage(
            content=(
                f"{context}\n\nQuais são todos os bens e direitos declarados no arquivo {pdf_identifier}? "
                "O resultado deve ser apresentado exclusivamente em XML com TODAS as características e detalhes "
                "de cada um dos bens, conforme exemplos abaixo:\n\n"
                "<?xml version=\"1.0\" ?>\n"
                "<SECTION Name=\"DECLARACAO DE BENS E DIREITOS\">\n"
                "    <TABLE>\n"
                "        <ROW No=\"1\">\n"
                "            <Field Name=\"GRUPO\" Value=\"01\"/>\n"
                "            <Field Name=\"CODIGO\" Value=\"01\"/>\n"
                "            <Field Name=\"DISCRIMINACAO\" Value=\"UT QUIS ALIQUAM LEO. DONEC ALIQUA\"/>\n"
                "            <Field Name=\"SITUACAOANTERIOR\" Value=\"23.445,00\"/>\n"
                "            <Field Name=\"SITUACAOATUAL\" Value=\"342.342,00\"/>\n"
                "            <Field Name=\"InscricaoMunicipal(IPTU)\" Value=\"23423424\"/>\n"
                "            <Field Name=\"Logradouro\" Value=\"RUA QUALQUER\"/>\n"
                "            <Field Name=\"Numero\" Value=\"89\"/>\n"
                "            <Field Name=\"Complemento\" Value=\"COMPLEM 2\"/>\n"
                "            <Field Name=\"Bairro\" Value=\"BRASILIA\"/>\n"
                "            <Field Name=\"Municipio\" Value=\"BRASÍLIA\"/>\n"
                "            <Field Name=\"UF\" Value=\"DF\"/>\n"
                "            <Field Name=\"CEP\" Value=\"1321587\"/>\n"
                "            <Field Name=\"AreaTotal\" Value=\"345,0 m²\"/>\n"
                "            <Field Name=\"DatadeAquisicao\" Value=\"12/12/1993\"/>\n"
                "            <Field Name=\"RegistradonoCartorio\" Value=\"Sim\"/>\n"
                "            <Field Name=\"NomeCartorio\" Value=\"CARTORIO DE SÇNJJKLCDF ASLK SAKÇK SAÇKLJ SAÇLKS\"/>\n"
                "            <Field Name=\"Matricula\" Value=\"2344234\"/>\n"
                "        </ROW>\n"
                "    </TABLE>\n"
                "</SECTION>\n"
                "Certifique-se de que todos os bens e direitos, com suas respectivas caracteristicas, "
                "estejam presentes no XML gerado, incluindo todos os valores, tais como SITUACAOANTERIOR "
                "e SITUACAOATUAL. Se uma informacao nao estiver disponivel no contexto, deixe o campo vazio. "
                "Use esses exemplos fornecidos como referencia ao processar as informacoes fornecidas via contexto. "
                "Os próximos itens devem conter a mesma estrutura de xml, porém podem possuir campos diferentes. "
                "Cada conjunto de GRUPO e CODIGO representam uma categoria de item diferente, com diferentes "
                "caracteristicas. TODAS AS CARACTERISTICAS SAO IMPORTANTES E DEVEM ESTAR PRESENTES NO XML, "
                "alguns exemplos incluem mas não se limitam a: RENAVAM, RegistrodeEmbarcacao, RegistrodeAeronave, "
                "CPF, CNPJ, NegociadosemBolsa, Bemoudireitopertencenteao, Titular, Conta, Agencia, Banco, e etc. "
                "Alguns desses itens podem estar presentes, ou não estar presentes de acordo com sua categoria, "
                "determinada pelo grupo e codigo, porém todos os itens devem possuir SITUACAOANTERIOR e SITUACAOATUAL "
                "informados no formato de data, conforme exemplo SITUACAO EM 31/12/2021, SITUACAO EM 31/12/2022. "
                "A data da SITUACAOANTERIOR será sempre anterior a data da SITUACAOATUAL."
            )
        )
    ]

    response = llm.predict_messages(messages)
    content = response.content.strip()

    # Remove marcadores de ```xml se existirem
    if content.startswith("```xml"):
        content = content[6:]
    if content.endswith("```"):
        content = content[:-3]

    return content

# ==============================
# MAIN
# ==============================
def main():
    # 1. Baixa PDF do MinIO
    download_file_from_minio(AWS_S3_BUCKET, object_name, file_path)

    # 2. Remove possíveis marca-d'água
    remove_watermark_advanced(file_path, output_pdf_path)

    # 3. Extrai texto via OCR
    text_extracted = extract_text_ocr(output_pdf_path)

    # 4. Divide em chunks e armazena no Milvus
    chunks = split_text_in_chunks(text_extracted, max_length=60000)
    store_texts_in_milvus(chunks, os.path.basename(output_pdf_path), execution_id)
    logging.info("PDF processado e texto armazenado no Milvus.")

    # 5. Gera consulta e obtém XML do LLM
    query_for_llm = (
        f"Quais são todos os bens e direitos do arquivo {file_identifier}?"
        " Forneça a resposta completa em XML."
    )
    xml_result = generate_xml_from_context(query_for_llm, execution_id, file_identifier)

    # 6. Salva XML local
    xml_filename = f"./{file_identifier}.xml"
    with open(xml_filename, 'w', encoding='utf-8') as f:
        f.write(xml_result)
    logging.info(f"Arquivo XML salvo localmente em {xml_filename}.")

    # 7. Sobe XML no MinIO
    upload_file_to_minio("irpf-xml", f"{file_identifier}.xml", xml_filename)

if __name__ == "__main__":
    main()