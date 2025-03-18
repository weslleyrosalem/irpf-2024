import os
import sys
import logging
import uuid
import re
import fitz
import pandas as pd
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
from minio import Minio
from minio.error import S3Error
import urllib3

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Milvus
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# ---------------------------------------------------------
# Configurações de logging
# ---------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(level=LOG_LEVEL)

# ---------------------------------------------------------
# Configurações de Infra
# ---------------------------------------------------------
AWS_S3_ENDPOINT = os.getenv("MINIO_ENDPOINT", "")
AWS_ACCESS_KEY_ID = os.getenv("MINIO_ACCESS_KEY", "")
AWS_SECRET_ACCESS_KEY = os.getenv("MINIO_SECRET_KEY", "")
AWS_S3_BUCKET = os.getenv("MINIO_BUCKET", "irpf-files")

# Milvus
MILVUS_HOST = os.getenv("MILVUS_HOST", "vectordb-milvus.milvus.svc.cluster.local")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_USERNAME = os.getenv("MILVUS_USERNAME", "")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "irpf")

# LLM e inference server
INFERENCE_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://vllm.vllm.svc.cluster.local:8000")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

# Parâmetros do LLM
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.01"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.92"))
LLM_PRESENCE_PENALTY = float(os.getenv("LLM_PRESENCE_PENALTY", "1.03"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "50000"))

file_identifier = os.getenv('file_identifier')
if not file_identifier:
    logging.error("Error: file_identifier environment variable is not set.")
    sys.exit(1)

object_name = f"{file_identifier}.pdf"
file_path = f"./{file_identifier}.pdf"
output_pdf_path = f"./{file_identifier}_no_watermark.pdf"

execution_id = str(uuid.uuid4())

http_client = urllib3.PoolManager(timeout=urllib3.Timeout(connect=5.0, read=10.0))
minio_client = Minio(
    AWS_S3_ENDPOINT,
    access_key=AWS_ACCESS_KEY_ID,
    secret_key=AWS_SECRET_ACCESS_KEY,
    secure=True,
    http_client=http_client
)

def download_file_from_minio(bucket, obj_name, local_path):
    try:
        minio_client.fget_object(bucket, obj_name, local_path)
        logging.info(f"Arquivo '{obj_name}' baixado de '{bucket}' para '{local_path}' com sucesso.")
    except S3Error as e:
        logging.error(f"Erro ao baixar '{obj_name}' do bucket '{bucket}': {e}")
        sys.exit(1)

def upload_file_to_minio(bucket, obj_name, local_path):
    try:
        minio_client.fput_object(bucket, obj_name, local_path)
        logging.info(f"Arquivo '{obj_name}' enviado para '{bucket}' com sucesso.")
    except S3Error as e:
        logging.error(f"Erro ao enviar '{obj_name}' para '{bucket}': {e}")

def remove_watermark_advanced(pdf_input, pdf_output):
    try:
        doc = fitz.open(pdf_input)
        for page in doc:
            for img in page.get_images(full=True):
                xref = img[0]
                page.delete_image(xref)
            annots = page.annots()
            if annots:
                for annot in annots:
                    annot_info = annot.info
                    if "Watermark" in annot_info.get("title", ""):
                        annot.set_flags(fitz.ANNOT_HIDDEN)
            page.apply_redactions()
        doc.save(pdf_output)
        logging.info(f"Marca-d'água (se existente) removida do PDF e salvo em: {pdf_output}")
    except Exception as e:
        logging.warning(f"Erro ao remover marca-d'água do PDF: {e}")

def extract_text_ocr(pdf_path):
    if not os.path.exists(pdf_path):
        logging.error(f"Arquivo {pdf_path} não encontrado para OCR.")
        return ""
    extracted_text = []
    try:
        pages = convert_from_path(pdf_path)
    except Exception as e:
        logging.error(f"Erro ao converter PDF em imagens: {e}")
        return ""

    for i, page_image in enumerate(pages, start=1):
        logging.info(f"Processando OCR na página {i} do PDF...")
        ocr_config = r'-c preserve_interword_spaces=1 --oem 1 --psm 1'
        ocr_data = pytesseract.image_to_data(page_image, lang='por', config=ocr_config, output_type=Output.DICT)
        df_ocr = pd.DataFrame(ocr_data)
        df_clean = df_ocr[(df_ocr.conf != '-1') & (df_ocr.text.str.strip() != "")]
        if df_clean.empty:
            continue
        df_text_len = df_clean[df_clean.text.str.len() > 3]
        avg_char_width = (df_text_len.width / df_text_len.text.str.len()).mean() if not df_text_len.empty else 5.0

        block_nums_sorted = df_clean.groupby('block_num').first().sort_values('top').index.tolist()
        page_text = []
        for block_num in block_nums_sorted:
            block_data = df_clean[df_clean['block_num'] == block_num]
            line_nums_sorted = block_data.groupby('line_num').first().sort_values('top').index.tolist()
            for line_num in line_nums_sorted:
                line_data = block_data[block_data['line_num'] == line_num].sort_values('left')
                line_str_parts = []
                prev_right = 0
                for _, token in line_data.iterrows():
                    left_pos = token['left']
                    gap = left_pos - prev_right
                    if gap > avg_char_width:
                        line_str_parts.append(" ")
                    line_str_parts.append(token['text'])
                    prev_right = left_pos + token['width']
                line_str = " ".join(line_str_parts)
                page_text.append(line_str.strip())
            page_text.append("")
        extracted_text.append("\n".join(page_text))
    return "\n".join(extracted_text)

def split_text_in_chunks(text, max_length=10000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_len = 0
    for w in words:
        w_len = len(w) + 1
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

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class EmbeddingFunctionWrapper:
    def __init__(self, model):
        self.model = model
    def embed_query(self, query: str):
        return self.model.encode(query)
    def embed_documents(self, documents: list[str]):
        return self.model.encode(documents)

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

def store_texts_in_milvus(text_chunks, pdf_file_name, exec_id):
    for i, chunk in enumerate(text_chunks):
        metadata = {"source": pdf_file_name, "part": i, "execution_id": exec_id}
        store.add_texts([chunk], metadatas=[metadata])

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=f"{INFERENCE_SERVER_URL}/v1",
    model_name=LLM_MODEL_NAME,
    top_p=LLM_TOP_P,
    temperature=LLM_TEMPERATURE,
    presence_penalty=LLM_PRESENCE_PENALTY,
    streaming=False,
    max_tokens=LLM_MAX_TOKENS,
)

def extract_partial_xml_from_chunk(chunk_text: str) -> str:
    messages = [
        SystemMessage(
            content=(
                "Você é um assistente financeiro avançado, especialista em declaração de imposto de renda e análise de documentos PDF. "
                "Seu objetivo é extrair com precisão todas as informações referentes aos bens e direitos declarados, bem como os bens da atividade rural, "
                "organizando os dados em um XML estruturado. O XML deve conter exatamente três seções, na seguinte ordem: "
                "'DECLARACAO DE BENS E DIREITOS', 'BENS DA ATIVIDADE RURAL - BRASIL' e 'BENS DA ATIVIDADE RURAL - EXTERIOR'.\n\n"
                "Para cada item identificado, inclua os campos obrigatórios: GRUPO, CODIGO, DISCRIMINACAO, SITUACAOANTERIOR e SITUACAOATUAL. "
                "Adicionalmente, insira somente os campos específicos que correspondem ao tipo de item extraído, por exemplo:\n"
                "  - Para imóveis: InscricaoMunicipal(IPTU), Logradouro, Numero, Complemento, Bairro, Municipio, UF, CEP, "
                "AreaTotal, DatadeAquisicao, RegistradonoCartorio, NomeCartorio e Matricula.\n"
                "  - Para automóveis: Renavam.\n"
                "  - Para embarcações: RegistrodeEmbarcacao.\n"
                "  - Para outros tipos, inclua os campos pertinentes conforme as informações extraídas.\n\n"
#                "Caso alguma informação não esteja disponível, deixe o valor do campo em branco. "
                "Importante: NÃO atribua automaticamente '105' como GRUPO para a seção 'DECLARACAO DE BENS E DIREITOS'. "
                "Cada item deve ter seu GRUPO e CODIGO corretos conforme os dados originais do documento, sem assumir '105 - Brasil' para todos, "
                "a menos que o próprio documento o indique. Caso alguma informação não esteja disponível, deixe o valor do campo em branco. "
                "As datas devem estar no formato DD/MM/AAAA (ex.: 31/12/2021) e SITUACAOANTERIOR deve ser sempre anterior a SITUACAOATUAL."            
                "NUNCA interrompa a resposta por conta do tamanho e não utilize o mesmo conjunto de campos para todos os itens; "
                "cada item deve refletir suas características específicas conforme extraídas do contexto.\n\n"
            )
        ),
        HumanMessage(
            content=(
                f"{chunk_text}\n\n"
                "Com base no texto acima, extraia todas as informações referentes aos bens/direitos e aos bens da atividade rural, "
                "gerando um XML que contenha apenas as seguintes seções: "
                "'DECLARACAO DE BENS E DIREITOS', 'BENS DA ATIVIDADE RURAL - BRASIL' e 'BENS DA ATIVIDADE RURAL - EXTERIOR'.\n\n"
                "Utilize o formato XML abaixo como referência, mas adapte-o conforme os dados extraídos e as características específicas de cada item:\n\n"
                "<?xml version=\"1.0\" ?>\n"
                "<XML_FINAL>\n"
                "    <SECTION Name=\"DECLARACAO DE BENS E DIREITOS\">\n"
                "         <TABLE>\n"
                "              <ROW No=\"1\">\n"
                "                   <Field Name=\"GRUPO\" Value=\"...\"/>\n"
                "                   <Field Name=\"CODIGO\" Value=\"...\"/>\n"
                "                   <Field Name=\"DISCRIMINACAO\" Value=\"...\"/>\n"
                "                   <Field Name=\"SITUACAOANTERIOR\" Value=\"...\"/>\n"
                "                   <Field Name=\"SITUACAOATUAL\" Value=\"...\"/>\n"
                "                   <!-- Insira aqui os campos específicos conforme o tipo de item (ex.: para imóveis ou automóveis) -->\n"
                "              </ROW>\n"
                "         </TABLE>\n"
                "    </SECTION>\n"
                "    <SECTION Name=\"BENS DA ATIVIDADE RURAL - BRASIL\">\n"
                "         <TABLE>\n"
                "              <!-- Itens com campos específicos para bens rurais no Brasil -->\n"
                "         </TABLE>\n"
                "    </SECTION>\n"
                "    <SECTION Name=\"BENS DA ATIVIDADE RURAL - EXTERIOR\">\n"
                "         <TABLE>\n"
                "              <!-- Itens com campos específicos para bens rurais no exterior -->\n"
                "         </TABLE>\n"
                "    </SECTION>\n"
                "</XML_FINAL>\n"
            )
        )
    ]
    try:
        response = llm.predict_messages(messages)
        content = response.content.strip()
    except Exception as e:
        logging.warning(f"Falha ao chamar LLM para chunk. Retornando vazio. Erro: {e}")
        return ""
    
    if content.startswith("```xml"):
        content = content[6:]
    if content.endswith("```"):
        content = content[:-3]
    
    return content


# ----------------------------------------------------------------
# Parse e acumula <ROW> em dicionários
# ----------------------------------------------------------------
def parse_and_accumulate_sections(xml_str: str, section_rows: dict):
    """
    Lê cada <SECTION Name="X"> e suas <ROW>... e acumula no dicionário section_rows,
    que é algo como:
       {
         "DECLARACAO DE BENS E DIREITOS": [ "<ROW>...</ROW>", "<ROW>...</ROW>" ],
         "BENS DA ATIVIDADE RURAL - BRASIL": [...],
         "BENS DA ATIVIDADE RURAL - EXTERIOR": [...]
       }
    """
    # Capturar blocos de SECTION
    sec_pattern = re.compile(
        r'<SECTION\s+Name="([^"]+)">\s*<TABLE>\s*(.*?)\s*</TABLE>\s*</SECTION>',
        flags=re.DOTALL | re.IGNORECASE
    )
    # Dentro do bloco de TABLE, capturar <ROW ...>...</ROW>
    row_pattern = re.compile(r'(<ROW\s+.*?</ROW>)', flags=re.DOTALL | re.IGNORECASE)

    # Identifica se a section name está em section_rows. Se sim, adiciona as ROWs.

    for match in sec_pattern.finditer(xml_str):
        sec_name = match.group(1).strip()
        table_content = match.group(2)  

        # Filtra sec_name
        valid_names = {
            "DECLARACAO DE BENS E DIREITOS",
            "BENS DA ATIVIDADE RURAL - BRASIL",
            "BENS DA ATIVIDADE RURAL - EXTERIOR"
        }
        if sec_name.upper() in (v.upper() for v in valid_names):
            # extrair ROWs
            rows_found = row_pattern.findall(table_content)
            
            section_rows[sec_name].extend(rows_found)

def generate_final_xml(section_rows: dict) -> str:
    """
    A partir das listas de ROWs no section_rows,
    gera um único <SECTION> por tipo.
    """
    final = ['<?xml version="1.0" ?>', '<XML_FINAL>']

    for sec_name in ["DECLARACAO DE BENS E DIREITOS",
                     "BENS DA ATIVIDADE RURAL - BRASIL",
                     "BENS DA ATIVIDADE RURAL - EXTERIOR"]:
        rows = section_rows[sec_name]
        final.append(f'<SECTION Name="{sec_name}">')
        final.append('    <TABLE>')
        if rows:
            for r in rows:
                final.append(r)
        else:

            pass
        final.append('    </TABLE>')
        final.append('</SECTION>')

    final.append('</XML_FINAL>')
    merged = "\n".join(final)
    return escape_xml_content(merged)

def escape_xml_content(xml_str: str) -> str:
    """
    Escapa caracteres especiais no conteúdo XML, para evitar
    erros de parser no navegador.
    """
    pattern_value = re.compile(r'(Value=")(.*?)(")')

    def replacer(match):
        prefix = match.group(1)
        content = match.group(2)
        suffix = match.group(3)
        escaped = content
        escaped = escaped.replace("&", "&amp;")
        escaped = escaped.replace("\"", "&quot;")
        escaped = escaped.replace("<", "&lt;")
        escaped = escaped.replace(">", "&gt;")
        escaped = escaped.replace("'", "&apos;")
        return f'{prefix}{escaped}{suffix}'

    result = pattern_value.sub(replacer, xml_str)
    return result


def main():
    # 1. Baixa PDF do MinIO
    download_file_from_minio(AWS_S3_BUCKET, object_name, file_path)
    # 2. Remove marca-d'água
    remove_watermark_advanced(file_path, output_pdf_path)
    # 3. Extrai texto via OCR
    text_extracted = extract_text_ocr(output_pdf_path)

    # 4. Divide em chunks e armazena no Milvus
    text_chunks = split_text_in_chunks(text_extracted, max_length=40000)
    store_texts_in_milvus(text_chunks, os.path.basename(output_pdf_path), execution_id)
    logging.info("PDF processado e texto armazenado no Milvus.")

    # *Estrutura* para acumular as ROWs de cada seção
    section_rows = {
        "DECLARACAO DE BENS E DIREITOS": [],
        "BENS DA ATIVIDADE RURAL - BRASIL": [],
        "BENS DA ATIVIDADE RURAL - EXTERIOR": []
    }

    # 5. Extrai XML parcial de cada chunk
    for i, chunk in enumerate(text_chunks):
        logging.info(f"Extraindo XML parcial do chunk {i+1}/{len(text_chunks)}...")
        partial_xml = extract_partial_xml_from_chunk(chunk)
        # 5.1. Faz parse do partial_xml e acumula
        parse_and_accumulate_sections(partial_xml, section_rows)

    # 6. Gera o XML final, unificando as ROWs de cada seção
    final_xml = generate_final_xml(section_rows)

    # 7. Salva
    xml_filename = f"./{file_identifier}.xml"
    with open(xml_filename, 'w', encoding='utf-8') as f:
        f.write(final_xml)
    logging.info(f"Arquivo XML salvo localmente em {xml_filename}.")

    # 8. Sobe no MinIO
    upload_file_to_minio("irpf-xml", f"{file_identifier}.xml", xml_filename)

if __name__ == "__main__":
    main()
