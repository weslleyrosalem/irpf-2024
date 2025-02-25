import os
import sys
import uuid
import logging

# (Optional) If you need to auto-install packages:
# import subprocess
# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# dependencies = [
#     "sentence-transformers", "pymilvus", "openai==0.28", "langchain", 
#     "langchain_community", "minio", "pymupdf", "Pillow", "pytesseract", 
#     "pandas", "langchain-milvus", "opencv-python", "pdf2image", "vllm"
# ]
# for dep in dependencies:
#     install(dep)

# Logging
logging.basicConfig(level=logging.DEBUG)

import fitz  # PyMuPDF
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

# -----------------------------
# CONFIG SECTION
# -----------------------------
AWS_S3_ENDPOINT = "minio-api-minio.apps.cluster-lqsm2.lqsm2.sandbox441.opentlc.com"
AWS_ACCESS_KEY_ID = "minio"
AWS_SECRET_ACCESS_KEY = "minio123"
AWS_S3_BUCKET = "irpf-files"

MILVUS_HOST = "vectordb-milvus.milvus.svc.cluster.local"
MILVUS_PORT = 19530
MILVUS_USERNAME = "root"
MILVUS_PASSWORD = "Milvus"
MILVUS_COLLECTION = "irpf"

INFERENCE_SERVER_URL = "http://vllm.vllm.svc.cluster.local:8000"
LLM_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# If you need Tesseract in Portuguese, 
# ensure your environment has the 'por' language data installed.

# -----------------------------
# SETUP
# -----------------------------
http_client = urllib3.PoolManager(
    timeout=urllib3.Timeout(connect=5.0, read=10.0)
)

# MinIO client
client = Minio(
    AWS_S3_ENDPOINT,
    access_key=AWS_ACCESS_KEY_ID,
    secret_key=AWS_SECRET_ACCESS_KEY,
    secure=True,
    http_client=http_client
)

# The file identifier (if not set by environment, define manually)
file_identifier = os.getenv('file_identifier', "joao-2023")
if not file_identifier:
    print("Error: file_identifier environment variable is not set.")
    sys.exit(1)

execution_id = str(uuid.uuid4())

# Download from MinIO
object_name = f"{file_identifier}.pdf"
file_path = f"./{file_identifier}.pdf"
output_pdf_path = f"./{file_identifier}_no_watermark.pdf"

try:
    client.fget_object(AWS_S3_BUCKET, object_name, file_path)
    print(f"'{object_name}' successfully downloaded to '{file_path}'.")
except S3Error as e:
    print("Error occurred:", e)
    sys.exit(1)

# -----------------------------
# WATERMARK REMOVAL
# -----------------------------
def remove_watermark_advanced(pdf_path, output_path):
    """
    Removes images from each page (which might remove content if 
    the PDF is entirely image-based). Also hides any annotation
    titled 'Watermark'. Adjust as needed!
    """
    doc = fitz.open(pdf_path)
    for page in doc:
        # Remove all images on the page
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            page.delete_image(xref)

        # If there's an annotation with title "Watermark", hide it
        annots = page.annots()
        if annots:
            for annot in annots:
                annot_info = annot.info
                if "Watermark" in annot_info.get("title", ""):
                    annot.set_flags(fitz.ANNOT_HIDDEN)

        page.apply_redactions()

    doc.save(output_path)
    print(f"Watermark removed: {output_path}")

# Comment out if removing all images kills your content
remove_watermark_advanced(file_path, output_pdf_path)


# -----------------------------
# OCR EXTRACTION
# -----------------------------
def extract_text_ocr(pdf_path):
    """
    Convert pages to images and run Tesseract. 
    If the PDF is actually text-based, you'd likely get better results
    with doc = fitz.open(pdf_path); doc[i].get_text() directly.
    """
    extracted_text = ""
    if not os.path.exists(pdf_path):
        print(f"The file {pdf_path} was not found.")
        return ""

    try:
        pages = convert_from_path(pdf_path)
        for i, page_img in enumerate(pages):
            print(f"Processing page {i+1}/{len(pages)} of PDF via OCR...")

            ocr_config = r'-c preserve_interword_spaces=1 --oem 1 --psm 1'
            ocr_data = pytesseract.image_to_data(
                page_img, 
                lang='por', 
                config=ocr_config, 
                output_type=Output.DICT
            )
            df = pd.DataFrame(ocr_data)
            # remove blocks with confidence -1 or blank text
            df = df[(df.conf != '-1') & (df.text.str.strip() != '')]
            if df.empty:
                continue

            # group by block_num so we can reconstruct in reading order
            block_nums = df.groupby('block_num').first().sort_values('top').index.tolist()

            for block_num in block_nums:
                block_df = df[df['block_num'] == block_num]
                # optional: skip extremely short tokens
                filtered = block_df[block_df.text.str.len() > 2]

                if filtered.empty:
                    continue

                avg_char_width = (filtered.width / filtered.text.str.len()).mean()
                prev_par, prev_line, prev_left = 0, 0, 0

                for idx, line_data in block_df.iterrows():
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
                    left_margin_chars = int(line_data['left']/avg_char_width) if avg_char_width else 0
                    if left_margin_chars > prev_left + 1:
                        spaces_to_add = left_margin_chars - prev_left
                        extracted_text += ' ' * spaces_to_add

                    extracted_text += line_data['text'] + ' '
                    prev_left += len(line_data['text']) + spaces_to_add + 1

                extracted_text += '\n'
        return extracted_text
    except Exception as e:
        print(f"Error processing PDF for OCR: {e}")
        return ""


# -----------------------------
# MILVUS SETUP
# -----------------------------
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
        print("No text parts found! Skipping Milvus insertion.")
        return

    print(f"Number of text_parts: {len(text_parts)}")
    for i, part in enumerate(text_parts):
        metadata = {"source": pdf_file, "part": i, "execution_id": execution_id}
        print(f"Inserting chunk #{i}, length {len(part)} chars")
        store.add_texts([part], metadatas=[metadata])


# -----------------------------
# LLM SETUP & QUERY
# -----------------------------
llm = ChatOpenAI(
    openai_api_key="EMPTY",
    openai_api_base=f"{INFERENCE_SERVER_URL}/v1",
    model_name=LLM_MODEL_NAME,
    top_p=0.92,
    temperature=0.01,
#    max_tokens=1024,
    presence_penalty=1.03,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

def query_information(query, execution_id, file_identifier):
    # We'll search for docs in Milvus that match the "execution_id" + "source"
    # source is expected to be file_identifier_no_watermark.pdf if you used 
    #    os.path.basename(output_pdf_path)
    # but if you want it to be strictly "file_identifier.pdf", 
    # make sure that is how you set metadata "source" above.
    # In the code above, we do: "source": pdf_file
    # and pdf_file is "18_01782176543_2023_2024_no_watermark.pdf"
    # so you should adjust accordingly if needed.

    documents = store.search(
        query=query, 
        k=1, 
        search_type="similarity", 
        filter={
            "execution_id": execution_id, 
            "source": f"{file_identifier}_no_watermark.pdf"  # or .pdf if that is your stored metadata
        }
    )

    if not documents:
        print("No docs found in Milvus for that query/filter. Returning empty answer.")
        return "No context found."

    context_parts = [doc.page_content for doc in documents]
    context_combined = "\n".join(context_parts)

    messages = [
    SystemMessage(content=(
        "Você é um assistente financeiro avançado, experiente com profundo conhecimento "
        "em declaração de imposto de renda. Seu único objetivo é extrair informações do "
        "contexto fornecido e gerar respostas no formato XML. NUNCA interrompa uma resposta "
        "devido ao seu tamanho. Crie o XML com TODAS as informações pertinentes à sessão de "
        "bens e direitos, respeitando TODOS seus atributos e todos seus detalhes."
    )),

    HumanMessage(content=(
        f"{context_combined}\n\n"
        "Quais são todos os bens e direitos declarados no arquivo "
        f"{file_identifier}?"
        " O resultado deve ser apresentado exclusivamente em XML com TODAS as características "
        "e detalhes de cada um dos bens, conforme exemplos abaixo:\n\n"
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
        "</SECTION>\n\n"
        "Certifique-se de que todos os bens e direitos, com suas respectivas caracteristicas, "
        "estejam presentes no XML gerado, incluindo todos os valores, tais como SITUACAOANTERIOR "
        "e SITUACAOATUAL. Se uma informacao nao estiver disponivel no contexto, deixe o campo vazio. "
        "Use esse exemplo fornecido como referencia ao processar as informacoes fornecidas via contexto, "
        "os próximos itens devem conter a mesma estrutra de xml, porem podem possuir campos diferentes, "
        "cada conjunto de GRUPO e CODIGO representam uma categoria de item diferente, com diferentes "
        "caracteristicas. TODAS AS CARACTERISTICAS SAO IMPORTANTES E DEVEM ESTAR PRESENTES NO XML, "
        "alguns exemplos incluem mas nao se limitam a: RENAVAM, RegistrodeEmbarcacao, RegistrodeAeronave, "
        "CPF, CNPJ, NegociadosemBolsa, Bemoudireitopertencenteao, Titular, Conta, Agencia, Banco, e etc. "
        "alguns desses itens podem estar presentes, ou não estar presentes de acordo com sua categoria, "
        "determinada pelo grupo e codigo, porem todos os itens devem possuir SITUACAOANTERIOR e "
        "SITUACAOATUAL informados no formato de data, conforme exemplo SITUACAO EM 31/12/2021 , "
        "SITUACAO EM 31/12/2022. a data da SITUACAOANTERIOR sera sempre anterior a data da SITUACAOATUAL, "
        "o que nao significa ser a data de hoje, porem mais recente que a data anterior.\n"
    ))
]


    response = llm.predict_messages(messages)
    return response.content

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    # 1) OCR Extract
    text = extract_text_ocr(output_pdf_path)
    print(f"\nExtracted text length: {len(text)} characters.\n")

    # 2) Split and store in Milvus
    text_parts = split_text(text)
    store_text_parts_in_milvus(text_parts, os.path.basename(output_pdf_path), execution_id)
    print("\nPDF processed and stored in Milvus successfully.")

    # 3) Query LLM
    query = (
      f"Quais são todos os bens e direitos suas informações, seus atributos e detalhes, "
      f"declarados no arquivo {file_identifier}? Não interrompa a resposta devido ao seu tamanho, "
      f"forneça a resposta com todo o contexto que tiver. "
      f"Valores monetarios devem estar em portugues do brasil, moeda real brl."
    )
    llm_result = query_information(query, execution_id, file_identifier)

    # 4) Clean up possible markdown fences
    result = llm_result.strip()
    if result.startswith("```xml"):
        result = result[6:]
    if result.endswith("```"):
        result = result[:-3]

    # 5) Save local XML
    xml_file_name = f"./{file_identifier}.xml"
    with open(xml_file_name, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"File {xml_file_name} has been successfully saved.")

    # 6) Upload XML to MinIO
    upload_object_name = f"{file_identifier}.xml"
    bucket_name = "irpf-xml"
    try:
        client.fput_object(bucket_name, upload_object_name, xml_file_name)
        print(f"'{upload_object_name}' successfully uploaded to bucket '{bucket_name}'.")
    except S3Error as e:
        print("Error occurred while uploading XML:", e)

if __name__ == "__main__":
    main()