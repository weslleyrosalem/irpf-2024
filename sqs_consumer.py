import os
import json
import random
import string
import boto3
from kfp import Client

# Configuração do SQS
SQS_QUEUE_URL = 'https://sqs.sa-east-1.amazonaws.com/123456789012/minha-fila'
AWS_REGION = 'sa-east-1'

# Configuração do KFP client
KFP_HOST = 'https://ds-pipeline-dspa-safra-ai.apps.rosa-5hxrw.72zm.p1.openshiftapps.com'
KFP_TOKEN = ''
PIPELINE_FILE = 'rhoai-pdf-to-xml.yaml'

# Inicializa o cliente SQS
sqs_client = boto3.client('sqs', region_name=AWS_REGION)

def generate_random_run_name(base_name):
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{base_name}-{random_suffix}"

def submit_pipeline(file_identifier):
    # Configura os argumentos do pipeline
    pipeline_arguments = {
        'file_identifier': file_identifier
    }
    # Cria o cliente KFP
    client = Client(
        host=KFP_HOST,
        existing_token=KFP_TOKEN
    )
    # Submete o pipeline com os argumentos
    run_name = generate_random_run_name("pdf-to-xml-safra-irpf")
    experiment_name = "pdf-to-xml-safra-irpf"
    result = client.create_run_from_pipeline_package(
        pipeline_file=PIPELINE_FILE,
        arguments=pipeline_arguments,
        run_name=run_name,
        experiment_name=experiment_name
    )
    # Saída do resultado
    print(result)

def consume_messages():
    while True:
        # Recebe mensagens da fila SQS
        response = sqs_client.receive_message(
            QueueUrl=SQS_QUEUE_URL,
            MaxNumberOfMessages=10,  # Número máximo de mensagens a serem recebidas
            WaitTimeSeconds=20  # Long polling
        )

        messages = response.get('Messages', [])
        if not messages:
            print("Nenhuma mensagem recebida. Continuando...")
            continue

        for message in messages:
            try:
                # Processa a mensagem
                record = json.loads(message['Body'])
                # Extrai o nome do arquivo sem extensão
                file_identifier = os.path.splitext(record.get('Key').split('/')[-1])[0]
                print(f'Received message: {record}')
                print(f'File Identifier: {file_identifier}')
                # Submete o pipeline com o identificador do arquivo
                submit_pipeline(file_identifier)
                # Exclui a mensagem da fila após o processamento
                sqs_client.delete_message(
                    QueueUrl=SQS_QUEUE_URL,
                    ReceiptHandle=message['ReceiptHandle']
                )
            except Exception as e:
                print(f"Erro ao processar mensagem: {e}")
                # Aqui você pode implementar lógica de tratamento de erros, como mover a mensagem para uma fila de erros

if __name__ == "__main__":
    consume_messages()
