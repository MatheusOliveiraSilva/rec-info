import os.path
import openai
from dotenv import load_dotenv

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# Para que as duas linhas abaixo funcionem, é necessário criar um arquivo .env na raiz do projeto.
# E adicionar o trecho de código: OPENAI_API_KEY=chave_da_api_do_openai
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Checa se já temos o index guardado no storage.
try:
    print("Tentando, carregar index...")
    # Se já tivermos o index guardado, vamos carregá-lo.
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    print("Index carregado.")

except FileNotFoundError:
    # Caso passe do if, precisamos construir o index. Para isso, vamos ler os documentos que queremos ingerir.
    print("Index não encontrado.")
    print("Carregando documentos...")
    documents = SimpleDirectoryReader("data").load_data()
    print("Documento carregado.")

    # Cria o index
    print("Criando index...")
    index = VectorStoreIndex.from_documents(documents)
    print("Index criado e salvo em ./storage")
    # Guarda o index no ./storage
    index.storage_context.persist()

# Com o index criado ou carregado, transformamos ele em um query engine.
query_engine = index.as_query_engine()

prompt = "O que é inpainting?"
print("Pergunta do usuário: " + prompt)

# Podemos agora fazer Queries usando o conhecimento do index criado a partir dos nossos documentos ingeridos.
response = query_engine.query(prompt)
print("Resposta do modelo: ")
print(response)

# Importa o módulo de avaliação do ragas
import ragevaluation as re

print("Avaliando o modelo...")
result = re.rag_evaluation(query_engine)
print(result)