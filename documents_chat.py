import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from pydantic.v1 import ValidationError
import os


# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Dieses Dokumentenformat wird nicht unterstützt!')
        return None

    data = loader.load()
    return data


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.invoke(q)
    return answer['result']


def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.00002


def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == "__main__":
    st.subheader('Large Language Model - Q&A App')

    error_message = ''

    with st.sidebar:
        api_key = st.text_input('OpenAI API Key', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        uploaded_file = st.file_uploader('Textdatei hochladen', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Größe der Chunks', min_value=100, max_value=2048, value=512,
                                     on_change=clear_history)
        k = st.number_input('Nutzung der k ähnlichsten Chunks', min_value=1, max_value=10, value=3,
                            on_change=clear_history)
        add_data = st.button('Daten hinzufügen', on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner('Lesen, Chunking und Embedding der Datei ...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size)
                st.write(f'Größe der Chunks: {chunk_size}\nAnzahl der Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding Kosten: ${embedding_cost:.4f}\n')

                try:
                    vector_store = create_embeddings(chunks)
                    st.session_state.vs = vector_store
                    st.success('Datei erfolgreich hochgeladen')
                except ValidationError as e:
                    error_message = 'Datei konnte nicht geladen werden:\n' \
                                    'Bitte geben Sie einen gültigen OpenAI API Key an.\n\n'

    if error_message != '':
        st.write(error_message)

    q = st.text_input('Stelle eine Fragen zum Inhalt des Textdokuments:')
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            answer = ask_and_get_answer(vector_store, q, k)
            st.text_area('Antwort des LLMs: ', value=answer)

            st.write("---")
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {q} \nA: {answer}'
            st.session_state.history = f'{value} \n{"-" * 100} \n{st.session_state.history}'

            st.text_area(label='Chat Verlauf', key='history', height=400)
