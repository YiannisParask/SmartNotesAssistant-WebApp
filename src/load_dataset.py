from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from langchain_milvus import Milvus
from typing import Any
from langchain_huggingface import HuggingFaceEmbeddings


class LoadAndVectorizeData:
    def __init__(
        self, data_path: str, collection_name: str, device: str, milvus_uri: str
    ):
        """Initialize the LoadAndVectorizeData class.

        Args:
            data_path (str): Path to the directory containing documents.
            collection_name (str): Name of the Milvus collection to store vectors.
            device (str): Device to use for embeddings, e.g., 'cuda' or 'cpu'.
        """
        self.data_path = data_path
        self.collection_name = collection_name
        self.device = device
        self.milvus_uri = milvus_uri

    def load_md_data(self) -> list:
        """Load documents from the specified directory and split them into chunks.

        Args:
            data_path (str): Path to the directory containing markdown files.

        Returns:
            list: List of document chunks.
        """
        loader: Any = DirectoryLoader(
            self.data_path,
            glob="**/*.md",
            show_progress=True,
        )
        docs: list = loader.load()

        print(f"loaded {len(docs)} documents")

        return docs

    def load_pdf_data(self) -> list:
        """Load documents from a directory containing PDF files.

        Args:
            data_path (str): Path to the directory containing PDF files.

        Returns:
            list: List of loaded documents.
        """
        # Load documents from a directory containing PDF files
        loader: Any = PyPDFDirectoryLoader(path=self.data_path)
        docs: list = loader.load()

        print(f"loaded {len(docs)} documents")
        # DEBUG: Print the first 300 characters of each document
        # for doc in docs:
        #     print(f"source: {doc.metadata['source']}")
        #     print(doc.page_content[:300], "...\n")

        return docs

    def split_docs(self, docs: list) -> list:
        """Split documents into smaller chunks for vectorization.

        Args:
            docs (list): List of documents to be split.

        Returns:
            list: List of document chunks after splitting.
        """
        chunk_size: int = 512
        chunk_overlap: float = np.round(chunk_size * 0.1, 0)
        print(f"Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}")

        # Define the splitter
        text_splitter: Any = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Split the documents into smaller chunks
        chunks: list = text_splitter.split_documents(docs)
        print(f"{len(docs)} docs split into {len(chunks)} child documents.")

        return chunks

    def get_embeddings_model(self, embeddings_model) -> HuggingFaceEmbeddings:
        """Get the HuggingFace embeddings model for vectorization.

        Returns:
            HuggingFaceEmbeddings: The embeddings model.
        """
        return HuggingFaceEmbeddings(
            model_name=embeddings_model,
            # device=self.device,
        )

    def save_to_milvus(self, dict_list: list, embeddings_model) -> None:
        """Save the vectorized data to a Milvus collection using LangChain.
        Args:
            dict_list (list): List of dictionaries containing document chunks and metadata.
        """
        print("Saving to Milvus...")

        # Initialize LangChain Milvus vectorstore
        Milvus.from_documents(
            documents=dict_list,
            embedding=embeddings_model,
            collection_name=self.collection_name,
            connection_args={
                "uri": self.milvus_uri,
            },
            text_field="chunk",
            enable_dynamic_field=True,
            drop_old=True,
        )

        print(f"Inserted {len(dict_list)} vectors into Milvus.")


# def main() -> None:
#     data_path: str = "/home/yiannisparask/Projects/Personal-Cheat-Sheets"

#     docs: list = load_md_data(data_path)

#     chunks: list = slit_docs(docs)

#     embeddings_model: HuggingFaceEmbeddings = get_embeddings_model()

#     save_to_milvus(chunks, embeddings_model)


# if __name__ == "__main__":
#     main()
