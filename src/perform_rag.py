from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_milvus import Milvus
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import Any
from transformers.pipelines import pipeline


class RagSearch:
    def __init__(self, milvus_uri: str, device: str, collection_name: str):
        """Initialize the RagSearch class.

        Args:
            milvus_uri (str): URI for the Milvus database.
            device (str): Device to use for embeddings, e.g., 'cuda' or 'cpu'.
            collection_name (str): Name of the Milvus collection to store vectors.
        """
        self.collection_name: str = collection_name
        self.milvus_uri: str = milvus_uri
        self.device: str = device
        self.text_generator: Any = None  # or get_vllm()
        self.embeddings_generator: Any = None


    def get_embeddings_model(self, embed_model_name: str) -> Any:
        """Download and initialize the HuggingFace embeddings model.

        Returns:
            HuggingFaceEmbeddings: An instance of HuggingFaceEmbeddings with the specified model.
        """
        if self.embeddings_generator is not None:
            return self.embeddings_generator
        else:
            self.embeddings_generator = HuggingFaceEmbeddings(
                model_name=embed_model_name,
            )
            return self.embeddings_generator


    def get_retriever(self) -> Any:
        """Create a Milvus vectorstore retriever using LangChain and return it.

        Args:
            uri (str): URI for the Milvus database.
            embeddings (SentenceTransformerEmbeddings): Embedding function to use.
            collection_name (str): Name of the Milvus collection.
            k (int): Number of top results to return.

        Returns:
            Any: A retriever object that can be used to query the Milvus vectorstore.
        """
        vectorstore = Milvus(
            embedding_function=self.embeddings_generator,
            collection_name=self.collection_name,
            connection_args={
                "uri": self.milvus_uri,
            },
            text_field="chunk",
        )
        top_k: int = 5
        return vectorstore.as_retriever(search_kwargs={"k": top_k})


    def build_prompt_template(self) -> PromptTemplate:
        """Create a PromptTemplate for RAG QA."""
        template = (
            "First, check if the provided Context is relevant to the user's question.\n"
            "Second, only if the provided Context is strongly relevant, answer the question using the Context.\n"
            "Otherwise, if the Context is not strongly relevant, answer the question without using the Context.\n"
            "Be clear and concise.\n\n"
            "### Context:\n{context}\n\n"
            "### Question:\n{question}\n\n"
            "### Answer:"
        )
        return PromptTemplate(
            input_variables=["context", "question"], template=template
        )


    def get_hg_llm(self, llm_model_name: str) -> Any:
        """Instantiate the HuggingFace LLM wrapper."""
        if self.text_generator is not None:
            return self.text_generator
        else:
            llm_pipeline = pipeline(task="text-generation", model=llm_model_name)
            self.text_generator = HuggingFacePipeline(pipeline=llm_pipeline)
            return self.text_generator


    def get_qa_chain(self, retriever, prompt: PromptTemplate) -> RetrievalQA:
        """Build and return a RetrievalQA chain with custom prompt.

        Args:
            llm (Any): The language model to use (HuggingFace or VLLM).
            retriever: The retriever object for fetching relevant documents.
            prompt (PromptTemplate): The prompt template to use for the chain.

        Returns:
            RetrievalQA: A RetrievalQA chain configured with the provided LLM, retriever, and prompt.
        """
        return RetrievalQA.from_chain_type(
            llm=self.text_generator,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )


    def perform_rag_query(self, chain: RetrievalQA, query: str) -> str:
        """Run the RetrievalQA chain on the given query.

        Args:
            chain (RetrievalQA): The RetrievalQA chain to use.
            query (str): The user's query string.

        Returns:
            str: The answer string.
        """
        result = chain.invoke({"query": query})
        answer_full = result["result"].strip()

        # Extract only the part after '### Answer:' if present
        answer = answer_full
        if "### Answer:" in answer_full:
            answer = answer_full.split("### Answer:", 1)[-1].strip()

        return answer
