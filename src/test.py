from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from config import *
import json
import math


TEST_CASES = [
    {
        "question": "How do I install vLLM?",
        "ground_truth": "You can install vLLM with uv or pip: uv pip install vllm"
    },
    {
        "question": "What is vLLM and where was it originally developed?",
        "ground_truth": "vLLM is a fast and easy-to-use library for LLM inference and serving. It was originally developed in the Sky Computing Lab at UC Berkeley and has grown into one of the most active open-source AI projects, maintained by a community of over 2000 contributors from dozens of academic institutions and companies."
    },
    {
        "question": "What types of distributed inference parallelism does vLLM support?",
        "ground_truth": "vLLM supports tensor, pipeline, data, expert, and context parallelism for distributed inference."
    },
    {
        "question": "What quantization formats does vLLM support?",
        "ground_truth": "vLLM supports FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and more."
    },
    {
        "question": "What hardware platforms does vLLM support beyond NVIDIA GPUs?",
        "ground_truth": "vLLM supports AMD GPUs, x86/ARM/PowerPC CPUs, and additional hardware such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, and MetaX GPU, among others."
    },
    {
        "question": "What model architectures does vLLM support?",
        "ground_truth": "vLLM supports over 200 model architectures on HuggingFace, including decoder-only LLMs (Llama, Qwen, Gemma), Mixture-of-Experts LLMs (Mixtral, DeepSeek-V3), hybrid attention and state-space models (Mamba, Qwen3.5), multi-modal models (LLaVA, Qwen-VL, Pixtral), embedding and retrieval models (E5-Mistral, GTE, ColBERT), and reward and classification models (Qwen-Math)."
    },
    {
        "question": "How should security vulnerabilities in vLLM be reported?",
        "ground_truth": "Security vulnerabilities should be reported privately using GitHub's vulnerability submission form. Reports will then be triaged by the vulnerability management team."
    },
    {
        "question": "What are the security severity categories defined in vLLM and what CVSS scores correspond to each?",
        "ground_truth": "There are four categories: CRITICAL (CVSS >= 9.0) for remote code execution without interaction; HIGH (CVSS 7.0-8.9) for serious flaws requiring advanced conditions; MODERATE (CVSS 4.0-6.9) for denial of service or partial disruption; and LOW (CVSS < 4.0) for minor issues like informational disclosures or non-exploitable flaws."
    },
    {
        "question": "What requirements must an organization meet to join the vLLM security prenotification group?",
        "ground_truth": "An organization must meet at least one of these criteria: substantial internal deployment of the upstream vLLM project, established internal security teams and comprehensive compliance measures, or active and consistent contributions to the upstream vLLM project."
    },
    {
        "question": "What types of benchmarks are included in the benchmarks/ directory of the vLLM repository?",
        "ground_truth": "The directory contains serving benchmarks (for online inference performance: latency and throughput), throughput benchmarks (for offline batch inference), specialized benchmarks (for structured output, prefix caching, long document QA, request prioritization, and multi-modal inference), and dataset utilities (for loading and sampling from ShareGPT, HuggingFace datasets, synthetic data, etc.)."
    },
    {
        "question": "What hardware platforms does the continuous performance benchmarking suite in vLLM cover?",
        "ground_truth": "The benchmarking suite covers latency, throughput, and fixed-QPS serving on B200, A100, H100, Intel Xeon Processors, Intel Gaudi 3 Accelerators, and Arm Neoverse processors, across different models."
    },
    {
        "question": "What email should be used for collaborations and partnerships with the vLLM project?",
        "ground_truth": "For collaborations and partnerships, you should contact collaboration@vllm.ai."
    },
    {
        "question": "What is the reference paper for citing vLLM in research, and what is the key mechanism it introduces?",
        "ground_truth": "The reference paper is 'Efficient Memory Management for Large Language Model Serving with PagedAttention' by Kwon et al. (2023), published in the ACM SIGOPS 29th Symposium on Operating Systems Principles. The key mechanism introduced is PagedAttention, which enables efficient management of attention key and value memory."
    },
    {
        "question": "What communication channels exist for different types of queries about vLLM?",
        "ground_truth": "The channels are: GitHub Issues for technical questions and feature requests, the vLLM Forum (discuss.vllm.ai) for user discussions, Slack (slack.vllm.ai) for coordinating contributions and development, GitHub Security Advisories for security disclosures, and collaboration@vllm.ai for collaborations and partnerships."
    },
    {
        "question": "When is the release branch cut performed for major and minor releases in vLLM?",
        "ground_truth": "For major and minor releases, the branch cut is performed 1-2 days before the release goes live. For patch releases, the previously cut release branch is reused."
    },
    {
        "question": "What types of cherry-picks are allowed after the branch cut in the vLLM release process?",
        "ground_truth": "Allowed cherry-picks include: regression fixes against the most recent release, critical fixes for severe issues such as silent incorrectness, backwards compatibility breaks, crashes, deadlocks, or large memory leaks, fixes to new features introduced in the most recent release, and release branch specific changes like version identifier updates or CI fixes. No feature work is allowed."
    },
    {
        "question": "What server APIs does vLLM support beyond the OpenAI-compatible API?",
        "ground_truth": "In addition to the OpenAI-compatible API server, vLLM also supports the Anthropic Messages API and gRPC."
    },
]


def load_vectorstore():
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )


def load_llm():
    return ChatOllama(model=OLLAMA_MODEL, temperature=0, num_predict=1000) #cambiar a 0.1 si falla


def build_rag_chain(db, llm):
    all_data = db.get()
    documents = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_data["documents"], all_data["metadatas"])
    ]

    semantic_retriever = db.as_retriever(search_kwargs={"k": 4})
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 4

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, semantic_retriever])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict RAG assistant.

        You MUST follow these rules:

        - Use ONLY the provided context.
        - If the answer is not explicitly in the context, reply EXACTLY: I don't know
        - Do NOT guess, infer, or use prior knowledge.
        - Do NOT redefine terms or expand acronyms.
        - Answer in MAX 3 sentenceS.
        - Be concise and technical.

        Any violation of these rules is incorrect."""),

            ("human", """Context:
        {context}

        Question:
        {input}

        Answer (max 200 chars):""")
    ])

    docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(ensemble_retriever, docs_chain)


def collect_results(chain, test_cases):
    questions, answers, contexts, ground_truths = [], [], [], []

    print(f"\nRunning {len(test_cases)} test cases...\n")

    for i, case in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {case['question']}")
        
        result = chain.invoke({"input": case["question"]})

        questions.append(case["question"])
        answers.append(result["answer"])
        contexts.append([doc.page_content for doc in result["context"]])
        ground_truths.append(case["ground_truth"])

        print(f"  Answer: {result['answer'][:100]}...")

    return questions, answers, contexts, ground_truths


def run_ragas_evaluation(questions, answers, contexts, ground_truths, llm, embeddings):
    print("\nRunning RAGAS evaluation...")

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    result = evaluate(
        dataset,
        metrics=[faithfulness, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    return result


def aggregate_scores(ragas_result):
    """Aggregate per-sample scores (list of dicts) into metric averages."""
    rows = ragas_result.scores  # list of dicts, one per sample
    aggregated = {}
    if rows:
        for key in rows[0].keys():
            vals = [
                r[key] for r in rows
                if isinstance(r.get(key), (int, float)) and not math.isnan(r.get(key, float("nan")))
            ]
            aggregated[key] = sum(vals) / len(vals) if vals else 0.0
    return aggregated


def save_results(questions, answers, contexts, ground_truths, ragas_result):
    # Convert EvaluationResult to dict to handle NaNs and access safely
    res_dict = aggregate_scores(ragas_result)
    def get_val(x):
        return x if isinstance(x, (int, float)) and not math.isnan(x) else 0.0

    output = {
        "ragas_scores": {
            "faithfulness": get_val(res_dict.get("faithfulness", 0)),
            "context_precision": get_val(res_dict.get("context_precision", 0)),
            "context_recall": get_val(res_dict.get("context_recall", 0))
        },
        "details": [
            {
                "question": q,
                "answer": a,
                "ground_truth": gt,
                "contexts": ctx
            }
            for q, a, gt, ctx in zip(questions, answers, ground_truths, contexts)
        ]
    }

    with open("evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to evaluation_results.json")


def print_summary(ragas_result):
    res_dict = aggregate_scores(ragas_result)
    def get_val(x):
        return x if isinstance(x, (int, float)) and not math.isnan(x) else 0.0

    f = get_val(res_dict.get("faithfulness", 0))
    cp = get_val(res_dict.get("context_precision", 0))
    cr = get_val(res_dict.get("context_recall", 0))

    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Faithfulness:       {f:.2%}")
    print(f"Context Precision:  {cp:.2%}")
    print(f"Context Recall:     {cr:.2%}")
    print("="*40)

    avg = (f + cp + cr) / 3
    print(f"Average:            {avg:.2%}")


def main():
    db = load_vectorstore()
    llm = load_llm()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    chain = build_rag_chain(db, llm)

    eval_llm = ChatMistralAI(model="mistral-small-latest", temperature=0)

    questions, answers, contexts, ground_truths = collect_results(chain, TEST_CASES)
    ragas_result = run_ragas_evaluation(
        questions, answers, contexts, ground_truths, eval_llm, embeddings
    )

    print_summary(ragas_result)
    save_results(questions, answers, contexts, ground_truths, ragas_result)


if __name__ == "__main__":
    main()