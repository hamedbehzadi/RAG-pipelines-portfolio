
def evaluate_retrieval(item: dict, retriever) -> dict:
    documents = retriever.invoke(item["question"])

    retrieved_pages = [
        document.metadata["page"] + 1
        for document in documents
    ]

    relevant_pages = item["relevant_pages"]

    if not relevant_pages:
        hit_at_k = None
    else:
        hit_at_k = int(
            any(page in relevant_pages for page in retrieved_pages)
        )

    return {
        "retrieved_pages": retrieved_pages,
        "hit_at_k": hit_at_k,
    }


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0

    answer_lower = answer.lower()

    matched = sum(
        keyword.lower() in answer_lower
        for keyword in expected_keywords
    )

    return matched / len(expected_keywords)

ABSTENTION_TEXT = (
    "I cannot answer this question from the provided document."
)
def evaluate_abstention(answer: str, should_answer: bool) -> int:
    abstained = ABSTENTION_TEXT.lower() in answer.lower()

    if should_answer:
        return int(not abstained)

    return int(abstained)