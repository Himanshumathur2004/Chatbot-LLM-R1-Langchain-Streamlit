from langchain_core.prompts import PromptTemplate

template =PromptTemplate(
    template = """
    Asuume you are an expert research assistant,

    **Research Paper:** "{paper_name}"
    Generate an explanation based on the following user-selected parameters:
    1.  **Detail Level:** `{detail_level}`
    
        If **Detail Level** is "Simple": Focus only on the core idea and the main outcome. Use an analogy if possible. Avoid all technical jargon.
        If **Detail Level** is "Mid-level": Explain the problem being solved, the core methodology, and the key results. You can include some essential technical terms but explain them briefly.
        If **Detail Level** is "Detailed": Provide a structured breakdown covering the background, methodology, experiments, results, and limitations of the paper.

Please generate the explanation now.
    2.  **Output Length:** `{output_length}`
    
        If **Output Length** is "Small": The entire explanation should be a single, concise paragraph (approx. 100-150 words).
        If **Output Length** is "Medium": The explanation should be a few paragraphs (approx. 250-350 words).   
        If **Output Length** is "Large": The explanation should be comprehensive and well-structured with sections (approx. 500+ words).

    """,
    input_variables=["paper_name", "detail_level", "output_length"],
    validate_template=True,
    )
template.save("research_assistant_prompt_template.json")