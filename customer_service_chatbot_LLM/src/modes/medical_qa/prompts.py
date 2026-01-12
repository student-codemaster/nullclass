"""Prompt templates for medical QA (conservative, source-aware).

These templates instruct the LLM to avoid hallucinations, to cite sources
from retrieved documents, and to provide concise, patient-friendly answers.
"""
try:
	from langchain_core.prompts import PromptTemplate
except Exception:
	# Minimal fallback PromptTemplate mimicking interface used in the project
	class PromptTemplate:
		def __init__(self, template: str, input_variables=None):
			self.template = template
			self.input_variables = input_variables or []

		def format(self, **kwargs):
			return self.template.format(**kwargs)

MEDICAL_PROMPT_TEMPLATE = """You are a medical question-answering assistant.
Use only the provided CONTEXT when answering. If the answer cannot be found in the context,
respond with "I don't know" and recommend consulting a qualified healthcare professional.

Provide a concise answer (no more than 200 words) and then list the source(s) you used.

CONTEXT:
{context}

QUESTION:
{question}

Answer:"""

MEDICAL_PROMPT = PromptTemplate(template=MEDICAL_PROMPT_TEMPLATE, input_variables=["context", "question"])
