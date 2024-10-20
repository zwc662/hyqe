"""Set of default prompts."""
 

############################################
# Tree
############################################

DEFAULT_SUMMARY_PROMPT_TMPL = (
    "Write a summary of the following. Try to use only the "
    "information provided. "
    "Try to include as many key details as possible.\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    'SUMMARY:"""\n'
)
 
# insert prompts
DEFAULT_INSERT_PROMPT_TMPL = (
    "Context information is below. It is provided in a numbered list "
    "(1 to {num_chunks}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "---------------------\n"
    "Given the context information, here is a new piece of "
    "information: {new_chunk_text}\n"
    "Answer with the number corresponding to the summary that should be updated. "
    "The answer should be the number corresponding to the "
    "summary that is most relevant to the question.\n"
)
 

# # single choice
DEFAULT_QUERY_PROMPT_TMPL = (
    "Some choices are given below. It is provided in a numbered list "
    "(1 to {num_chunks}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return "
    "the choice that is most relevant to the question: '{query_str}'\n"
    "Provide choice in the following format: 'ANSWER: <number>' and explain why "
    "this summary was selected in relation to the question.\n"
)
 

# multiple choice
DEFAULT_QUERY_PROMPT_MULTIPLE_TMPL = (
    "Some choices are given below. It is provided in a numbered "
    "list (1 to {num_chunks}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return the top choices "
    "(no more than {branching_factor}, ranked by most relevant to least) that "
    "are most relevant to the question: '{query_str}'\n"
    "Provide choices in the following format: 'ANSWER: <numbers>' and explain why "
    "these summaries were selected in relation to the question.\n"
)
 


DEFAULT_REFINE_PROMPT_TMPL = (
    "The original query is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the query. "
    "If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
)
 


DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
 
DEFAULT_TREE_SUMMARIZE_TMPL = (
    "Context information from multiple sources is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the information from multiple sources and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)



############################################
# Keyword Table
############################################

DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "Some text is provided below. Given the text, extract up to {max_keywords} "
    "keywords from the text. Avoid stopwords."
    "---------------------\n"
    "{text}\n"
    "---------------------\n"
    "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
)
 


# NOTE: the keyword extraction for queries can be the same as
# the one used to build the index, but here we tune it to see if performance is better.
DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "A question is provided below. Given the question, extract up to {max_keywords} "
    "keywords from the text. Focus on extracting the keywords that we can use "
    "to best lookup answers to the question. Avoid stopwords.\n"
    "---------------------\n"
    "{question}\n"
    "---------------------\n"
    "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
)
 


############################################
# Structured Store
############################################

DEFAULT_SCHEMA_EXTRACT_TMPL = (
    "We wish to extract relevant fields from an unstructured text chunk into "
    "a structured schema. We first provide the unstructured text, and then "
    "we provide the schema that we wish to extract. "
    "-----------text-----------\n"
    "{text}\n"
    "-----------schema-----------\n"
    "{schema}\n"
    "---------------------\n"
    "Given the text and schema, extract the relevant fields from the text in "
    "the following format: "
    "field1: <value>\nfield2: <value>\n...\n\n"
    "If a field is not present in the text, don't include it in the output."
    "If no fields are present in the text, return a blank string.\n"
    "Fields: "
)
 

# NOTE: taken from langchain and adapted
# https://github.com/langchain-ai/langchain/blob/v0.0.303/libs/langchain/langchain/chains/sql_database/prompt.py
DEFAULT_TEXT_TO_SQL_TMPL = (
    "Given an input question, first create a syntactically correct {dialect} "
    "query to run, then look at the results of the query and return the answer. "
    "You can order the results by a relevant column to return the most "
    "interesting examples in the database.\n\n"
    "Never query for all the columns from a specific table, only ask for a "
    "few relevant columns given the question.\n\n"
    "Pay attention to use only the column names that you can see in the schema "
    "description. "
    "Be careful to not query for columns that do not exist. "
    "Pay attention to which column is in which table. "
    "Also, qualify column names with the table name when needed. "
    "You are required to use the following format, each taking one line:\n\n"
    "Question: Question here\n"
    "SQLQuery: SQL Query to run\n"
    "SQLResult: Result of the SQLQuery\n"
    "Answer: Final answer here\n\n"
    "Only use tables listed below.\n"
    "{schema}\n\n"
    "Question: {query_str}\n"
    "SQLQuery: "
)
 

DEFAULT_TEXT_TO_SQL_PGVECTOR_TMPL = """\
Given an input question, first create a syntactically correct {dialect} \
query to run, then look at the results of the query and return the answer. \
You can order the results by a relevant column to return the most \
interesting examples in the database.

Pay attention to use only the column names that you can see in the schema \
description. Be careful to not query for columns that do not exist. \
Pay attention to which column is in which table. Also, qualify column names \
with the table name when needed.

IMPORTANT NOTE: you can use specialized pgvector syntax (`<->`) to do nearest \
neighbors/semantic search to a given vector from an embeddings column in the table. \
The embeddings value for a given row typically represents the semantic meaning of that row. \
The vector represents an embedding representation \
of the question, given below. Do NOT fill in the vector values directly, but rather specify a \
`[query_vector]` placeholder. For instance, some select statement examples below \
(the name of the embeddings column is `embedding`):
SELECT * FROM items ORDER BY embedding <-> '[query_vector]' LIMIT 5;
SELECT * FROM items WHERE id != 1 ORDER BY embedding <-> (SELECT embedding FROM items WHERE id = 1) LIMIT 5;
SELECT * FROM items WHERE embedding <-> '[query_vector]' < 5;

You are required to use the following format, \
each taking one line:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use tables listed below.
{schema}


Question: {query_str}
SQLQuery: \
"""
 


# NOTE: by partially filling schema, we can reduce to a QuestionAnswer prompt
# that we can feed to ur table
DEFAULT_TABLE_CONTEXT_TMPL = (
    "We have provided a table schema below. "
    "---------------------\n"
    "{schema}\n"
    "---------------------\n"
    "We have also provided context information below. "
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and the table schema, "
    "give a response to the following task: {query_str}"
)

DEFAULT_TABLE_CONTEXT_QUERY = (
    "Provide a high-level description of the table, "
    "as well as a description of each column in the table. "
    "Provide answers in the following format:\n"
    "TableDescription: <description>\n"
    "Column1Description: <description>\n"
    "Column2Description: <description>\n"
    "...\n\n"
)
 
# NOTE: by partially filling schema, we can reduce to a refine prompt
# that we can feed to ur table
DEFAULT_REFINE_TABLE_CONTEXT_TMPL = (
    "We have provided a table schema below. "
    "---------------------\n"
    "{schema}\n"
    "---------------------\n"
    "We have also provided some context information below. "
    "{context_msg}\n"
    "---------------------\n"
    "Given the context information and the table schema, "
    "give a response to the following task: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "Given the new context, refine the original answer to better "
    "answer the question. "
    "If the context isn't useful, return the original answer."
)
 

############################################
# Knowledge-Graph Table
############################################

DEFAULT_KG_TRIPLET_EXTRACT_TMPL = (
    "Some text is provided below. Given the text, extract up to "
    "{max_knowledge_triplets} "
    "knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.\n"
    "---------------------\n"
    "Example:"
    "Text: Alice is Bob's mother."
    "Triplets:\n(Alice, is mother of, Bob)\n"
    "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
    "Triplets:\n"
    "(Philz, is, coffee shop)\n"
    "(Philz, founded in, Berkeley)\n"
    "(Philz, founded in, 1982)\n"
    "---------------------\n"
    "Text: {text}\n"
    "Triplets:\n"
)
 

############################################
# HYDE
##############################################

HYDE_TMPL = (
    "Please write a passage to answer the question\n"
    "Try to include as many key details as possible.\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    'Passage:"""\n'
)

 
############################################
# HYQE1
##############################################
HYQE1_TMPL_0 = (
    "Please write a short passage to rephrase the question\n"
    "Try to include the key information only.\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    'Passage:"""\n'
)

HYQE1_TMPL_1 = (
    "Please write a short passage to describe the problem inferred from the following text.\n"
    "Remove any extraneous information irrelevant to the problem.\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    'Passage:"""\n'
)

HYQE1_TMPL_3 = (
"Please write a list of short questions to retrieve answers for the following problem.\n"
"Do not ask for information that does not help resolving the problem.\n"
"Write one question at a line.\n"
"Do not add index at the beginning of the line.\n"
"\n"
"{}\n"
"\n"
"\n"
'Questions:"""\n'
)

HYQE1_TMPL_4 = (
"Which general questions has been answered by the following passage?\n"
"Write one question at a line.\n"
"\n"
"\n"
"Passage:{}\n"
"\n"
"\n"
'Questions:"""\n'
)

HYQE1_TMPL_5 = (
"Write a list of questions that can be answered by the following passage?\n"
"\n"
"\n"
"Passage:{context}\n"
"\n"
"\n"
"Please write different questions in different lines."
"And each question must be written in the same form as '{query}'.\n"
"\n"
"\n"
"Questions:\n"
)

HYQE1_TMPL_6 = (
"Which kinds of questions can be answered based on the following passage within the brackets:\n"
"\n"
"[{}]\n"
"\n"
"\n"
"Questions must be very short, different, and one at a line.\n"
"Do not mention 'according to the passage'.\n"
"Do not mention any information beyond the passage.\n"
"\n"
'Questions:\n'
)

HYQE1_TMPL_7 = (
"Which kinds of questions can be answered based on the following passage within the brackets:\n"
"\n"
"[{}]\n"
"\n"
"\n"
"Questions must be very short, different, and one at a line.\n"
"Do not mention 'according to the passage'.\n"
"Do not mention any information beyond the passage.\n"
"Write no more than 10 questions.\n"
"\n"
'Questions:\n'
)

HYQE1_TMPL_8 = (
"Which kinds of questions can be answered based on the following passage:\n"
"\n"
"```<passage>"
"{}\n"
"</passage>```"
"\n"
"Questions must be very short, different, and one at a line.\n"
"Do not mention 'according to the passage'.\n"
"Do not mention any information beyond the passage.\n" 
"\n"
'Questions:\n'
)

HYQE1_TMPL_9 = (
"Which kinds of questions can be answered based on the following passage:\n"
"\n"
"```<passage>"
"{}\n"
"</passage>```"
"\n"
"Questions must be very short, different, and be written on separate lines.\n"
"If the passage provides no meaningful content, respond with a 'No Content'.\n"
"\n"
)

HYQE1_TMPL_10 = (
"Please provide succinct statements that can be backed up by the following passage.\n"
"\n"
"```<passage>"
"{}\n"
"</passage>```"
"\n"
"Write the statements on separate lines.\n"
"Avoid referencing any identifiers that directly point to the text provided or the statement you generate."
"\n"
)

HYQE1_TMPL_11 = (
"Which kinds of questions can be answered based on the following passage:\n"
"\n"
"```<passage>"
"{}\n"
"</passage>```"
"\n"
"Questions must be very short, different, and be written on separate lines.\n"
"If the passage provides no meaningful content, respond with a 'No Content'.\n"
"\n"
)
HYQE1_TMPL_12 = (
"Which topics could the 'Content' section of the following passage be arguing about.\n"
"If the 'Content' section provides no meaningful argument, respond with a single 'No content'."
"\n"
"```<passage>\n"
"{}\n"
"</passage>```"
"\n"
"Topics are questions.\n"
"Each question must be very short, different, and be written on separate lines.\n"
"Do not mention the passage itself or the author of the passage..\n"
"\n"
)

############################################
# Simple Input
############################################


REPHRASE_TMPL = (
    "Rephrase the following text. "
    "Try to use words as different as possible.\n"
    "\n"
    "\n"
    "{}\n"
    "\n"
    "\n"
)

IRRELEVANT_TMPL = (
    "Add a few irrelevant words before the following texts but try not to change the main subject.\n" 
    "\n"
    "\n"
    "{}\n"
    "\n"
    "\n"
)
 


############################################
# Pandas
############################################

DEFAULT_PANDAS_TMPL = (
    "You are working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)

 

############################################
# JSON Path
############################################

DEFAULT_JSON_PATH_TMPL = (
    "We have provided a JSON schema below:\n"
    "{schema}\n"
    "Given a task, respond with a JSON Path query that "
    "can retrieve data from a JSON value that matches the schema.\n"
    "Task: {query_str}\n"
    "JSONPath: "
)
 

############################################
# Choice Select
############################################

DEFAULT_CHOICE_SELECT_PROMPT_TMPL = (
    "A list of documents is shown below. Each document has a number next to it along "
    "with a summary of the document. A question is also provided. \n"
    "Respond with the numbers of the documents "
    "you should consult to answer the question, in order of relevance, as well \n"
    "as the relevance score. The relevance score is a number from 1-10 based on "
    "how relevant you think the document is to the question.\n"
    "Do not include any documents that are not relevant to the question. \n"
    "Example format: \n"
    "Document 1:\n<summary of document 1>\n\n"
    "Document 2:\n<summary of document 2>\n\n"
    "...\n\n"
    "Document 10:\n<summary of document 10>\n\n"
    "Question: <question>\n"
    "Answer:\n"
    "Doc: 9, Relevance: 7\n"
    "Doc: 3, Relevance: 4\n"
    "Doc: 7, Relevance: 3\n\n"
    "Let's try this now: \n\n"
    "{context_str}\n"
    "Question: {query_str}\n"
    "Answer:\n"
)
 

############################################
# RankGPT Rerank template
############################################

RANKGPT_RERANK_PROMPT_TMPL = (
    "Search Query: {query}. \nRank the {num} passages above "
    "based on their relevance to the search query. The passages "
    "should be listed in descending order using identifiers. "
    "The most relevant passages should be listed first. "
    "The output format should be [] > [], e.g., [1] > [2]. "
    "Only response the ranking results, "
    "do not say any word or explain."
)
 

############################################
# JSONalyze Query Template
############################################

DEFAULT_JSONALYZE_PROMPT_TMPL = (
    "You are given a table named: '{table_name}' with schema, "
    "generate SQLite SQL query to answer the given question.\n"
    "Table schema:\n"
    "{table_schema}\n"
    "Question: {question}\n\n"
    "SQLQuery: "
)

 