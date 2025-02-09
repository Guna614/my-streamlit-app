import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Caching the heavy model so it loads only once
@st.cache_resource
def load_llm_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        device_map="auto",        # Use CPU if GPU is not available
        torch_dtype=torch.float16   # Remove or adjust on CPU-only setups
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

llm_pipeline = load_llm_pipeline()

st.title("Tables and Summaries")

# For demonstration, we create a dummy table object.
# In your actual code, you would extract these tables (with their HTML) from your document.
class DummyTable:
    def __init__(self, html):
        self.metadata = type("meta", (), {"text_as_html": html})

dummy_table_html = """
<table border="1">
  <tr>
    <th>Country</th>
    <th>Population (millions)</th>
  </tr>
  <tr>
    <td>USA</td>
    <td>331</td>
  </tr>
  <tr>
    <td>Canada</td>
    <td>38</td>
  </tr>
  <tr>
    <td>UK</td>
    <td>67</td>
  </tr>
</table>
"""

# Assume 'tables' is a list of such table objects
tables = [DummyTable(dummy_table_html)]  # Add more as needed

# Loop through each table and generate its summary
for i, table in enumerate(tables):
    # Extract HTML and convert to a DataFrame
    table_html = table.metadata.text_as_html
    dfs = pd.read_html(table_html)
    df = dfs[0]  # Assuming the first table is the desired one
    
    # Convert DataFrame to string for summarization
    table_text = df.to_string(index=False)
    
    # Display the table text in the app (using st.code for better formatting)
    st.subheader(f"Table {i+1} Text")
    st.code(table_text)
    
    # Create the prompt to generate a summary
    prompt = (
        "Summarize the following table data:\n\n"
        f"{table_text}\n\n"
        "Provide a concise summary of the key points."
    )
    
    # Generate summary (you may need to adjust max_new_tokens depending on your use case)
    with st.spinner("Generating summary..."):
        response = llm_pipeline(prompt, max_new_tokens=300, num_return_sequences=1)
    summary = response[0]['generated_text']
    
    # Display the generated summary
    st.subheader(f"Summary for Table {i+1}")
    st.write(summary)
    st.markdown("---")
