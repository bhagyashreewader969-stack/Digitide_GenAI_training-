import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="Mini LLM Chat", page_icon="🤖")

@st.cache_resource
def load_model():
    model_id = "distilgpt2"  # lightweight model for Streamlit Cloud
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return tokenizer, model

tokenizer, model = load_model()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply = reply[len(prompt):].strip()
        message_placeholder.markdown(reply)
        st.session_state["messages"].append({"role": "assistant", "content": reply})
