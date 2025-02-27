import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load model and tokenizer (using MarianMT as an example)
def load_model_and_tokenizer(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

# Translate function
def translate_text(text, model, tokenizer):
    tokenized_text = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
    translated = model.generate(**tokenized_text)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return translated_text

# Streamlit app
def main():
    st.title("Language Translator")

    # Language selection
    st.sidebar.header("Select Languages")
    src_lang = st.sidebar.selectbox("Source Language", ["en", "fr", "de", "es", "it", "hi"])
    tgt_lang = st.sidebar.selectbox("Target Language", ["en", "fr", "de", "es", "it", "hi"])

    # Input text
    text = st.text_area("Enter text to translate:")

    if st.button("Translate"):
        if src_lang == tgt_lang:
            st.warning("Source and target languages cannot be the same.")
        elif text.strip() == "":
            st.warning("Please enter text to translate.")
        else:
            with st.spinner("Translating..."):
                model, tokenizer = load_model_and_tokenizer(src_lang, tgt_lang)
                translation = translate_text(text, model, tokenizer)
                st.success("Translation:")
                st.write(translation)

if __name__ == "__main__":
    main()
