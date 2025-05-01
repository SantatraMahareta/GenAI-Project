import streamlit as st
import time


def main():
    st.set_page_config("Information Retrieval")
    st.header("Information-Retrieval-System")

    with  st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF",accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                time.sleep(2)

                st.success("Done")




if __name__=="__main__":
    main()