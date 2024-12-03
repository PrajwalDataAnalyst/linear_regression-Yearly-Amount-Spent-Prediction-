import streamlit as st
import numpy as np
import pickle

# Load the pre-trained model (make sure 'lm.pkl' is in the same directory as this script)
lm = pickle.load(open('lm.pkl', 'rb'))

# Set up the page layout with a sidebar for file download
st.set_page_config(page_title="Prediction App", layout="wide")

# Sidebar for download
with st.sidebar:
    # Create a download button in the sidebar
    def download_model():
        # Open the pickle file in binary mode
        with open('lm.pkl', 'rb') as f:
            return f.read()

    # Streamlit button for downloading the pickle file
    st.download_button(
        label="Download Model (lm.pkl)",  # Button label
        data=download_model(),  # Load model file data
        file_name="lm.pkl",  # Name of the file to be downloaded
        mime="application/octet-stream"  # MIME type for binary file
    )

# Streamlit main content
st.title("Yearly Amount Spent Prediction")

# Collect user inputs for the features
avg_session_length = st.number_input('Avg. Session Length', min_value=0.0, step=0.1, value=31.926272)
time_on_app = st.number_input('Time on App', min_value=0.0, step=0.1, value=11.109461)
time_on_website = st.number_input('Time on Website', min_value=0.0, step=0.1, value=37.268959)
length_of_membership = st.number_input('Length of Membership', min_value=0.0, step=0.1, value=2.664034)

# Prepare the input data for prediction
input_data = np.array([[avg_session_length, time_on_app, time_on_website, length_of_membership]])

# When the user clicks the "Predict" button
if st.button('Predict'):
    # Make prediction using the pre-trained model
    predicted_value = lm.predict(input_data)

    # Display the result
    st.write(f"Predicted Yearly Amount Spent: ${predicted_value[0]:,.2f}")

# Add your LinkedIn, Website, and GitHub links at the bottom using markdown
st.markdown("""
    ---
<p style="text-align: center;">Find me on <a href="https://prajwaldataanalyst.netlify.app/" target="_blank">Website</a> | Check my <a href="https://github.com/PrajwalDataAnalyst" target="_blank">GitHub</a> | Check my <a href="https://www.linkedin.com/in/prajwal10da" target="_blank">LinkedIn</a></p>
    """, unsafe_allow_html=True)
