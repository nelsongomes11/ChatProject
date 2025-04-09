# pylint: disable-all
from huggingface_hub import hf_hub_download
import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from streamlit_chat import message
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
import os
from dotenv import load_dotenv
from kokoro import KPipeline
import soundfile as sf
import torch

## Setting up the Delilah prompt

delilah_system_prompt = '''
You will give your responses as a real person called Delilah.
Delilah is a human, not an AI model. Delilah will never ever say that they are an AI.
They are the Tower 3 firewatch, and the user is the new firewatch for Tower 4.
They can only talk through the radio as their towers are kilometers apart.
Delilah and the user will become attached to each other with time.
This happens in Shoshone National Forest, Wyoming.

'''

background_image_path = "assets/background.jpg"

page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://www.camposanto.com/external/wallpapers/Launch_Yellow/FWY_3440x1440.jpg");
background-size: cover;
}

[data-testid="stMainBlockContainer"]{
background-color:rgba(0,0,0,0.6);
margin:1rem;
padding-top:0.5rem;
border-radius:2rem;
}
[data-testid="stHeader"]{
background-color:rgba(0,0,0,0);
}

</style>
"""

footer_hide = """
<style>
.st-emotion-cache-hzygls{
background-color:rgba(0,0,0,0);
}
</style>
"""
st.markdown(footer_hide, unsafe_allow_html=True)

load_dotenv()

api_key=os.getenv("GROQ_API_KEY")

token = os.getenv("HF_TOKEN")
hf_hub_download("hexgrad/Kokoro-82M", "config.json", token=token)



st.markdown(
    """
    <h1 style='text-align: center; color: white;'>Firewatch</h1>
    """,
    unsafe_allow_html=True
)
st.markdown(page_bg_img, unsafe_allow_html=True)


delilah_avatar = "assets/delilah_avatar.png"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": delilah_system_prompt},  
        {"role": "Delilah", "content": "Hello? Anyone there?"}  
    ]

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "Delilah":
        st.chat_message("Delilah", avatar=delilah_avatar).write(msg["content"])

chat_model = ChatGroq(groq_api_key=api_key, model="Gemma2-9b-It")

# Load TTS pipeline
pipeline = KPipeline(lang_code='a')

def text_to_speech(text):
    
    lines = text.split('\n')
    audio_data = []
    
    for line in lines:
        if line.strip():  
            generator = pipeline(line, voice='af_heart')
            for i, (gs, ps, audio) in enumerate(generator):
                audio_data.append(audio)
    
    
    combined_audio = torch.cat(audio_data)
    
    
    audio_file = 'delilah_reply.wav'
    sf.write(audio_file, combined_audio.numpy(), 24000)
    
    return audio_file




if user_input := st.chat_input():
    
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    
    
    
    model_messages = [SystemMessage(content=delilah_system_prompt)]
    
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            model_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "Delilah":
            model_messages.append(AIMessage(content=msg["content"]))
    
    
    delilah_response = chat_model.invoke(model_messages)
    delilah_reply = delilah_response.content
    
    
    st.session_state.messages.append({"role": "Delilah", "content": delilah_reply})
    
    with st.chat_message("Delilah", avatar=delilah_avatar):
        st.write(delilah_reply)
    
    with st.spinner("*loading audio*"):
        audio_file = text_to_speech(delilah_reply)
        st.audio(audio_file,autoplay=True)