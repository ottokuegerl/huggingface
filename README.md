# my scripts based on Hugging Face libraries
## sentiment analysis --> pipeline("sentiment-analysis")
## text2image         --> engine="stable-diffusion-xl-1024-v1-0"
## audio2text         --> model="openai/whisper-large-v3"

### ############################################
### Setting Up a Virtual Environment
### ############################################

*) create environment:
  
   python -m venv hf_env_wpl

   --> essentially, pyvenv sets up a new directory that contains a few items which
       we can view with the ls command: ls hf_env_wpl

*) activate environment:
   to use this environment, you need to activate it:
   
   cd hf_env_wpl   
   source ./bin/activate
   
   now we see: (hf_env_wpl) [nuc8@nuc8 hf_env_wpl]$ 

   or
   
   source hf_env_wpl/bin/activate
   now we see: (hf_env_wpl) [nuc8@nuc8 AI]$ 


*) now work with python :-)
   python -m pip install --upgrade pip

   pip install --upgrade openai
   pip install requests beautifulsoup4
   
   pip list
