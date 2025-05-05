# Sumarizz
Fine tuned models which convert scientific papers to genz slang summaries

### How to use this project
  - The finalized fine-tuned models have been pushed to Hugging face hub for ease of use
  - In the demo directory, there is demo.ipynb file, run it to get gradio link

### Modules to this project

1. PDF parsing
  - Plain pdfs cannot be accepted by the model, to extract text from them, we have a script, that takes the dir to the pdf files, parses them, if the text limit is more than 16k (which is the max size input for LED model), we remove sections in priority: 
         - References, contributors, institute or conference details
  - it write the details to a csv file

2. Dataset scraping
   - To get dataset, we tried scraping multiple sites to achieve high quality data. We used aimodels.fyi/ and https://paperswithcode.com/sota to get latest ai paperies and their summaries. 
   - With aimodels.fyi, we only got 138 papers, but when we were checking ROGUE scores against fine tuned model on this dataset, it did not improve much
  - when we got more papers from paperswithcode, we got 503 rows dataset with 503 papers and ROGUE scores improved. but the quality of summaries were bad. 

3. Preference list generation
  - To fine tune with DPO, a preference dataset with winning and loosing responses are needed for each prompt/input

5. Fine tuning explorations
   - We fine tuned different models with different LoRa settings and picked the best performant one (in terms of highest ROGUE score improvement from base model)
  - After we picked a LoRa model, we fine tuned it further using DPO.

6. Demo
   - To run this project

### How to use demo module
  - The resultant finetuning models have been published to HuggingFace for ease of use
  - The demo folder contains secret.toml (with aws_access_keys), video.mp4 (for video overlay), pdf_parser.py (to parse text from input pdfs), demo.ipynb (which has code to give the gradio link to try out the project)
    
