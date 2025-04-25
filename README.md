# sumarizz
Fine tuned models which convert scientific papers to genz slang summaries

Project report: https://docs.google.com/document/d/1EbOwhHTK6D1P-QePqJXpO8Pf36J2ikfYrKhRYx9msqo/edit?tab=t.0


### Modules to this project

1. PDF parsing
  - Plain pdfs cannot be accepted by the model, to extract text from them, we have a script, that takes the dir to the pdf files, parses them, if the text limit is more than 16k (which is the max size input for LED model), we remove sections in priority: 
         - References, contributors, institute or conference details
  - it write the details to a csv file

2. Dataset scraping
   - To get dataset, we tried scraping multiple sites to achieve high quality data. We used aimodels.fyi/ and https://paperswithcode.com/sota to get latest ai paperies and their summaries. 
   - With aimodels.fyi, we only got 138 papers, but when we were checking ROGUE scores against fine tuned model on this dataset, it did not improve much
  - when we got more papers from paperswithcode, we got 503 rows dataset with 503 papers and ROGUE scores improved. but the quality of summaries were bad. 

3. Fine tuning explorations


