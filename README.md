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
   - We fine tuned different models with different LoRa settings

 Experiment
r (LoRA HP - Rank)
Alpha (LoRA HP)
LoRA target_modules
ROUGE-1
ROUGE-2
ROUGE-L
ROUGE-Lsum
4bit Quantization
 
 
 
 
 
 
 
Pretrained
-
-
-
22.02
5.79
12.02
15.74
FT #1
8
16
["query","value"]
21.17
5.71
11.53
15.06
Half Precision FP16
 
 
 
 
 
 
 
Pretrained
-
-
-
26.42
5.92
13.32
18.32
FT #2
8
16
["query","value"]
26.3
5.59
13.65
18.64
Pretrained
-
-
-
26.83
4.52
14.53
17.17
FT #3
8
16
["query","value","key","output"]
26.79
4.61
14.56
17.21
Pretrained
-
-
-
28.48
4.34
14.29
16.73
FT #4 w Global Attn
8
16
["query","value","key","output"]
27.86
4.05
14.29
16.54
Pretrained
-
-
-
29.1
3.97
14.96
18.74
FT #5 w Global Attn
16
16
["query","value","key","output"]
28.69
4.87
15/09
18.82
 
 
 
 
 
 
 
 
Pretrained
-
-
-
24.85
6.33
15.57
17.25
FT #Final w Global Attn
16
32
["query","value","key","output"]
25.86
6.74
15.73
17.97



