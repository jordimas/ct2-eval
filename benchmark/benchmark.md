Task: create a python script that allows to benchmark
  - ctranslate2 4.71
  - vLLM
  
How
- Use CPU or CUDA for inference. Default CPU
- Use UV to setup the Python envirometnt
- Model google/gemma-3-1b-it for testing
- Make 10 times the question "What is the capital of France?"
- Show latency, avg, min, max, std for tokens per second
- Use as little code as possible
