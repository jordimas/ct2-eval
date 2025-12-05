download-translation-model:
	wget https://www.softcatala.org/pub/softcatala/opennmt/models/2022-11-22/eng-cat-2024-09-24.zip
	unzip -o eng-cat-2024-09-24.zip
	
download-whisper-model:
	ct2-transformers-converter --model openai/whisper-tiny --output_dir whisper-tiny-ct2
		
run:
	python3 translate.py
	python3 whisper.py		

convert-gemma3:
	ct2-transformers-converter --model google/gemma-3-1b-it --output_dir gemma-3-1b-it.ct2
	

# 
# Benchmark
# 
build-whisper.cpp:
	git clone https://github.com/ggerganov/whisper.cpp || true
	cd whisper.cpp && make
	wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin
	
run-whisper.cpp:
	cd whisper.cpp && ./main -m ggml-medium.bin -f ../audio/15GdH1.mp3
    
donwload-llamacpp-model:
	wget https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/google_gemma-3-4b-it-Q4_1.gguf?download=true
	
