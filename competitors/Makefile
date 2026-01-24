# 
# Benchmark
# 
build-whisper.cpp:
	git clone https://github.com/ggerganov/whisper.cpp || true
	cd whisper.cpp && make
	wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin
	
run-whisper.cpp:
	/usr/bin/time -v whisper.cpp/build/bin/whisper-cli -m ggml-medium.bin audio/15GdH9-curt.mp3
	
run-ctranslate2:
	/usr/bin/time -v whisper-ctranslate2 --device cpu --model medium  audio/15GdH9-curt.mp3
 	
    
donwload-llamacpp-model:
	wget https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/google_gemma-3-4b-it-Q4_1.gguf?download=true
	
