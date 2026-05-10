# 
# Benchmark
# 
get-whisper.cpp:
	git clone https://github.com/ggerganov/whisper.cpp || true

build-whisper.cpp:
	cd whisper.cpp && cmake -B build -DGGML_CUDA=0 && cmake --build build -j --config Release

donwload-whisper-model:
	wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin
	
run-whisper.cpp-cpu:
	/usr/bin/time -v whisper.cpp/build/bin/whisper-cli -m ggml-medium.bin audio/15GdH9-curt.mp3
	
run-ctranslate2-cpu:
	/usr/bin/time -v whisper-ctranslate2 --device cpu --model medium  audio/15GdH9-curt.mp3
 	
    
donwload-llamacpp-model:
	wget https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/google_gemma-3-4b-it-Q4_1.gguf?download=true
	
