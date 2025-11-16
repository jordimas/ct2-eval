download-translation-model:
	wget https://www.softcatala.org/pub/softcatala/opennmt/models/2022-11-22/eng-cat-2024-09-24.zip
	unzip -o eng-cat-2024-09-24.zip
	
download-whisper-model:
	ct2-transformers-converter --model openai/whisper-medium --output_dir whisper-medium-ct2
		
