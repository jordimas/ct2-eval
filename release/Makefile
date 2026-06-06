download-translation-model:
	wget https://www.softcatala.org/pub/softcatala/opennmt/models/2022-11-22/eng-cat-2024-09-24.zip
	unzip -o eng-cat-2024-09-24.zip
	
download-whisper-model:
	ct2-transformers-converter --force --model openai/whisper-tiny --output_dir whisper-tiny-ct2 --copy_file normalizer.json
		
run:
	@python3 translate.py --no_verbose
	@python3 generate.py --no_verbose
#	@python3 whisper.py --no_verbose

convert-gemma3:
	ct2-transformers-converter --model google/gemma-3-270m --output_dir gemma-3-270m.ct2
