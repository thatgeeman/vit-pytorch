clean:
	rm -rf _docs/
	rm -rf _proc/_docs
deps:
	pipenv lock -r > requirements.txt