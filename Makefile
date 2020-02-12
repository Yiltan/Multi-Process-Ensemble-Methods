PDF_READER='zathura'
SPELL_LANG='en_GB'

OUT_DIR=out
FILE=$(shell ls | grep tex)
PDF=$(OUT_DIR)/$(shell ls out/ | grep pdf)

make : $(FILE)
	latexmk -pdf -outdir=$(OUT_DIR) -r .latexmkrc $(FILE)
clean :
	rm $(OUT_DIR)/*

open : $(PDF)
	nohup $(PDF_READER) $(PDF) > /dev/null &

spell :
	for file in $(shell ls src/*.tex); do \
		aspell -d $(SPELL_LANG) --mode=tex -c $$file; \
	done
