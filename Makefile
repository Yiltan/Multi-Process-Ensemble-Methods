UNAME=$(shell uname)

ifeq ($(UNAME), Linux)
	PDF_READER='zathura'
else
  ifeq ($(UNAME), Darwin)
	PDF_READER='open'
  endif
endif

SPELL_LANG='en_GB'

OUT_DIR=out
FILE=$(shell ls -p | grep -v / | grep tex)
PDF=$(OUT_DIR)/$(shell ls out/ | grep pdf)

make : $(FILE)
	latexmk -pdf -outdir=$(OUT_DIR) -r .latexmkrc $(FILE)
clean :
	rm $(OUT_DIR)/*

open : $(PDF)
	nohup $(PDF_READER) $(PDF) > /dev/null &

spell :
	for file in $(shell ls tex/*.tex); do \
		aspell -d $(SPELL_LANG) --mode=tex -c $$file; \
	done
