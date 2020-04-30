UNAME=$(shell uname)

ifeq ($(UNAME), Darwin)
	PDF_READER='open'
	SPELL_LANG='uk'
else # ($(UNAME), Linux) # Else it is Linux, unsure on windows.
	PDF_READER='zathura'
	SPELL_LANG='en_GB'
endif

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
		echo $$file; \
		aspell -d $(SPELL_LANG) --mode=tex -c $$file; \
	done
		#aspell -d $(SPELL_LANG) --mode=tex -c $$file; \
