PDF=jfk_operational_risk_report
OUTDIR=build

.PHONY: pdf clean

pdf:
	latexmk -xelatex -interaction=nonstopmode -synctex=1 -file-line-error -outdir=$(OUTDIR) $(PDF).tex

clean:
	latexmk -c -outdir=$(OUTDIR)
