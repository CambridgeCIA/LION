SPHINXBUILD ?= sphinx-build
SOURCEDIR = source
BUILDDIR = _build

.PHONY: html clean

html:
	python generate_api_tree.py
	$(SPHINXBUILD) -W --keep-going -b html $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)
