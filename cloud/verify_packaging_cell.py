PYPROJECT='[build-system]\nrequires = ["setuptools>=68"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\nname = "khojki-ocr-takehome"\nversion = "0.1.0"\ndescription = "Auditable printed Khojki OCR prototype"\nrequires-python = ">=3.10"\ndependencies = ["torch>=2.2", "numpy>=1.26", "Pillow>=10", "opencv-python-headless>=4.9", "scipy>=1.12", "pypdf>=4", "pypdfium2>=4.30"]\n\n[project.optional-dependencies]\ntrain = ["uharfbuzz>=0.39", "freetype-py>=2.4"]\ntest = ["pytest>=8"]\nbaseline = ["requests>=2.31"]\n\n[project.scripts]\nkhojki-ocr = "khojki_ocr.__main__:main"\n\n[tool.setuptools.packages.find]\nwhere = ["src"]\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
# Verify the documented editable install and console entry point.
(ROOT/'pyproject.toml').write_text(PYPROJECT)
run(['-m','pip','install','-q','--no-deps','-e',ROOT],RES/'round2-package-install.log',120)
run(['-c','import subprocess; subprocess.run(["khojki-ocr","--help"],check=True)'],RES/'round2-console-help.log',30)
print('Editable install and console entry point verified.')
