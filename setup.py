"""Include deployment resources without duplicating large files in the checkout."""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithResources(build_py):
    def run(self):
        super().run()
        root = Path(__file__).parent
        resources = [
            *sorted((root / "assets").glob("*.ttf")),
            root / "assets/OFL.txt",
            root / "models/printed-khojki.pt",
        ]
        for source in resources:
            destination = (
                Path(self.build_lib) / "khojki_ocr/resources" / source.parent.name
            )
            self.mkpath(str(destination))
            self.copy_file(str(source), str(destination / source.name))


setup(cmdclass={"build_py": BuildWithResources})
