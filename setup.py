import re
from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).parent

init = here / 'src' / 'pytorch_chatbot' / '__init__.py'

with init.open(mode='rt', encoding='utf-8') as fp:
    txt = fp.read()

try:
    pattern = r"^__version__ = '([^']+)'\r?$"
    version = re.findall(pattern, txt, re.M)[0]
except IndexError:
    raise RuntimeError('Unable to determine version.')

requirements = here / 'requirements' / 'prod.txt'

try:
    with requirements.open(mode='rt', encoding='utf-8') as fp:
        install_requires = (
            line.split('#')[0].strip() for line in fp if not
            line.startswith('git+')
        )
        install_requires = list(filter(None, install_requires))
except IndexError:
    raise RuntimeError('requirements/prod.txt is broken')

setup(
    name='pytorch_chatbot',
    version=version,
    install_requires=install_requires,
    python_requires='>=3.6.0',
    include_package_data=True,
    package_data={},
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    entry_points={
        'console_scripts': [
            'train = pytorch_chatbot.train:entrypoint',
            'predict = pytorch_chatbot.predict:entrypoint',
        ],
    },
)
