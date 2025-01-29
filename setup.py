from setuptools import setup, find_packages

setup(
    name='promptTuning',
    version='0.1.0',
    description = 'Prompt Tuning',
    author='Min Woo Park',
    author_email = 'alsdn0110@snu.ac.kr',
    packages = find_packages(where='src'),  # src 폴더 내부의 패키지들을 찾습니다
    package_dir={'': 'src'}, 
    python_requires='>=3.11',
)

