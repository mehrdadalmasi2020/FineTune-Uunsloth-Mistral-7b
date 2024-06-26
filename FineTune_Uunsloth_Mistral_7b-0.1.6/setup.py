from setuptools import setup, find_packages

setup(
    name='FineTune_Uunsloth_Mistral_7b',
    version='0.1.6',
    packages=find_packages(),
    install_requires=[
        'datasets>=1.6.0',
        'transformers>=4.6.0',
        'pandas>=1.1.0',
        'peft',
        'trl', 
        'bitsandbytes>=0.26.0', 
    ],
    entry_points={
        'console_scripts': [
            'finetune-mistral=FineTune_Uunsloth_Mistral_7b.fine_tune:main',
        ],
    },
    author='Mehrdad Almasi and Demival VASQUES FILHO',
    author_email='Mehrdad.al.2023@gmail.com, demival.vasques@uni.lu ',
    description='A package for fine-tuning Mistral model and generating responses.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    #url='https://github.com/yourusername/FineTune_Uunsloth_Mistral_7b_Mistral',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
